import numpy as np
import pickle
import math
import einops
import itertools
import torch
import torch.nn.functional as F

from torch import nn
from src import utils
from efficientnet_pytorch import EfficientNet

model_urls = {
    "efficientnet_backbone": "https://mever.iti.gr/visloc/efficientnet_backbone.pt",
    "mvmf_head": "https://mever.iti.gr/visloc/mvmf_head.pt",
    "rrm_head": "https://mever.iti.gr/visloc/rrm_head.pt",
}


class VectorizedMixtureVMF(nn.Module):
    def __init__(
        self, num_of_mixtures, initial_mixture_parameters=None, prekappa0=16.0, **kwargs
    ):
        super(VectorizedMixtureVMF, self).__init__()
        self.initial_mixture_parameters = initial_mixture_parameters
        self.prekappa0 = prekappa0
        self.num_of_mixtures = num_of_mixtures

        self.mus = nn.Parameter(
            torch.zeros((2, self.num_of_mixtures), dtype=torch.double)
        )
        self.kappas = nn.Parameter(
            torch.zeros(self.num_of_mixtures, dtype=torch.double)
        )

        if self.initial_mixture_parameters:
            assert len(self.initial_mixture_parameters) == self.num_of_mixtures
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.initial_mus = []
        self.initial_kappas = []
        for lat, lon in self.initial_mixture_parameters:
            self.initial_mus.append((lat, lon))
        self.initial_mus = np.transpose(np.array(self.initial_mus))
        self.mus.copy_(torch.tensor(self.initial_mus, dtype=torch.double))

    def forward(self, x, mixture_weights):
        mus3d = utils.latlon_to_cart_torch(self.mus[0, :], self.mus[1, :]).transpose(
            0, 1
        )

        kappas = self.kappas + self.prekappa0
        kappas = torch.exp(kappas).double()

        probs = 1e-9 + torch.sum(
            (
                kappas
                * torch.exp(kappas * (torch.matmul(x, mus3d.double()) - 1))
                / (2 * math.pi * (1 - torch.exp(-2 * kappas)))
            )
            * mixture_weights,
            axis=1,
        )

        return probs


class MixtureWeights(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(MixtureWeights, self).__init__()

        layers = []
        prev_features = in_features
        for num_features in out_features[:-1]:
            layers.extend(
                [torch.nn.Linear(prev_features, num_features), torch.nn.ReLU()]
            )
            prev_features = num_features
        layers.append(torch.nn.Linear(prev_features, out_features[-1]))
        self.seq_model = torch.nn.Sequential(*layers)

    def forward(self, features):
        return self.seq_model(features)


class MVMFModule(nn.Module):
    def __init__(
        self,
        initial_mu_kappa=None,
        dim_of_features=1792,
        prekappa0=16.0,
        pretrained=False,
        **kwargs
    ):
        super(MVMFModule, self).__init__()

        if initial_mu_kappa is None:
            initial_mu_kappa = pickle.load(open("data/initial_mu_kappa.pkl", "rb"))

        self.initialization = initial_mu_kappa

        num_of_mixtures = len(initial_mu_kappa)

        print("Number of mixtures: {}".format(num_of_mixtures))

        self.mvmf = VectorizedMixtureVMF(
            num_of_mixtures,
            initial_mixture_parameters=initial_mu_kappa,
            prekappa0=prekappa0,
        )

        self.mixture_weights = MixtureWeights(dim_of_features, [num_of_mixtures])

        self.softmax = nn.Softmax(1)

        if pretrained:
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    model_urls["mvmf_head"], map_location="cpu"
                )
            )

    @property
    def mus(self):
        return self.mvmf.mus

    @property
    def kappas(self):
        return self.mvmf.kappas

    @property
    def mixture_weights_params(self):
        return self.mixture_weights.parameters()

    def forward(self, labels, features):
        mixture_weights = self.softmax(self.mixture_weights(features))
        probs = self.mvmf(labels, mixture_weights)

        return mixture_weights, probs

    def calc_loss_and_acc(self, labels_2d, features):
        labels_2d = (labels_2d[:, 0], labels_2d[:, 1])
        labels_3d = utils.latlon_to_cart_torch(*labels_2d)
        mixture_weights, probs = self.forward(labels_3d, features)
        loss = -torch.log2(probs).mean()
        acc = self.calc_acc(labels_2d, mixture_weights)

        return loss, acc

    def calc_acc(self, labels_2d, mixture_weights):
        mus_of_heighest_weighted_density = torch.index_select(
            self.mvmf.mus, 1, torch.argmax(mixture_weights, dim=1)
        ).transpose(0, 1)
        lat, lon = (
            mus_of_heighest_weighted_density[:, 0],
            mus_of_heighest_weighted_density[:, 1],
        )
        dists = utils.haversine_torch(labels_2d[0], labels_2d[1], lat, lon)

        n = dists.shape[0]

        acc = {
            "1km": torch.sum(dists <= 1, dtype=torch.float32) / n,
            "25km": torch.sum(dists <= 25, dtype=torch.float32) / n,
            "200km": torch.sum(dists <= 200, dtype=torch.float32) / n,
            "750km": torch.sum(dists <= 750, dtype=torch.float32) / n,
            "2500km": torch.sum(dists <= 2500, dtype=torch.float32) / n,
        }

        return acc

    @torch.no_grad()
    def run_inference(self, features, return_probs=False):
        mixture_weights = self.softmax(self.mixture_weights(features))
        predictions = torch.index_select(
            self.mvmf.mus, 1, torch.argmax(mixture_weights, dim=1)
        ).transpose(0, 1)

        if return_probs:
            return predictions, mixture_weights
        else:
            return predictions

    def calc_probs(self, labels_2d, mixture_weights):
        labels_2d = (labels_2d[:, 0], labels_2d[:, 1])
        labels_3d = utils.latlon_to_cart_torch(*labels_2d)
        probs = self.mvmf(labels_3d, mixture_weights)
        return probs


class CrossEntropyModule(nn.Module):
    def __init__(self, initial_mus, in_features=1792, **kwargs):
        super(CrossEntropyModule, self).__init__()

        if initial_mus is None:
            initial_mus = pickle.load(open("data/initial_mu_kappa.pkl", "rb"))

        self.initialization = initial_mus
        out_features = [len(initial_mus)]
        self.features_to_logits = MixtureWeights(in_features, out_features)
        self.mus = nn.Parameter(
            torch.tensor([(lat, lon) for lat, lon in initial_mus]), requires_grad=False
        )

        self.softmax = nn.Softmax(1)

    @property
    def classification_layer_params(self):
        return self.features_to_logits.parameters()

    def calc_loss_and_acc(self, labels_2d, features):
        logits = self.features_to_logits(features)
        dists = utils.apply_fn_to_cart_product_general(
            self.mus[None, :, :], labels_2d[:, None, :], utils.haversine_wrapper
        )
        dists = einops.rearrange(dists, "b m t -> b (m t)", t=1, m=self.mus.shape[0])
        labels_index = torch.argmin(dists, dim=1)
        loss = nn.functional.cross_entropy(logits, labels_index)

        predictions = torch.index_select(self.mus, 0, torch.argmax(logits, dim=1))
        acc = self._calc_acc(labels_2d, predictions)

        return loss, acc

    def _calc_acc(self, labels, predictions):
        dists = utils.haversine_torch(
            labels[:, 0], labels[:, 1], predictions[:, 0], predictions[:, 1]
        )

        n = dists.shape[0]

        acc = {
            "1km": torch.sum(dists <= 1, dtype=torch.float32) / n,
            "25km": torch.sum(dists <= 25, dtype=torch.float32) / n,
            "200km": torch.sum(dists <= 200, dtype=torch.float32) / n,
            "750km": torch.sum(dists <= 750, dtype=torch.float32) / n,
            "2500km": torch.sum(dists <= 2500, dtype=torch.float32) / n,
        }

        return acc

    @torch.no_grad()
    def run_inference(self, features, return_probs=False):
        logits = self.softmax(self.features_to_logits(features))
        predictions = torch.index_select(self.mus, 0, torch.argmax(logits, dim=1))

        if return_probs:
            return predictions, logits
        else:
            return predictions


class MultiCrossEntropyModule(nn.Module):
    def __init__(self, partitions, in_features=1792, **kwargs):
        super(MultiCrossEntropyModule, self).__init__()

        self.initialization = partitions

        self.coarse_partition = torch.tensor(partitions[0][0]).to("cuda")
        self.middle_partition = torch.tensor(partitions[1][0]).to("cuda")
        self.middle_parents = torch.tensor(partitions[1][1], dtype=torch.long).to(
            "cuda"
        )
        self.fine_partition = torch.tensor(partitions[2][0]).to("cuda")
        self.fine_parents = torch.tensor(partitions[2][1], dtype=torch.long).to("cuda")

        self.coarse_features_to_logits = MixtureWeights(
            in_features, [self.coarse_partition.shape[0]]
        )
        self.middle_features_to_logits = MixtureWeights(
            in_features, [self.middle_partition.shape[0]]
        )
        self.fine_features_to_logits = MixtureWeights(
            in_features, [self.fine_partition.shape[0]]
        )

        self.softmax = nn.Softmax(1)

    @property
    def classification_layer_params(self):
        return itertools.chain(
            self.coarse_features_to_logits.parameters(),
            self.middle_features_to_logits.parameters(),
            self.fine_features_to_logits.parameters(),
        )

    def calc_fine_probs(self, features):
        fine_logits = self.fine_features_to_logits(features)
        middle_logits = self.middle_features_to_logits(features)
        coarse_logits = self.coarse_features_to_logits(features)

        fine_probs = self.softmax(fine_logits)
        middle_probs = self.softmax(middle_logits)
        coarse_probs = self.softmax(coarse_logits)

        final_probs = (
            fine_probs
            * middle_probs.index_select(dim=1, index=self.fine_parents)
            * coarse_probs.index_select(
                dim=1,
                index=self.middle_parents.index_select(dim=0, index=self.fine_parents),
            )
        )

        return final_probs

    def calc_single_loss(self, labels_2d, logits, partition):
        dists = utils.apply_fn_to_cart_product_general(
            partition[None, :, :],
            labels_2d[:, None, :],
            utils.haversine_wrapper,
        )
        dists = einops.rearrange(dists, "b m t -> b (m t)", t=1, m=partition.shape[0])

        labels_index = torch.argmin(dists, dim=1)
        loss = nn.functional.cross_entropy(logits, labels_index)

        return loss

    def calc_loss_and_acc(self, labels_2d, features):
        fine_logits = self.fine_features_to_logits(features)
        middle_logits = self.middle_features_to_logits(features)
        coarse_logits = self.coarse_features_to_logits(features)

        loss = (
            self.calc_single_loss(labels_2d, coarse_logits, self.coarse_partition)
            + self.calc_single_loss(labels_2d, middle_logits, self.middle_partition)
            + self.calc_single_loss(labels_2d, fine_logits, self.fine_partition)
        )

        fine_probs = self.softmax(fine_logits)
        middle_probs = self.softmax(middle_logits)
        coarse_probs = self.softmax(coarse_logits)

        final_probs = (
            fine_probs
            * middle_probs.index_select(dim=1, index=self.fine_parents)
            * coarse_probs.index_select(
                dim=1,
                index=self.middle_parents.index_select(dim=0, index=self.fine_parents),
            )
        )

        predictions = torch.index_select(
            self.fine_partition, 0, torch.argmax(final_probs, dim=1)
        )
        acc = self._calc_acc(labels_2d, predictions)

        return loss, acc

    def _calc_acc(self, labels, predictions):
        dists = utils.haversine_torch(
            labels[:, 0], labels[:, 1], predictions[:, 0], predictions[:, 1]
        )

        n = dists.shape[0]

        acc = {
            "1km": torch.sum(dists <= 1, dtype=torch.float32) / n,
            "25km": torch.sum(dists <= 25, dtype=torch.float32) / n,
            "200km": torch.sum(dists <= 200, dtype=torch.float32) / n,
            "750km": torch.sum(dists <= 750, dtype=torch.float32) / n,
            "2500km": torch.sum(dists <= 2500, dtype=torch.float32) / n,
        }

        return acc

    @torch.no_grad()
    def run_inference(self, features, return_probs=False):
        probs = self.calc_fine_probs(features)
        predictions = torch.index_select(
            self.fine_partition, 0, torch.argmax(probs, dim=1)
        )

        if return_probs:
            return predictions, probs
        else:
            return predictions


class Residual(nn.Module):
    def __init__(self, dims=1792, hidden_dims=4096, **kwargs):
        super(Residual, self).__init__()

        self.norm = nn.LayerNorm(dims)
        self.fc1 = nn.Linear(dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, dims)

    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x))) + x
        return self.norm(x)


class RetrievalModule(nn.Module):
    def __init__(self, dims=1792, pretrained=False, **kwargs):
        super(RetrievalModule, self).__init__()

        self.dims = dims
        self.norm = nn.LayerNorm(dims)
        self.res = Residual(dims)

        if pretrained:
            self.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    model_urls["rrm_head"], map_location="cpu"
                )
            )
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.res(x)
        return F.normalize(x, p=2, dim=-1)


class Backbone(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(Backbone, self).__init__()

        self.base_cnn = EfficientNet.from_pretrained(
            "efficientnet-b4", include_top=False
        )

        if pretrained:
            self.base_cnn.load_state_dict(
                torch.hub.load_state_dict_from_url(model_urls["efficientnet_backbone"])
            )

    def forward(self, images):
        return self.base_cnn(images).flatten(1)


class GeoLocModel(nn.Module):
    def __init__(self, pretrained=False, **kwargs):
        super(GeoLocModel, self).__init__()

        self.backbone = Backbone(pretrained=pretrained)
        self.cls_head = MVMFModule(pretrained=pretrained)
        self.rrm_head = RetrievalModule(pretrained=pretrained)

    def extract_features(self, images):
        return self.backbone(images)

    def get_prediction(self, features):
        if features.dim() > 2:
            features = self.extract_features(features)
        return self.cls_head.run_inference(features)

    def get_probs(self, features):
        if features.dim() > 2:
            features = self.extract_features(features)
        return self.cls_head.run_inference(features, return_probs=True)[1]

    def get_embeddings(self, features):
        if features.dim() > 2:
            features = self.extract_features(features)
        return self.rrm_head(features)

    def forward(self, images):
        features = self.extract_features(images)
        prediction, cell_probs = self.cls_head.run_inference(
            features, return_probs=True
        )
        embeddings = self.get_embeddings(features)
        return prediction, cell_probs, embeddings
