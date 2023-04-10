import os
import h5py
import torch
import math
import pickle
import einops
import collections
import numpy as np

from PIL import Image
from torchvision import transforms
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset


class CustomDict(collections.UserDict):
    """An entry of the dictionary can be accessed if the query key contains the
    entry key as a substring delimited by "_".
    """

    def __missing__(self, key):
        init_key = key
        key = str(key)
        if key.endswith(".jpg"):
            key = key[:-4]
        split = key.split("_", 1)
        if len(split) > 1:
            if split[0] in self.data:
                return self[split[0]]
            else:
                return self[split[1]]
        raise KeyError(init_key)

    def __contains__(self, key):
        key = str(key)
        if key in self.data:
            return True
        else:
            split = key.split("_", 1)
            if len(split) > 1:
                if split[0] in self.data:
                    return True
                else:
                    return split[1] in self
            return False


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, rootDir, transform=None):
        self.images = [
            i for i in os.listdir(rootDir) if i.split(".")[-1] in {"jpg", "jpeg", "png"}
        ]
        self.rootDir = rootDir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(os.path.join(self.rootDir, self.images[idx])).convert(
                "RGB"
            )
            if self.transform is not None:
                image = self.transform(image)
        except:
            print("Error with image: {}".format(self.images[idx]))
            image = torch.zeros(1, 3, 224, 224)
        return self.images[idx], image


def is_contained(img_str, set_of_str):
    """If the set_of_str has an image id 12345, then if img_str contains the id
    between underscores it should be accepted, for example
    bafg_34_12345.jpg in {"12345"} -> True
    """
    img_str = os.path.splitext(img_str)[0]
    if img_str in set_of_str:
        return True
    else:
        split_img_str = img_str.split("_", 1)
        if len(split_img_str) > 1:
            if split_img_str[0] in set_of_str:
                return True
            else:
                return is_contained(split_img_str[1], set_of_str)
        else:
            return False


def skip_imgs_of_ds(ds, imgs_to_skip):
    # we assume that ds is a subset datasets
    imgs = ds.dataset.img_ids
    indices = ds.indices

    imgs_to_skip = set(imgs_to_skip)

    count = 0
    new_indices = []
    for idx in indices:
        if is_contained(imgs[idx], imgs_to_skip):
            print("Skipping image {} with index {}".format(imgs[idx], idx))
            count += 1
            continue
        else:
            new_indices.append(idx)

    ds.indices = new_indices
    return count


class Preprocessing:
    def __init__(self, mode, backbone="resnet"):
        if backbone == "efficientnet":
            image_size = 300
        else:
            image_size = 224

        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if mode == "training":
            self.preprocess_fn = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalization,
                ]
            )
        elif mode == "validation":
            self.preprocess_fn = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalization,
                ]
            )
        elif mode == "inference":
            if backbone == "efficientnet":
                self.preprocess_fn = transforms.Compose(
                    [
                        transforms.Resize(image_size),
                        transforms.ToTensor(),
                        normalization,
                    ]
                )
            else:
                self.preprocess_fn = transforms.Compose(
                    [
                        transforms.FiveCrop(image_size),
                        transforms.Lambda(
                            lambda crops: torch.stack(
                                [
                                    normalization(transforms.ToTensor()(crop))
                                    for crop in crops[2:]
                                ]
                            )
                        ),
                    ]
                )
        else:
            raise Exception("Unknown preprocessing mode")

    def __call__(self, img_path_list):
        return self.preprocess_fn(img_path_list)


def is_metric_strictly_worse(metric, iterable_of_metrics):
    list_of_metrics = list(iterable_of_metrics)
    if not metric:
        if list_of_metrics:
            return True
        else:
            return False
    else:
        better_metrics = filter(lambda x: x[0] > metric[0], list_of_metrics)
        better_metrics = map(lambda x: x[1:], better_metrics)
        return is_metric_strictly_worse(metric[1:], better_metrics)


def remove_checkpoints_if_better_exists(checkpoints_dir, parser_fn=None):
    assert os.path.isdir(checkpoints_dir), "Should be given checkpoints directory."
    if not parser_fn:
        parser_fn = lambda x: list(map(float, x.split("_")[-5:]))
    file_names = []
    metrics = []
    for file_name in os.listdir(checkpoints_dir):
        file_names.append(file_name)
        metrics.append(parser_fn(file_name))
    file_name_to_remove = []
    for i_metric, metric in enumerate(metrics):
        if is_metric_strictly_worse(
            metric, metrics[:i_metric] + metrics[i_metric + 1 :]
        ):
            file_name_to_remove.append(file_names[i_metric])
    for file_name in file_name_to_remove:
        full_path = os.path.join(checkpoints_dir, file_name)
        print("Removed {}".format(file_name))
        os.remove(full_path)


def apply_fn_to_cart_product(x, y, fn):
    """x is a [B, K, T] tensor and y is a [B, N, T] tensor. fn is a function
    with signature [-1, T], [-1, T] -> [-1,1]. The apply_fn_to_cart_product function
    applies fn to all the B*K*N pairs and returns a [B, K, N] tensor

    Arguments:
        x {[type]} -- [description]
        y {[type]} -- [description]
        fn {function} -- [description]
    """
    b, k, t = x.shape
    by, n, ty = y.shape
    assert b == by and t == ty

    x = x[:, None, :, :].repeat_interleave(n, dim=1)
    y = y[:, :, None, :].repeat_interleave(k, dim=2)

    x = einops.rearrange(x, "b n k t -> (b n k) t")
    y = einops.rearrange(y, "b n k t -> (b n k) t")

    res = fn(x, y)
    res = einops.rearrange(res, "(b n k) -> b k n", b=b, k=k, n=n)

    return res


def prediction_density(idx, probs, coords, scales=[1, 25, 200, 750, 2500]):
    """
    Computes prediction density

    - probs: numpy array containing cell probabilities (shape = (C, ))
    - coords: numpy matrix containing cell coordinates (shape = (C, 2))
    - scales: python list with granularity scales to compute prediction density

    returns a dict with a prediction density estimation for each scale
    """
    pd = dict.fromkeys(scales)

    dist = haversine(coords[idx], coords.T)

    for scale in scales:
        cells = dist <= scale  # select all cells within radius scale
        pd[scale] = probs[cells].sum()  # sum of their probabilities

    return pd


def apply_fn_to_cart_product_general(x, y, fn):
    """Apply fn to cartesian product of x and y

    x and y should match on all non singleton dimensions and match exactly on the last
    dimension. If the last dimension has size t then fn: (-1, t), (-1,t) -> (-1, s)
    For example if x has shape (1,3,4,5) and y (3,1,1,5) the output will have shape
    (3,3,4,s), where s is the output size of fn.

    Arguments:
        x {[type]} -- [description]
        y {[type]} -- [description]
        fn {function} -- [description]
    """
    x_shape = x.shape
    y_shape = y.shape
    max_shape = [max(z) for z in zip(x_shape, y_shape)]
    x = x.expand(max_shape)
    y = y.expand(max_shape)

    res = fn(x.reshape(-1, max_shape[-1]), y.reshape(-1, max_shape[-1]))
    new_shape = list(max_shape)
    if len(res.shape) == 1:
        new_shape[-1] = 1
    else:
        new_shape[-1] = res.shape[1]
    res = res.reshape(new_shape)

    return res


def haversine(X, Y):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [X[1], X[0], Y[1], Y[0]])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def haversine_torch(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    lat1 = lat1 / 180 * math.pi
    lon1 = lon1 / 180 * math.pi
    lat2 = lat2 / 180 * math.pi
    lon2 = lon2 / 180 * math.pi

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        torch.sin(dlat / 2.0) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
    )

    c = 2 * torch.asin(torch.sqrt(a))

    km = 6371 * c

    return km


def find_label(lat_lon, mus):
    dists = apply_fn_to_cart_product_general(
        mus[None, :, :], lat_lon[:, None, :], haversine_wrapper
    )
    dists = einops.rearrange(dists, "b m t -> b (m t)", t=1, m=mus.shape[0])
    label = torch.argmin(dists, dim=1).cpu().numpy()
    return label.tolist()


def spatial_clustering(candidates, similarities, radius=1, a=0):
    clusters = DBSCAN(eps=radius, min_samples=1, metric=haversine).fit(candidates)
    images_latlon = list(
        sorted(
            [candidates[clusters.labels_ == c] for c in np.unique(clusters.labels_)],
            key=len,
            reverse=True,
        )
    )[0]
    similarities = list(
        sorted(
            [similarities[clusters.labels_ == c] for c in np.unique(clusters.labels_)],
            key=len,
            reverse=True,
        )
    )[0]
    pr = np.sum(
        [
            s**a * np.array(latlon_to_cart(*j))
            for s, j in zip(similarities, images_latlon)
        ],
        axis=0,
    ) / np.sum(similarities**a)
    pr = cart_to_latlon(*pr)
    return pr


def single_vmf(x, mu, kappa):
    return (
        kappa
        * math.exp(kappa * (x[0] * mu[0] + x[1] * mu[1] + x[2] * mu[2] - 1))
        / (2 * math.pi * (1 - math.exp(-2 * kappa)))
    )


def vMF(y, mu, kappa):
    return (
        kappa
        * torch.exp(kappa * (torch.sum(mu * y, axis=1) - 1))
        / (2 * math.pi * (1 - torch.exp(-2 * kappa)))
    )


def vectorized_mvmf_torch(x, mus, kappas, mixture_weights):
    """Vactorized Calculation of Mixture of von Mishes-Fisher

    Arguments:
        x {[type]} -- [B, 3] cartesian coordinates
        mus {[type]} -- [2, M] lat lon mean direction coordinates
        kappas {[type]} -- [M] concentration parameters
        mixture_weights {[type]} -- [B, M] mixture weights

    Returns:
        [B] mvmf density
    """
    mus3d = latlon_to_cart_torch(mus[0, :], mus[1, :]).transpose(0, 1)
    return torch.sum(
        (
            kappas
            * torch.exp(kappas * (torch.matmul(x, mus3d) - 1))
            / (2 * math.pi * (1 - torch.exp(-2 * kappas)))
        )
        * mixture_weights,
        axis=1,
    )


def latlon_to_cart(lat, lon):
    # https://vvvv.org/blog/polar-spherical-and-geographic-coordinates

    lat_rad = lat / 180 * math.pi
    lon_rad = lon / 180 * math.pi

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return [x, y, z]


def cart_to_latlon(x, y, z):
    lat_rad = math.asin(z)
    lon_rad = math.atan2(y, x)

    lat = lat_rad / math.pi * 180
    lon = lon_rad / math.pi * 180

    return lat, lon


def latlon_to_cart_torch(lat, lon):
    lat = lat / 180 * math.pi
    lon = lon / 180 * math.pi

    return torch.stack(
        (
            torch.cos(lat) * torch.cos(lon),
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat),
        ),
        dim=1,
    )


def cart_to_latlon_torch(t):
    lat_rad = torch.asin(t[:, 2])
    lon_rad = torch.atan2(t[:, 1], t[:, 0])

    lat = lat_rad / math.pi * 180
    lon = lon_rad / math.pi * 180

    return lat, lon


def mk3d(lat, lon):
    lats_rad = lat  # /180*math.pi
    lons_rad = lon  # /180*math.pi

    x = np.sin(lats_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.cos(lats_rad) * np.cos(lons_rad)

    return [x, y, z]


def mk3d_torch(lat, lon):
    lat = lat / 180 * math.pi
    lon = lon / 180 * math.pi

    return torch.stack(
        (
            torch.sin(lat),
            torch.cos(lat) * torch.sin(lon),
            torch.cos(lat) * torch.cos(lon),
        ),
        axis=1,
    )


def spherical2cart_torch(s):
    azimuth, elevation = s[:, 0], s[:, 1]
    x = torch.cos(elevation) * torch.cos(azimuth)
    y = torch.cos(elevation) * torch.sin(azimuth)
    z = torch.sin(elevation)

    return torch.stack([x, y, z], dim=1)


def cart2spherical_torch(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    azimuth = torch.atan2(y, x)
    elevation = torch.atan2(z, torch.sqrt(x**2 + y**2))

    return torch.stack([azimuth, elevation], dim=1)


def haversine_wrapper(x, y):
    return haversine_torch(x[:, 0], x[:, 1], y[:, 0], y[:, 1])


def calc_best_hv(mus, labels):
    """ "Haversine distance for optimal assignment

    Arguments:
        mus -- [M, 2] in degrees
        labels -- [B, 2]
    """
    b, t = labels.shape
    mus = mus[None, :, :].repeat_interleave(b, dim=0)
    dists = apply_fn_to_cart_product(mus, labels[:, None, :], haversine_wrapper)
    dists = torch.squeeze(dists)

    best_dists, _ = torch.min(dists, dim=1)

    return best_dists


def get_gps_coords_and_img_ids(ds):
    labels = ds.dataset.labels
    img_ids = ds.dataset.img_ids
    indices = ds.indices

    gps_coords = [labels[idx][[0, 1]] for idx in indices]
    img_ids = [img_ids[idx] for idx in indices]

    return gps_coords, img_ids


def get_initial_mu_kappa(
    ds, tmin, cond_to_term_fn, stop_splitting_after_cell_level, output=None
):
    print("Calculating gps coordinates and img ids...", flush=True)
    gps_coords, imgs = get_gps_coords_and_img_ids(ds)

    import queue

    import s2sphere as s2

    initial_cells = {
        s2.CellId.from_face_pos_level(0, 0, 0): [],
        s2.CellId.from_face_pos_level(1, 0, 0): [],
        s2.CellId.from_face_pos_level(2, 0, 0): [],
        s2.CellId.from_face_pos_level(3, 0, 0): [],
        s2.CellId.from_face_pos_level(4, 0, 0): [],
        s2.CellId.from_face_pos_level(5, 0, 0): [],
    }

    print("Splitting gps coordinates to initial cells...")
    for idx, gps in enumerate(gps_coords):
        gps_cell = s2.CellId.from_lat_lng(s2.LatLng.from_degrees(gps[0], gps[1]))
        gps_cell.img = imgs[idx]
        for cell in initial_cells:
            if cell.intersects(gps_cell):
                initial_cells[cell].append(gps_cell)
                break

    print("Creating priority queue...")
    priority_list = [
        [-len(initial_cells[cell]), cell, initial_cells[cell], cell]
        for cell in initial_cells
    ]
    s2cells = queue.PriorityQueue()
    for item in priority_list:
        s2cells.put(item)

    print("Starting iterative partitioning")
    stage = 0
    iteration = 0
    partitions = []
    imgs_to_skip = set()
    while True:
        iteration += 1
        maxsize = max(
            [
                len(gps_cells)
                for (priority, cell, gps_cells, parent) in s2cells.queue
                if not cell.is_leaf()
            ]
        )
        if iteration % 1000 == 0:
            print(
                "iteration: {}, maxsize: {}, number_of_classes {}".format(
                    iteration, maxsize, s2cells.qsize()
                )
            )
        if cond_to_term_fn[stage](maxsize, s2cells.qsize()):
            print(
                "Stage {} terminated at iteration: {}, maxsize: {}, number_of_classes {}".format(
                    stage, iteration, maxsize, s2cells.qsize()
                )
            )
            inital_mus_kappas = []
            parents = []
            for i_item, item in enumerate(s2cells.queue):
                priority, cell, gps_cells, parent = item
                item[3] = cell
                if len(gps_cells) == 0:
                    continue
                elif len(gps_cells) < tmin:
                    imgs_to_skip.update([gps_cell.img for gps_cell in gps_cells])
                else:
                    lats, lons = [], []
                    for gps_cell in gps_cells:
                        latlng = gps_cell.to_lat_lng()
                        lats.append(latlng.lat().degrees)
                        lons.append(latlng.lng().degrees)
                    inital_mus_kappas.append((np.mean(lats), np.mean(lons), cell))
                    parents.append(parent)
            if stage > 0:
                partitions.append((inital_mus_kappas, parents))
            else:
                partitions.append((inital_mus_kappas,))
            stage += 1
            if stage == len(cond_to_term_fn):
                # if len(cond_to_term_fn) == 1:
                #     partitions = partitions[0]
                break
                # return partitions, imgs_to_skip

        else:
            (neglen, cell, gps_cells, parent) = s2cells.get()
            try:
                children = dict([(c, []) for c in cell.children()])
                if cell.level() >= stop_splitting_after_cell_level:
                    s2cells.put([0, cell, gps_cells, parent])
                    continue
                for gps_cell in gps_cells:
                    for child in children:
                        if child.intersects(gps_cell):
                            children[child].append(gps_cell)
                            break
                for child in children:
                    s2cells.put([-len(children[child]), child, children[child], parent])
            except AssertionError:
                print("cell=", cell, " is a leaf; len(gps_cells)=", len(gps_cells))
                s2cells.put([1, cell, gps_cells, parent])

    # postprocessing
    parents = [[]]
    for stage in range(1, len(cond_to_term_fn)):
        parents.append([])
        for parent_cell in partitions[stage][1]:
            for i, cell in enumerate(partitions[stage - 1][0]):
                if parent_cell == cell[2]:
                    parents[stage].append(i)
    if output:
        with open(output, "wb") as f:
            pickle.dump(partitions, f)
    new_partitions = [
        ([(lat, lon) for lat, lon, cell in partitions[i][0]], parents[i])
        for i in range(len(partitions))
    ]
    if len(cond_to_term_fn) == 1:
        new_partitions = new_partitions[0][0]
    return new_partitions, imgs_to_skip
