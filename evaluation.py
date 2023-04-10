import argparse
import h5py
import torch
import pickle
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from src import utils, modules


def pprint_results(dists, evaluation_radius):
    for r in evaluation_radius:
        print(
            "Acc@{}{}:\t{:.1f}%".format(
                int(r) if r >= 1 else int(r * 1000),
                "km" if r >= 1 else "m",
                np.mean((dists < r)) * 100,
            )
        )
    print(
        "Median Error: {:.1f}km".format(np.median(dists)),
    )


@torch.no_grad()
def evaluate(args):
    print("\nEvaluation Arguments")
    print("-" * 30)
    for k, v in sorted(dict(vars(args)).items()):
        print("{}: {}".format(k, v))

    print("\n> loading model")
    model = modules.GeoLocModel(pretrained=True).eval()
    if not args.use_cpu:
        model = model.to(args.gpu)

    mus = model.cls_head.mus.t().cpu().numpy()
    cells_assignments = pickle.load(open("data/cells_assignments.pkl", "rb"))

    transform = utils.Preprocessing("inference", backbone="efficientnet")

    print("\n> loading evaluation dataset")
    dataset = utils.ImageDataset(args.image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8)
    with open("data/metadata_{}.pkl".format(args.dataset), "rb") as f:
        metadata = pickle.load(f)
        metadata = utils.CustomDict(metadata)
    print("Total images:", len(dataset))

    print("\n> loading background collection")
    with h5py.File(args.background, "r", driver=None) as hdf5_file:
        back_col_emb = hdf5_file["features"][:, :].T.astype(np.float32)
        back_col_cells = hdf5_file["labels"][:, :]
    print("Background collection features:", back_col_emb.T.shape)
    print("Background collection cells:", back_col_cells.shape)

    dists, localizable = [], []
    print("\n> predict image locations")
    for image_ids, tensors in tqdm(dataloader):
        try:
            if not args.use_cpu:
                tensors = tensors.to(args.gpu)
            prediction, cell_probs, embeddings = model(tensors)
            embeddings = embeddings.cpu().numpy()
            cell_probs = cell_probs[0].cpu().numpy()
            max_cell = np.argmax(cell_probs)

            gt = metadata[image_ids[0]]
            gt = [float(gt["lat"]), float(gt["lon"])]

            # Search within Cell scheme
            if max_cell in cells_assignments:
                idxs = np.array(list(cells_assignments[max_cell]))
                sims = np.dot(embeddings, back_col_emb[:, idxs])[0]

                NNs = np.argsort(-sims)[: args.top_k]
                sims = sims[NNs]
                candidates = back_col_cells[idxs[NNs]]

                pr = utils.spatial_clustering(candidates, sims, radius=args.eps, a=0)
            else:
                pr = prediction[0].cpu().numpy()

            dist = utils.haversine(gt, pr)

            conf = utils.prediction_density(
                max_cell, cell_probs, mus, scales=args.eval_radius
            )

            dists.append(dist)
            localizable.append(conf[args.conf_scale] > args.conf_thres)
        except:
            print("Error with image: {}".format(image_ids[0]))

    dists = np.array(dists)
    localizable = np.array(localizable)

    print("\nGeolocation results")
    print("=" * 20)

    print("Total predictions:", len(dists))
    print("-" * 20)
    pprint_results(dists, args.eval_radius)

    print(
        "\nLocalizable images: {} ({:.1f}%)".format(
            np.sum(localizable), np.mean(localizable) * 100
        )
    )
    print("-" * 20)
    pprint_results(dists[localizable == 1], args.eval_radius)

    print(
        "\nNon-localizable images: {} ({:.1f}%)".format(
            np.sum(1 - localizable), np.mean(1 - localizable) * 100
        )
    )
    print("-" * 20)
    pprint_results(dists[localizable == 0], args.eval_radius)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--image_folder", type=str, required=True)
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["im2gps", "im2gps3k", "yfcc4k", "yfcc25k"],
        required=True,
    )
    parser.add_argument(
        "-s",
        "--eval_radius",
        default=[0.1, 1, 25, 200, 750, 2500],
        type=lambda x: list(map(float, x.split(","))),
    )
    parser.add_argument(
        "-b", "--background", type=str, default="./back_coll_features.hdf5"
    )
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-k", "--top_k", type=int, default=10)
    parser.add_argument("-e", "--eps", type=float, default=1.0)
    parser.add_argument("-cpu", "--use_cpu", action="store_true")
    parser.add_argument("-ct", "--conf_thres", type=float, default=0.2)
    parser.add_argument("-cs", "--conf_scale", type=int, default=25)
    args = parser.parse_args()

    evaluate(args)
