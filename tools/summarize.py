"""Aggregate evaluation results using the paper's Table 1/2 conventions."""

import argparse
import csv
import json
import statistics
from pathlib import Path


DATASETS = [
    "imagenet", "caltech101", "oxford_pets", "stanford_cars", "oxford_flowers",
    "food101", "fgvc_aircraft", "sun397", "dtd", "eurosat", "ucf101"
]
DOMAINS = ["imagenetv2", "imagenet_sketch", "imagenet_a", "imagenet_r"]


def hm(base, new):
    return 2 * base * new / (base + new) if base + new else 0.0


def read(path):
    return json.loads(path.read_text())


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    root = Path(args.root)
    per_seed, summary = [], []
    if args.track == "transfer":
        for dataset in DATASETS + DOMAINS:
            for seed in (1, 2, 3):
                r = read(root / dataset / ("seed" + str(seed)) / "results.json")
                if r["readout"] != "fusion" or r["epoch"] != 5:
                    raise ValueError("Transfer results must use final epoch 5 and fusion")
                per_seed.append(dict(dataset=dataset, seed=seed, n=r["n"],
                                     correct=r["correct"], accuracy=r["accuracy"]))
        for dataset in DATASETS + DOMAINS + ["Avg-10", "Avg-4"]:
            selected = DATASETS[1:] if dataset == "Avg-10" else (
                DOMAINS if dataset == "Avg-4" else [dataset]
            )
            values = [statistics.mean(
                r["accuracy"] for r in per_seed
                if r["seed"] == seed and r["dataset"] in selected
            ) for seed in (1, 2, 3)]
            summary.append(dict(dataset=dataset, accuracy=statistics.mean(values),
                                sd=statistics.stdev(values)))
    else:
        for dataset in DATASETS:
            for seed in (1, 2, 3):
                folder = root / dataset / ("seed" + str(seed))
                b = read(folder / "base/results.json")
                n = read(folder / "new/results.json")
                expected = "fusion" if dataset == "imagenet" else "VT"
                if b["readout"] != expected or n["readout"] != "fusion":
                    raise ValueError("Incorrect Base/New readout for " + dataset)
                if b["epoch"] != 20 or n["epoch"] != 20:
                    raise ValueError("Base-to-new uses final epoch 20")
                per_seed.append(dict(
                    dataset=dataset, seed=seed, base=b["accuracy"], new=n["accuracy"],
                    hm=hm(b["accuracy"], n["accuracy"]),
                    base_correct=b["correct"], base_count=b["n"],
                    new_correct=n["correct"], new_count=n["n"]
                ))
        for dataset in DATASETS + ["Average"]:
            values = []
            for seed in (1, 2, 3):
                rows = [r for r in per_seed if r["seed"] == seed
                        and (dataset == "Average" or r["dataset"] == dataset)]
                b = statistics.mean(r["base"] for r in rows)
                n = statistics.mean(r["new"] for r in rows)
                values.append(dict(base=b, new=n, hm=hm(b, n)))
            b = statistics.mean(r["base"] for r in values)
            n = statistics.mean(r["new"] for r in values)
            summary.append(dict(dataset=dataset, base=b, new=n, hm=hm(b, n),
                                base_sd=statistics.stdev(r["base"] for r in values),
                                new_sd=statistics.stdev(r["new"] for r in values),
                                hm_sd=statistics.stdev(r["hm"] for r in values)))
    output = Path(args.output)
    write(output / "per_seed.csv", per_seed)
    write(output / "summary.csv", summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["transfer", "base2new"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())

