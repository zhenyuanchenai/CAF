"""Portable sample manifests for the released experiments."""

import json
from pathlib import Path

from dassl.data.data_manager import build_data_loader
from dassl.data.datasets import Datum
from dassl.data.transforms import build_transform


def read_manifest(cfg):
    with open(cfg.DATASET.MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    classnames = manifest["classnames"]
    half = manifest["base_count"]
    split = cfg.DATASET.SUBSAMPLE_CLASSES
    if split == "all":
        indices = list(range(len(classnames)))
    elif split == "base":
        indices = list(range(half))
    elif split == "new":
        indices = list(range(half, len(classnames)))
    else:
        raise ValueError("SUBSAMPLE_CLASSES must be all, base, or new")
    remap = {old: new for new, old in enumerate(indices)}
    root = Path(cfg.DATASET.ROOT).expanduser().resolve()

    def rows(items):
        data = []
        for relative, label in items:
            if label not in remap:
                continue
            relative = Path(relative)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Manifest paths must stay inside DATASET.ROOT")
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            data.append(Datum(
                impath=str(path), label=remap[label], classname=classnames[label]
            ))
        return data

    test = rows(manifest["test"])
    if not test:
        raise ValueError("Empty test split")
    train = []
    if not cfg.TRAINER.CAF.EVAL_ONLY:
        if not cfg.DATASET.TRAIN_MANIFEST:
            raise ValueError("Training with a manifest also needs TRAIN_MANIFEST")
        with open(cfg.DATASET.TRAIN_MANIFEST, encoding="utf-8") as f:
            train_manifest = json.load(f)
        train = rows(train_manifest["train"])
        counts = {label: 0 for label in range(len(indices))}
        for item in train:
            counts[item.label] += 1
        if set(counts.values()) != {cfg.DATASET.NUM_SHOTS}:
            raise ValueError("Training manifest does not match the requested shots")
    return [classnames[i] for i in indices], train, test


def build_manifest_loaders(trainer):
    cfg = trainer.cfg
    classnames, train, test = read_manifest(cfg)
    trainer.classnames = classnames
    trainer.lab2cname = dict(enumerate(classnames))
    trainer.num_classes = len(classnames)
    trainer.num_source_domains = 0
    trainer.train_loader_u = None
    trainer.val_loader = None
    trainer.train_loader_x = None
    if train:
        trainer.train_loader_x = build_data_loader(
            cfg, sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=train, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            tfm=build_transform(cfg, is_train=True), is_train=True
        )
    trainer.test_loader = build_data_loader(
        cfg, sampler_type="SequentialSampler", data_source=test,
        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        tfm=build_transform(cfg, is_train=False), is_train=False
    )
    print("Dataset: {} / {} ({} classes, {} training, {} test images)".format(
        cfg.DATASET.NAME, cfg.DATASET.SUBSAMPLE_CLASSES,
        len(classnames), len(train), len(test)
    ))

