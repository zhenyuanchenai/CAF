# Models, assets and logs

## Download mirrors

Both folders contain the same five release packages. Choose the mirror for your
region and select the exact filename from the package table.

| Region | Mirror | Access code |
|:--|:--|:--:|
| China / 中国 | [Baidu Netdisk / 百度网盘](https://pan.baidu.com/s/1N8m1WrvD2FHx7qnspi7qoA?pwd=j3wc) | `j3wc` |
| Other regions | [Google Drive](https://drive.google.com/drive/folders/13Qb29zphKBotgyRMZR7fAWJJq3V3VTtk?usp=sharing) | — |

| Package | Size | Contents |
|:--|--:|:--|
| `CAF_transfer_models.zip` | 1.38 MB | 3 ImageNet-source models, final epoch 5 |
| `CAF_base2new_models.zip` | 15.17 MB | 33 shared-coefficient CAF models, final epoch 20 |
| `CAF_star_base2new_models.zip` | 15.17 MB | 33 dataset-specific CAF* entries, final epoch 20 |
| `CAF_reproducibility_assets.zip` | 15.59 MB | Exact sample manifests, text banks and prompt initializations |
| `CAF_logs.zip` | 29.72 MB | Original logs and exported evaluation records; also [browse online](../logs/README.md) |

## Extraction

The assets package is required for training and evaluation. For evaluation,
also extract the model package for the experiment you want to reproduce.
Run these commands from the repository root after downloading the archives:

```bash
# Required by every experiment.
unzip CAF_reproducibility_assets.zip

# Extract the model package(s) you want to evaluate.
unzip CAF_transfer_models.zip
unzip CAF_base2new_models.zip
unzip CAF_star_base2new_models.zip
```

The resulting `assets/` and `pretrained/` directories should sit beside
`train.py`. Original logs are already included in the source repository;
`CAF_logs.zip` is an optional standalone copy.

Archive sizes and SHA-256 hashes are recorded in
[release_files.csv](../results/release_files.csv). Each package also includes a
file-level checksum list. For example, after extracting the assets:

```bash
sha256sum -c CAF_reproducibility_assets_SHA256SUMS.txt
```

The official CLIP ViT-B/16 backbone is downloaded automatically on first use,
verified by SHA-256 and cached at `~/.cache/clip/ViT-B-16.pt`.
It is shared across runs, so the model packages contain only the learned CAF
parameters (approximately **0.50 MB per checkpoint**).
See [INSTALL.md](../INSTALL.md#backbone-assets-and-models) for environment details.

## Model layout

```text
pretrained/
├── transfer/imagenet/seed{1,2,3}/CAF/model.pth.tar-5
└── base2new/
    ├── caf/<dataset>/seed{1,2,3}/CAF/model.pth.tar-20
    └── caf_star/<dataset>/seed{1,2,3}/CAF/model.pth.tar-20
```

The downloadable model entries cover **66 distinct trained models**: three
source models plus 63 distinct Base-to-New models. Flowers102 is the same
configuration in CAF and CAF*; its three checkpoints are shared and appear in
both scheme folders for convenient commands.

Each checkpoint contains the 24 learned prompt tensors (122,880 values), final
learned alpha/beta, epoch, loss coefficients and a hash identifying its original
checkpoint. Every learned tensor is copied without numerical modification.
No frozen CLIP backbone, absolute server path or training-job state is embedded.
The [model index](../results/models.csv) lists every released file and both its
released and original SHA-256 values.

These compact files are for **evaluation**. Optimizer state is intentionally not
distributed in the inference packages. Checkpoints written by the training code
also contain the prompt optimizer, cosine scheduler, fusion Adam, gradient
scaler and training EMA, and support training continuation.

## Original logs

Detailed logs are committed under [logs/](../logs/README.md), so they can be read
directly on GitHub without a separate download. Log folders mirror the model hierarchy:

```text
logs/transfer/imagenet/seed1/
logs/base2new/caf/caltech101/seed1/
logs/base2new/caf_star/caltech101/seed1/
```

`metrics_epoch*.jsonl` and the included epoch-history files are original training
records copied byte-for-byte. Each `provenance.json` identifies the final model
and the native log hashes. `evaluation_results.json` exports results from the
completed experiments in a portable format. These exported evaluation records
are not presented as newly generated console logs.

These are the records of the original experiments that produced the published
weights.

## Reproducibility assets

`assets/initialization/` contains only the untrained prompt tensors.
`assets/references/` contains ordered frozen text vectors.
`assets/splits/` contains relative image paths and labels used for training or
metric computation. Test labels never enter the fusion statistics.
The assets package contains no dataset images and no trained model.
