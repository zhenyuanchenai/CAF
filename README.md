<h1 align="center">CAF</h1>

<p align="center">
  <strong>When to Trust Adaptation:<br>
  Contextual Fusion for Transferable Vision-Language Prompting</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.11824">Paper (arXiv)</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#model-zoo">Models &amp; Assets</a> &nbsp;·&nbsp;
  <a href="#imagenet-to-target-transfer">Transfer</a> &nbsp;·&nbsp;
  <a href="#base-to-new-generalization">Base-to-New</a> &nbsp;·&nbsp;
  <a href="#training-logs">Training Logs</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="docs/method.svg" alt="CAF: contextual fusion of four frozen and prompted vision-language routes" width="100%">
</p>

Official implementation of **CAF (Contextual Adaptive Fusion)**. CAF combines
four frozen/prompted vision-language routes using prediction entropy and
unlabeled collection context. Only the visual/text prompts and two fusion
scalars are learned; the CLIP backbone remains frozen.

## Installation

1. Set up the environment following [INSTALL.md](INSTALL.md).
2. Prepare the dataset images following [DATASETS.md](DATASETS.md).
3. Download the resources below and extract them into the repository root.

All commands run from the repository root. The code follows the
[PromptSRC](https://github.com/muzairkhattak/PromptSRC) organization, with a
`TrainerX` trainer, YACS configurations and separate training/evaluation scripts.

## Model zoo

The five release packages are available from either mirror. Open the folder
and select the package by its filename.

| Region | Download mirror | Access code |
|:--|:--|:--:|
| China / 中国 | [Baidu Netdisk / 百度网盘](https://pan.baidu.com/s/1N8m1WrvD2FHx7qnspi7qoA?pwd=j3wc) | `j3wc` |
| Other regions | [Google Drive](https://drive.google.com/drive/folders/13Qb29zphKBotgyRMZR7fAWJJq3V3VTtk?usp=sharing) | — |

| Package | Contents |
|:--|:--|
| `CAF_transfer_models.zip` | Three ImageNet-source models, seeds 1/2/3, epoch 5 |
| `CAF_base2new_models.zip` | CAF: shared loss coefficients, 11 datasets × 3 seeds, epoch 20 |
| `CAF_star_base2new_models.zip` | CAF*: dataset-specific coefficients, 11 datasets × 3 seeds, epoch 20 |
| `CAF_reproducibility_assets.zip` | Exact splits, class order, prompt initializations and frozen text banks |
| `CAF_logs.zip` | Original detailed training logs and evaluation records |

The **assets package is required for both training and evaluation**.
To evaluate released models, also download the corresponding model package.
Extract packages so that `assets/` and `pretrained/` sit beside `train.py`.

Each model checkpoint is approximately **0.50 MB**: it contains the learned
prompts and fusion scalars. The official **CLIP ViT-B/16 backbone is downloaded
automatically** on first use and cached locally. Original logs can also be
[browsed directly on GitHub](logs/README.md).

[Package details and extraction](docs/MODEL_ZOO.md) ·
[Model index](results/models.csv) ·
[SHA-256 checksums](results/release_files.csv)

## ImageNet-to-target transfer

Train on all 1,000 ImageNet classes with **16 images per class** for **5 epochs**,
using `(lambda_margin, lambda_text) = (1, 40)`. The same source model is evaluated
on **15 sets in total**: ImageNet validation, ten cross-dataset targets and four
ImageNet domain variants.

### Train

The reproduction script runs seeds 1, 2 and 3.

```bash
DATA=/path/to/datasets
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/transfer.sh "$DATA" train
```

### Evaluate

```bash
DATA=/path/to/datasets

# Evaluate the released models on all 15 sets.
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/transfer.sh "$DATA" test pretrained

# Evaluate models trained by the command above.
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/transfer.sh "$DATA" test output
```

Loss coefficients are listed in [scripts/reproduce/transfer.sh](scripts/reproduce/transfer.sh).
See [single-target evaluation and protocol details](docs/EXPERIMENTS.md#transfer).

### Results

| ImageNet | Avg-10 | Avg-4 |
|:--:|:--:|:--:|
| **72.26 ± 0.03** | **69.96 ± 0.14** | **61.36 ± 0.08** |

<details>
<summary><strong>Full results on all 15 evaluation sets</strong></summary>

| Dataset | Accuracy (%) |
|:--|--:|
| ImageNet | 72.26 ± 0.03 |
| Caltech101 | 94.10 ± 0.12 |
| OxfordPets | 93.11 ± 0.15 |
| StanfordCars | 68.39 ± 0.07 |
| Flowers102 | 71.88 ± 0.26 |
| Food101 | 87.31 ± 0.08 |
| FGVCAircraft | 29.82 ± 0.33 |
| SUN397 | 70.65 ± 0.17 |
| DTD | 54.83 ± 0.22 |
| EuroSAT | 56.36 ± 1.35 |
| UCF101 | 73.09 ± 0.12 |
| ImageNetV2 | 65.63 ± 0.15 |
| ImageNet-Sketch | 52.21 ± 0.11 |
| ImageNet-A | 49.63 ± 0.26 |
| ImageNet-R | 77.97 ± 0.08 |
| Avg-10 | 69.96 ± 0.14 |
| Avg-4 | 61.36 ± 0.08 |

</details>

Accuracies are percentages, reported as mean ± sample standard deviation over
three seeds. Avg-10 and Avg-4 average datasets equally and exclude source
ImageNet. [Per-seed CSV](results/transfer_per_seed.csv) ·
[Summary CSV](results/transfer_summary.csv)

## Base-to-new generalization

Both variants use **16-shot Base training**, **seeds 1/2/3** and **epoch 20**:

| Variant | Training loss coefficients |
|:--|:--|
| **CAF** | Shared `(lambda_margin, lambda_text) = (4, 2.5)` on all eleven datasets |
| **CAF*** | Dataset-specific coefficients listed in the [reproduction script](scripts/reproduce/base2new.sh) |

### Train

```bash
DATA=/path/to/datasets

# Shared-coefficient CAF.
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/base2new.sh "$DATA" caf train

# Dataset-specific CAF*.
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/base2new.sh "$DATA" caf_star train
```

### Evaluate

```bash
DATA=/path/to/datasets

# Evaluate the released models.
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/base2new.sh "$DATA" caf test pretrained
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/base2new.sh "$DATA" caf_star test pretrained
```

Replace `pretrained` with `output` to evaluate newly trained models.

**Table 2 readout:** fusion on ImageNet Base and every New split; the VT route
on the other ten Base splits. The reproduction script applies these choices
explicitly. [Full configurations and protocol](docs/EXPERIMENTS.md#base-to-new)

### Results

| Dataset | CAF Base | CAF New | CAF HM | CAF* Base | CAF* New | CAF* HM |
|:--|--:|--:|--:|--:|--:|--:|
| Average | 85.78 | 79.52 | 82.53 | 86.09 | 79.77 | 82.81 |
| ImageNet | 78.36 | 73.15 | 75.67 | 78.67 | 73.27 | 75.87 |
| Caltech101 | 98.77 | 95.89 | 97.31 | 98.75 | 96.14 | 97.43 |
| OxfordPets | 96.10 | 98.02 | 97.05 | 96.72 | 97.97 | 97.34 |
| StanfordCars | 82.95 | 76.96 | 79.84 | 83.70 | 76.76 | 80.08 |
| Flowers102 | 98.16 | 76.83 | 86.20 | 98.16 | 76.83 | 86.20 |
| Food101 | 90.45 | 92.53 | 91.48 | 91.02 | 92.50 | 91.75 |
| FGVCAircraft | 45.70 | 43.01 | 44.31 | 46.18 | 44.63 | 45.39 |
| SUN397 | 83.36 | 81.35 | 82.34 | 83.41 | 81.70 | 82.55 |
| DTD | 85.30 | 69.97 | 76.88 | 84.53 | 71.26 | 77.33 |
| EuroSAT | 96.13 | 83.61 | 89.43 | 97.14 | 83.19 | 89.63 |
| UCF101 | 88.26 | 83.36 | 85.74 | 88.69 | 83.22 | 85.87 |

Base and New are averaged over three seeds before computing HM. The Average
row first averages Base and New over eleven datasets, then computes their HM.
[Per-seed CSV](results/base2new_per_seed.csv) ·
[Summary CSV](results/base2new_summary.csv)

<details>
<summary><strong>Run with custom loss coefficients</strong></summary>

Paper settings are listed in `scripts/reproduce/`. General training scripts
accept PV (`lambda_margin`) and FT (`lambda_text`) as hyperparameters:

```bash
DATA=/path/to/datasets
# DATA_ROOT DATASET SEED PV FT [OUTPUT_DIR]
CUDA_VISIBLE_DEVICES=0 bash scripts/caf/base2new_train.sh "$DATA" caltech101 1 4 2.5
```

Both coefficients can also be passed directly to `train.py` using
`--lambda-margin` and `--lambda-text`.
[General script interfaces](docs/EXPERIMENTS.md#general-script-interfaces)

</details>

## Training logs

Detailed original logs are included in the repository for direct browsing.

| Experiment | Logs |
|:--|:--|
| ImageNet source transfer | [Three source seeds](logs/transfer/imagenet/README.md) |
| Base-to-New CAF | [Eleven datasets × three seeds](logs/base2new/caf/README.md) |
| Base-to-New CAF* | [Eleven datasets × three seeds](logs/base2new/caf_star/README.md) |

The [complete log index](logs/README.md) links to every run's per-update
`metrics_epoch*.jsonl`, per-epoch history, model provenance and exported
evaluation counts. `CAF_logs.zip` provides the same log tree for bulk download
through either mirror above.

## Reproducibility

The release preserves the actual training sample lists, class order, frozen text
banks and untrained prompt initializations. Released checkpoints contain the
original learned tensors and fusion scalars. Flowers102 shares three models
between CAF and CAF*, giving **66 distinct trained models across 69 entries**.

Evaluation estimates context from the **complete unlabeled evaluation split**.
Base and New are processed separately; training EMA and target labels are not
used to estimate test context.

See [training and inference details](docs/EXPERIMENTS.md#training-details),
[configuration selection](docs/EXPERIMENTS.md#configuration-selection) and
[model provenance](docs/MODEL_ZOO.md#model-layout).

## Acknowledgements

This implementation builds on [PromptSRC](https://github.com/muzairkhattak/PromptSRC),
[Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) and
[CLIP](https://github.com/openai/CLIP). Frozen text descriptions follow
[ManiPT](https://github.com/xi203040/ManiPT).
[UPSTREAM.md](docs/UPSTREAM.md) records revisions, modifications and notices.

## License

The code is distributed under the [MIT license](LICENSE).
Third-party notices are retained in [licenses/](licenses/).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{chen2026trustadaptationcontextualfusion,
  title={When to Trust Adaptation: Contextual Fusion for Transferable Vision-Language Prompting},
  author={Zhenyuan Chen and Wenshuo Tao and Zhuo Huang and Shuo Chen and Ming-Ming Cheng and Xiang Li},
  year={2026},
  eprint={2611.11824},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.11824}
}
```
