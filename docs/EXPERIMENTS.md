# Experiments

## Transfer

The transfer setting trains one model per seed on ImageNet and applies that same
model to ImageNet validation, ten cross-dataset targets and four domain variants.
It uses learned fusion scalars, not a post-hoc alpha/beta grid.

```bash
DATA=/path/to/datasets

# Train three source models with the paper's PV=1 and FT=40.
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/transfer.sh "$DATA" train

# Test all 15 sets using released models.
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/transfer.sh "$DATA" test pretrained

# Test newly trained models.
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/transfer.sh "$DATA" test output
```

To evaluate one target:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/caf/transfer_test.sh \
  "$DATA" 1 pretrained/transfer/imagenet/seed1 imagenet_a
```

The targets are `caltech101 oxford_pets stanford_cars oxford_flowers food101
fgvc_aircraft sun397 dtd eurosat ucf101` and
`imagenetv2 imagenet_sketch imagenet_a imagenet_r`.
Source ImageNet is reported separately from Avg-10 and Avg-4.

The reproduction script lists the source-training loss coefficients and passes
them into the general training script. Testing all seeds also runs the public
aggregator and writes `output/tables/transfer/{per_seed,summary}.csv`.

The three transfer runs use a common 16-shot-per-class subset sampled with seed
42 and a common untrained prompt initialization. Seeds 1/2/3 vary the training
randomness, including augmentation and ordering. The included manifests and
initialization file reproduce this distinction.

## Base-to-new

CAF uses `(lambda_margin, lambda_text) = (4, 2.5)` throughout. CAF* uses:

| Dataset | Script identifier | CAF* lambda_margin | CAF* lambda_text |
|:--|:--|--:|--:|
| ImageNet | imagenet | 1 | 2.5 |
| Caltech101 | caltech101 | 0.5 | 0 |
| OxfordPets | oxford_pets | 2 | 20 |
| StanfordCars | stanford_cars | 0 | 2.5 |
| Flowers102 | oxford_flowers | 4 | 2.5 |
| Food101 | food101 | 4 | 40 |
| FGVCAircraft | fgvc_aircraft | 2 | 0 |
| SUN397 | sun397 | 2 | 5 |
| DTD | dtd | 4 | 5 |
| EuroSAT | eurosat | 1 | 2.5 |
| UCF101 | ucf101 | 0 | 2.5 |

### Train

```bash
DATA=/path/to/datasets
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/base2new.sh "$DATA" caf train
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/base2new.sh "$DATA" caf_star train
```

There are 63 distinct Base-to-New models: 33 CAF plus 33 CAF* minus three shared
Flowers102 models. When CAF has already been trained, the CAF* reproduction
script reuses its completed Flowers102 run directories if the corresponding
CAF* output directories do not exist. Otherwise, CAF* trains or resumes its own
run. The configurations are written explicitly in
`scripts/reproduce/base2new.sh`; the general training script does not choose
loss coefficients based on a scheme name.

### Evaluate

```bash
DATA=/path/to/datasets
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/base2new.sh "$DATA" caf test pretrained
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce/base2new.sh "$DATA" caf_star test pretrained
```

Replace `pretrained` with `output` to evaluate newly trained models. Each invocation
evaluates Base and New with their own class vocabulary and complete collection
statistics, then writes `output/tables/<scheme>/{per_seed,summary}.csv`.

| Evaluation split | Readout |
|:--|:--|
| ImageNet Base | Four-route CAF fusion |
| Base on the other ten datasets | VT |
| New on all eleven datasets | Four-route CAF fusion |

This readout is part of the Table 2 protocol. Do not substitute VT for ImageNet
Base or silently report fused Base on the other ten datasets.

## General script interfaces

The scripts in `scripts/caf/` run a specified experiment. The paper settings
and seed loops live in `scripts/reproduce/`.

```bash
bash scripts/caf/base2new_train.sh DATA_ROOT DATASET SEED PV FT [OUTPUT_DIR]
bash scripts/caf/transfer_train.sh DATA_ROOT SEED PV FT [OUTPUT_DIR]

bash scripts/caf/base2new_test.sh DATA_ROOT DATASET SEED MODEL_DIR [OUTPUT_DIR] [BASE_READOUT]
bash scripts/caf/transfer_test.sh DATA_ROOT SEED MODEL_DIR [DATASET|all] [OUTPUT_DIR]
```

PV is `lambda_margin` and FT is `lambda_text`. They are passed to `train.py`
using `--lambda-margin` and `--lambda-text`, which accept integer, decimal and
scientific notation. Generic training output defaults to a directory containing
the supplied PV/FT values, so different coefficient choices have separate outputs.
The reproduction scripts explicitly use the `caf` and `caf_star` output folders
corresponding to the model packages.

Evaluation loads a specific model directory; the training loss coefficients are
not evaluation controls. `BASE_READOUT` is `VT` by default and may be `fusion`.
The reproduction script explicitly uses `fusion` for ImageNet Base. New always
uses fusion.

For a single run with explicit coefficients and paths:

```bash
DATA=/path/to/datasets
CUDA_VISIBLE_DEVICES=0 bash scripts/caf/base2new_train.sh \
  "$DATA" caltech101 1 0.5 0 output/custom/caltech101/seed1
CUDA_VISIBLE_DEVICES=0 bash scripts/caf/base2new_test.sh \
  "$DATA" caltech101 1 output/custom/caltech101/seed1 \
  output/evaluation/custom/caltech101/seed1 VT
```

Training and testing use the dataset's YAML preset for batch size, optimizer and
epoch count. All original detailed logs are available in the [log index](../logs/README.md).

## Training details

| Setting | Source transfer | Base-to-New, ten datasets | Base-to-New, ImageNet |
|:--|:--|:--|:--|
| Backbone | CLIP ViT-B/16 | CLIP ViT-B/16 | CLIP ViT-B/16 |
| Trainable prompts | 8 tokens, depth 12, both modalities | Same | Same |
| Labeled examples | 16 per ImageNet class | 16 per Base class | 16 per Base class |
| Epochs | 5 | 20 | 20 |
| Original images per update | 32 | 4 | 32 |
| Independent views per image | 4 | 4 | 4 |
| Prompt SGD peak LR | 0.01 | 0.0025 | 0.01 |
| Prompt momentum / decay | 0.9 / 0.0005 | Same | Same |
| Prompt warmup | 1 epoch, absolute LR 1e-5 | Same | Same |
| Following prompt schedule | Dassl cosine | Dassl cosine | Dassl cosine |
| Fusion optimizer | Adam, LR 0.001 | Same | Same |
| Fusion Adam betas / epsilon / decay | (0.9, 0.999) / 1e-8 / 0 | Same | Same |
| Fusion warmup | 1 epoch, LR 2.5e-6 | Same | Same |
| Initial alpha / beta | 16 / 10.849894523620605 | Same | Same |
| Shared training EMA rate | 0.01 | Same | Same |
| Test image batch size | 128 | 64 | 64 |

The prompt objective is VT cross-entropy plus `lambda_margin` times the positive
frozen-margin preservation loss on V0, plus `lambda_text` times the text-anchor
loss. Fusion cross-entropy has coefficient one and receives detached four-route
logits. It updates only alpha and beta, followed by nonnegative projection.
The shared soft-probability EMA starts uniform, persists across epochs and is
updated after prediction and parameter updates.

Images use 224-pixel bicubic crops, random horizontal flipping, RandAugment
(N=2, M=10), and CLIP normalization. Testing uses resize and center crop.
The four views are concatenated in view-major order.
Encoder autocast is enabled; logits/loss/fusion are FP32 and probability,
entropy and frequency accumulation are FP64. Deep prompts retain the residual
stream dtype. The released initialization tensors preserve the exact starting
point, including the text prefix initialized from "a photo of a".

Training resets its random seed at each epoch. For zero-based epoch `e`,
Base-to-New uses `seed + 100003 * e`; transfer uses
`seed + 100003 * (e + 1)`. Data-loader workers are recreated each epoch.
Base-to-New data selection and initialization are paired by seed across loss
configurations. No validation-selected epoch or checkpoint averaging is used.

## Inference and aggregation

Routes are always ordered `00, V0, 0T, VT`. Let `h` be normalized route entropy
and `q` be the negative log frequency of the route's top-1 class, estimated from
the shared average of all four route soft probabilities over the complete
evaluation set. The class frequency is multiplied by the vocabulary size.
Context is zeroed when all four top-1 predictions agree.
Weights are `softmax(-alpha * h + beta * q)`; the final prediction is the
weighted sum of the **original logits**.

No test labels enter frequency estimation, scalar fitting or epoch selection.
The learned source scalars are retained for all transfer targets. Base-to-New
uses the scalars learned on that run's Base data.

For Table 1, average each target's accuracy over three seeds. For each seed,
Avg-10/Avg-4 average the corresponding target accuracies equally; sample SD is
computed over those three seed averages.

For Table 2, average Base and New over seeds, then compute
`HM = 2 * Base * New / (Base + New)`. The Average row first averages Base and
New equally across eleven datasets, then applies the same formula. SD uses
seed-level quantities, with `ddof=1`.

## Configuration selection

The release reproduces the supplied manuscript's final settings, including its
reported loss-weight selection. CAF is the shared-coefficient choice (4, 2.5);
CAF* is the reported dataset-specific selection over the explored loss grid.
ImageNet CAF* uses (1, 2.5), final epoch 20. Transfer uses (1, 40), final epoch 5.

The loss-weight study is exploratory and uses the reported benchmark results
for configuration selection. Its selected improvements should not be interpreted
as an independently held-out estimate or a claim of statistical significance.
The release does not rerun selection or tune coefficients on a user's test set.
