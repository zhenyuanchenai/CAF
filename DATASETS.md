# Datasets

Obtain the datasets using the
[PromptSRC dataset instructions](https://github.com/muzairkhattak/PromptSRC/blob/bb95c77b634d63488f2cad81ff4a72d53bdd06d5/docs/DATASETS.md).
Dataset images are not redistributed in this release.

Use one dataset root with the layout below. In particular, ImageNet validation
images must be organized into their class subdirectories.

```text
datasets/
├── imagenet/images/{train,val}/
├── caltech-101/101_ObjectCategories/
├── oxford_pets/images/
├── stanford_cars/{cars_train,cars_test}/
├── oxford_flowers/jpg/
├── food-101/images/
├── fgvc_aircraft/images/
├── sun397/SUN397/
├── dtd/images/
├── eurosat/2750/
├── ucf101/UCF-101-midframes/
├── imagenetv2/imagenetv2-matched-frequency-format-val/
├── imagenet-sketch/images/
├── imagenet-adversarial/imagenet-a/
└── imagenet-rendition/imagenet-r/
```

## Exact experiment splits

The supplied scripts use the relative paths and class vocabulary in
`assets/splits/`. These manifests preserve the image order and training
examples of the released runs; they do not contain image pixels or machine-local
absolute paths.

- `test.json`: the ordered complete test collection, class vocabulary and
  `base_count`. Base is the first `base_count` classes and New is the remaining
  classes. New labels are remapped to zero-based indices in that vocabulary.
- `train_seed1.json`, `train_seed2.json`, `train_seed3.json`: the selected
  16-shot Base samples for Base-to-New.
- `test_transfer.json`: the complete evaluation set for source transfer.
- `imagenet/train_transfer.json`: the 16,000 source training images used by all
  three transfer seeds.

Class names in a frozen text bank must match the manifest in both content and
order. The loader validates this and rejects a missing image or an incorrect
shot count. For ImageNet-A/R, use the supplied subset vocabulary rather than
assuming all 1,000 columns are present.

## Frozen text banks

`assets/references/<dataset>/all.pt` contains the ordered GPT-description
prototypes for the full vocabulary. `base.pt` contains the Base vocabulary used
for Base training and testing. New evaluation selects the New suffix from
`all.pt`. Each prototype is the normalized mean of individually normalized
frozen CLIP description embeddings. The cached vectors preserve the original
experiment values.

The original descriptions are included in `gpt_file/`. For an optional rebuild:

```bash
python -B tools/build_references.py \
  --manifest assets/splits/caltech101/test.json \
  --descriptions gpt_file/caltech_prompt.json \
  --split all \
  --output rebuilt/caltech101/all.pt
```

Use `--split base` for the Base bank. ImageNet domain variants use
`gpt_file/imagenet_prompt.json` with their own target manifests.
Rebuilding on a different software/device stack may introduce numerical
differences; the released banks are the reference for evaluating released models.
