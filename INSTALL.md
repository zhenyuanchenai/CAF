# Installation

The released experiments used Linux, Python 3.8.20, PyTorch 1.12.1+cu116,
torchvision 0.13.1+cu116, NumPy 1.24.4, SciPy 1.10.1 and Pillow 10.4.0.
Use this environment when comparing training runs with the archived results.

## Environment

```bash
conda create -n caf python=3.8.20 -y
conda activate caf

# Run these commands from the CAF repository.
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

# Install the same Dassl revision used to check the data and optimizer behavior.
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ../Dassl.pytorch
git -C ../Dassl.pytorch checkout c61a1b570ac6333bd50fb5ae06aea59002fb20bb
pip install --no-deps --no-build-isolation -e ../Dassl.pytorch

python -B train.py --help
```

The PyTorch installation command follows the
[official previous-version instructions](https://pytorch.org/get-started/previous-versions/#v1121).
The pinned Dassl revision predates the PyTorch 2.x scheduler API; use the versions
above rather than upgrading one dependency in isolation.

## Backbone, assets and models

The code downloads the official OpenAI CLIP ViT-B/16 backbone on first use.
Its SHA-256 is
`5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.
The trainer checks this hash before loading the backbone.

Download the packages listed in [MODEL_ZOO.md](docs/MODEL_ZOO.md). Extract them
inside the repository so that `assets/` and `pretrained/` are siblings
of `train.py`. The weights package is needed for evaluation; the assets package
is needed for both training and evaluation. The original dataset images must be
obtained separately. Original training logs are already included in `logs/`;
the log archive is an optional bulk download.

The shell scripts set the CPU library thread counts to one and run on the GPU
selected by `CUDA_VISIBLE_DEVICES`. Set `ASSETS`, `OUTPUT_ROOT` or `PYTHON`
to override their default values (`assets`, `output`, `python`).
Commands should be launched from the repository root.

## Outputs

Training writes the standard console log, per-update `metrics_epoch*.jsonl`,
and `CAF/model.pth.tar-<epoch>` checkpoints under the selected output directory.
Re-running a training command resumes the latest complete epoch in that directory.
Use a new `OUTPUT_ROOT` to start a separate run.

Evaluation writes four-route `logits.npy`, ordered `classes.json` and
`paths.json`, `predictions.npy`, `shared_mean_probability.npy` and
`results.json`. Logits are stored as a disk-backed array to avoid retaining the
entire collection on the GPU.
