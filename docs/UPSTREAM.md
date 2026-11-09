# Upstream code and attribution

## PromptSRC

The public repository follows
[PromptSRC](https://github.com/muzairkhattak/PromptSRC) at commit
`bb95c77b634d63488f2cad81ff4a72d53bdd06d5`.

The dataset definitions and their YAML files, CLIP tokenizer/BPE vocabulary and
CLIP utilities are retained from this revision, with the version warning
compared numerically. `train.py` preserves the
argument/configuration/registry structure and registers CAF only.

`clip/model.py` retains the upstream CLIP and independent visual/language
prompting structure. CAF-specific changes are limited to the prompt-enabled
encoder construction, deep-prompt dtype, contiguous visual token layout and
activation checkpointing. Unused prompt-method branches are removed.
Deep prompts use the current residual-stream dtype, preserving the numerical
behavior of the trained CAF models; changing them to an unconditional FP16 cast
would alter those models.

CAF's trainer and detached scalar fusion are in `trainers/caf.py` and
`trainers/fusion.py`. The manifest loader in `datasets/caf_splits.py` preserves
the actual experiment class order and sample selection without machine-specific
paths. The repository contains no cluster scheduler, job monitor, recovery
controller or experiment-search launcher.

The PromptSRC MIT notice is retained in [LICENSE](../LICENSE) and
[licenses/PromptSRC.txt](../licenses/PromptSRC.txt).

## Dassl and CLIP

The external [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) dependency is
pinned to `c61a1b570ac6333bd50fb5ae06aea59002fb20bb`. Its data wrapper, transforms,
RandAugment and learning-rate scheduler match the corresponding components
used by the original experiments. No local modification to Dassl is required.

The backbone and tokenizer originate from [OpenAI CLIP](https://github.com/openai/CLIP).
The backbone checkpoint is downloaded from its original location and verified
by SHA-256. Its MIT notice is in [licenses/CLIP.txt](../licenses/CLIP.txt).

## Text descriptions

The frozen GPT-description JSON files follow the description bank distributed
with [ManiPT](https://github.com/xi203040/ManiPT), revision
`26a447dcbad3324b568ebafa8adeec36f44d6e97`, which credits CoPrompt's description
format. The associated MIT notices are retained in
[licenses/ManiPT.txt](../licenses/ManiPT.txt).
The release uses the exact description content and cached feature banks from
the completed CAF experiments.

## Method figure

`docs/method.svg` is the method overview supplied with the CAF manuscript.
Its photographic examples carry the credits recorded with that figure:

- Eastern Bluebird (male): Neal Lewis / NPS, Shenandoah National Park.
  [Source](https://commons.wikimedia.org/wiki/File:Eastern_Bluebird_(male)_(45635802031).jpg).
  U.S. public domain, official National Park Service work.
- Portrait of a cat: Leon Brooks.
  [Source](https://commons.wikimedia.org/wiki/File:Portrait_of_a_cat.jpg).
  Public-domain dedication by the photographer.
- Jackson Lake, Grand Teton National Park: Kimberly Finch / NPS.
  [Source](https://commons.wikimedia.org/wiki/File:Jackson_Lake.jpg).
  U.S. public domain, official National Park Service work.

Photographs were cropped and resampled for framing. The diagram is distributed
as the supplied manuscript artwork.
