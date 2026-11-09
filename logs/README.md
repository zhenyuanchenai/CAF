# Original training logs

The detailed logs of the released experiments are included here for direct
browsing on GitHub. Each run has a README linking every epoch's update log,
the original epoch history, evaluation counts and model provenance.

| Experiment | Runs |
|:--|:--|
| ImageNet source transfer, PV=1 / FT=40 | [Three source seeds](transfer/imagenet/README.md) |
| Base-to-New CAF, shared PV=4 / FT=2.5 | [Eleven datasets × three seeds](base2new/caf/README.md) |
| Base-to-New CAF*, dataset-specific PV/FT | [Eleven datasets × three seeds](base2new/caf_star/README.md) |

The per-update JSONL and per-epoch JSON files are original training records,
copied without modification. `evaluation_results.json` is explicitly labeled
as a portable export of the completed experiment's results.
`provenance.json` lists model and native-log hashes.

There are 66 distinct trained models across the three experiment groups.
The three Flowers102 runs are shared by CAF and CAF*, and their logs appear
under both scheme folders to make the result tables easy to navigate.

The archive download contains this same log tree for bulk access. No model
checkpoint or dataset image is stored in this directory.
