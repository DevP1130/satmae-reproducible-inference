# satmae-reproducible-inference

Reproducible inference workflow for SatMAE (NeurIPS 2022) on fMoW-Sentinel, run on Georgia Tech AI Makerspace (SLURM cluster).

This repo evaluates the officially released, already fine-tuned SatMAE ViT-Base
(group-channels) checkpoint on the fMoW-Sentinel **validation** split only, and
checks the result against the 62.65% top-1 accuracy reported in the upstream
README. No training, no ONNX export, no benchmarking -- inference reproduction
only.

## Overview

- **Model:** SatMAE multi-spectral ViT-Base, `group_c` variant, released
  fine-tuned checkpoint `finetune-vit-base-e7.pth`.
- **Upstream code:** [sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE),
  vendored as a git submodule at `third_party/SatMAE`.
- **Data:** fMoW-Sentinel, validation split only.
- **Target:** reproduce the upstream README's reported 62.65% top-1 val
  accuracy for ViT-Base (200-epoch pretrain + finetune). Pass = within 1
  percentage point (61.65%-63.65%).

## Provenance

| Item | Value |
| --- | --- |
| Upstream code repo | https://github.com/sustainlab-group/SatMAE |
| Upstream commit (submodule pin) | `0b210aceb37a14bbbd897110db5b104b3271d818` (`main`, 2025-08-10) |
| Checkpoint | `finetune-vit-base-e7.pth` |
| Checkpoint source | https://zenodo.org/record/7338613 (DOI [10.5281/zenodo.7338613](https://doi.org/10.5281/zenodo.7338613)) |
| Checkpoint SHA-256 | TBD -- written to `weights/finetune-vit-base-e7.pth.sha256` by `scripts/download_weights.sh` |
| Dataset | fMoW-Sentinel, validation split only |
| Dataset source | https://purl.stanford.edu/vg497cb6002 (DOI [10.25740/vg497cb6002](https://doi.org/10.25740/vg497cb6002)) |
| Reported upstream top-1 (val) | 62.65% |

Model args (from the upstream README's finetune command and
`models_vit_group_channels.py` / `main_finetune.py` -- see
`scripts/evaluate.py` for where each is hard-coded):

- `--model vit_base_patch16 --model_type group_c`
- `--input_size 96 --patch_size 8`
- `--dataset_type sentinel --dropped_bands 0 9 10`
- `--grouped_bands 0 1 2 6 --grouped_bands 3 4 5 7 --grouped_bands 8 9`
- `--nb_classes 62`
- `global_pool=True` (upstream default)

## Environment setup

Home directory on ICE is quota-limited to 30 GB, so the conda env is built
under `~/scratch` (300 GB quota) instead of the default `~/.conda/envs`:

```bash
mkdir -p ~/scratch/envs
conda env create -f env/environment.yml -p ~/scratch/envs/satmae
conda activate ~/scratch/envs/satmae
```

`env/environment.yml` pins `torch==1.12.1` / `timm==0.3.2` to match the
checkpoint's original training environment. See Troubleshooting below for
why this matters.

## Data + weights download

```bash
# Checkpoint (wget + sha256sum, writes checksum file)
bash scripts/download_weights.sh

# Validation split images + val.csv (edit the TODO URLs first -- see script)
bash scripts/download_data.sh

# Rebuild the image_path column to match the on-disk layout, write the
# prepared CSV the evaluator reads
python scripts/prepare_csv.py --csv data/val.csv --split val \
  --data-root data --out data/val_prepared.csv
```

`scripts/download_data.sh` has placeholder `TODO_*` URLs: the Stanford
Digital Repository item page
(https://purl.stanford.edu/vg497cb6002) requires accepting the fMoW /
Sentinel-2 usage terms in a browser before a direct file link is exposed, so
there's no single stable URL to hardcode. Open the page, accept the terms,
copy the validation-split archive and `val.csv` links into the script (or
download manually into `data/`), then re-run.

`scripts/prepare_csv.py` rewrites `image_path` to:

```
fmow-sentinel/<split>/<category>/<category>_<location_id>/<category>_<location_id>_<image_id>.tif
```

as documented in the upstream README, and supports `--subset N` for a
class-stratified fallback sample if the full validation set can't be
downloaded or run in time.

## Run smoke test

Evaluates the first 100 images only, to confirm the environment, checkpoint,
and data paths all work before committing to a full run (15 min time limit).

```bash
sbatch slurm/smoke_test.sbatch
```

## Run full eval

```bash
# Fine-tuned checkpoint on the full validation split
sbatch slurm/eval_full.sbatch

# Randomly initialized weights, sanity baseline (expected ~1/62 = 1.6% top-1)
sbatch slurm/eval_random.sbatch
```

Each job writes `results/metrics.json` (top-1, top-5, n_images, per-class
accuracy, runtime, GPU name, peak GPU memory) and
`results/confusion_matrix.csv`, plus `nvidia-smi` output, torch/CUDA
versions, wall time, and `sacct` output into `logs/`.

You can also run `scripts/evaluate.py` directly (e.g. interactively on a
compute node) -- see the sbatch files for the exact invocation and flags
(`--limit N` for a smoke-size run, `--random-init` for the baseline).

## Validation

| Check | Expected | Observed |
| --- | --- | --- |
| Smoke test (100 images) runs without error | completes, prints top-1/top-5 | TBD |
| Full val top-1 accuracy | 62.65% +/- 1 pt (61.65%-63.65%) | TBD |
| Full val top-5 accuracy | upstream not reported; sanity only | TBD |
| Random-init baseline top-1 | ~1/62 = 1.6% | TBD |
| n_images evaluated | full val split count | TBD |

Fill in "Observed" from `results/metrics.json` after each run, and log every
run in `RUN_LOG.md`.

## Compute resources used

Runs use the `coc-gpu` partition on Georgia Tech ICE (account `coc`, QOS
`coc-ice`) with 1x NVIDIA L40S (`--gres=gpu:l40s:1`, as set in `slurm/*.sbatch`).
1x NVIDIA A100 (`--gres=gpu:a100:1`) is available as an alternative on the
same partition if L40S nodes are busy.

**Avoid the AMD MI210 GPUs also present on `coc-gpu`.** This codebase
(PyTorch + CUDA, `rasterio`, etc.) requires CUDA; MI210 nodes are ROCm-only
and will not run this code. Do not set `--gres=gpu:mi210:1`.

- Cluster: TBD
- GPU: TBD (see `logs/*_nvidia-smi.txt`)
- Wall time (full eval): TBD (see `logs/*_walltime.txt`)
- Peak GPU memory: TBD (see `results/metrics.json`)

## Troubleshooting

- **`torch._six` ImportError:** `util/misc.py` in the upstream repo does
  `from torch._six import inf`, which was removed in torch>=1.13. This repo
  pins `torch==1.12.1` in `env/environment.yml` so that module still works,
  *and* `scripts/evaluate.py` never imports `util/misc.py` or
  `engine_finetune.py` in the first place -- it only imports
  `models_vit_group_channels.py` and `util/datasets.py`, reimplementing the
  small top-1/top-5 accuracy helper itself. If you extend this repo to reuse
  more of the upstream training code, you'll hit this again.
- **`timm` version mismatch:** the checkpoint and `models_vit_group_channels.py`
  assume the old `timm==0.3.2` API (e.g. `VisionTransformer` internals). Newer
  timm renamed/removed attributes the upstream code depends on. Keep the pin.
- **`torch.distributed.launch` / DDP:** the upstream commands all launch via
  `torch.distributed.launch --nproc_per_node=8`. This repo is single-GPU only;
  `scripts/evaluate.py` does not use distributed training/eval at all, so
  there's nothing to launch -- just `python scripts/evaluate.py ...`.
- **Hard-coded wandb entity:** `main_finetune.py` calls
  `wandb.init(project=args.wandb, entity="mae-sentinel")` unconditionally if
  `--wandb` is set. `scripts/evaluate.py` doesn't use `main_finetune.py` at
  all, so wandb never gets initialized -- no API key or entity needed.
- **`image_path` column:** the raw `val.csv` from the Stanford repository
  does not contain a usable `image_path`; always run
  `scripts/prepare_csv.py` first (see Data + weights download above).
- **Missing/unexpected keys on checkpoint load:** `scripts/evaluate.py`
  prints `model.load_state_dict(..., strict=False)`'s missing/unexpected key
  lists on every run. For this checkpoint against this exact model config
  (`input_size=96`, `patch_size=8`, `group_c`, `nb_classes=62`) both lists
  should be empty; non-empty lists mean an arg mismatch somewhere above.

## Cleanup

```bash
rm -rf data/ weights/ results/smoke results/random_baseline
# results/metrics.json, results/confusion_matrix.csv, and logs/ stay tracked
```

`data/`, `weights/`, `*.pth`, and other large artifacts are gitignored and
safe to delete locally at any time; re-run the download scripts to restore
them.

## License notes

- **Code:** upstream SatMAE (`third_party/SatMAE`) is licensed CC BY-NC 4.0
  -- non-commercial use only. This repo's own scripts inherit that
  restriction where they import upstream code.
- **Data:** fMoW-Sentinel is derived from the Functional Map of the World
  (fMoW) dataset and Sentinel-2 imagery; both carry their own usage terms
  (fMoW license + Copernicus/Sentinel-2 terms) that you accept when
  downloading from the Stanford Digital Repository. Do not redistribute the
  imagery; this repo only stores download scripts, not the data itself.
