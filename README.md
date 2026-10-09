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
| Checkpoint SHA-256 | see `weights/finetune-vit-base-e7.pth.sha256` (written by `scripts/download_weights.sh`); not reproduced inline here since `weights/` is gitignored and outside this submission's tracked directories |
| Dataset | fMoW-Sentinel, validation split only |
| Dataset source | https://purl.stanford.edu/vg497cb6002 (DOI [10.25740/vg497cb6002](https://doi.org/10.25740/vg497cb6002)), Version 1 |
| Dataset archive | `fmow-sentinel.tar.gz`, 77,497,587,986 bytes (checksum in `data/checksums_data.sha256`, not tracked here) |
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
under `~/scratch` (300 GB quota) instead of the default `~/.conda/envs`.
Building it directly on the login node was killed (login nodes enforce
resource limits); instead it's built as a batch job on the `coc-cpu`
partition:

```bash
# From the repo root:
sbatch slurm/build_env.sbatch
```

This ran as job `6106735` (`coc-cpu`, 7m44s) and resolved to `torch 1.12.1`,
`CUDA 11.3`, `timm 0.3.2`, matching the checkpoint's original training
environment. The exact resolved versions are pinned in
`env/environment.lock.yml` and `env/pip-freeze.txt`.

`timm==0.3.2` itself needs one source patch: its
`timm/models/layers/helpers.py` does `from torch._six import container_abcs`,
which doesn't exist under that name even in the pinned torch 1.12.1 (this
surfaced as an `ImportError` in job 6106735's smoke-check). Fix it once,
after the env is built:

```bash
conda activate ~/scratch/envs/satmae
bash scripts/patch_timm.sh
```

See Troubleshooting below for how this differs from the separate
`util/misc.py` `torch._six.inf` issue in the upstream repo.

## Data + weights download

```bash
# Checkpoint (wget + sha256sum, writes checksum file)
bash scripts/download_weights.sh

# Validation split images + val.csv, via the Stanford Digital Repository.
# Run as a batch job, not on the login node -- this moves ~77.5 GB and
# took ~5h13m at ~3.9 MB/s (job 6107078, coc-cpu).
sbatch slurm/download_data.sbatch

# The archive (fmow-sentinel.tar.gz, 77,497,587,986 bytes) is only needed to
# extract val/ from it -- delete it afterward to stay inside the 300 GB
# scratch quota:
rm -f data/fmow-sentinel.tar.gz

# Rebuild the image_path column to match the on-disk layout, write the
# prepared CSV the evaluator reads
python scripts/prepare_csv.py --csv data/val.csv --split val \
  --data-root data --out data/val_prepared.csv

# Optional: class-stratified subset for a faster secondary check
python scripts/prepare_csv.py --csv data/val.csv --split val \
  --data-root data --subset 5000 --out data/val_subset5000.csv
```

`scripts/download_data.sh` extracts only the `val/` split from the archive
(`tar -xzf ... --wildcards '*/val/*'`): 92,263 `.tif` files land under
`data/fmow-sentinel/val/`. `val.csv` itself lists 84,939 images; all 84,939
resolve correctly through `scripts/prepare_csv.py`'s rebuilt `image_path`
(the extra `.tif` files on disk are not referenced by any `val.csv` row and
can be ignored).

`scripts/prepare_csv.py` rewrites `image_path` to:

```
fmow-sentinel/<split>/<category>/<category>_<location_id>/<category>_<location_id>_<image_id>.tif
```

as documented in the upstream README, and supports `--subset N` for a
class-stratified fallback sample if the full validation set can't be
downloaded or run in time.

**Always submit from the repo root.** Every script under `slurm/` resolves
the repo path from `$SLURM_SUBMIT_DIR` (the directory `sbatch` was invoked
from), not from the script's own location -- see Troubleshooting for why.

## Run smoke test

Evaluates the first 100 images only, to confirm the environment, checkpoint,
and data paths all work before committing to a full run (15 min time limit).

```bash
sbatch slurm/smoke_test.sbatch
```

## Run full eval

```bash
# Fine-tuned checkpoint on the full validation split (84,939 images)
sbatch slurm/eval_full.sbatch

# Secondary: class-stratified 5,000-image subset, useful as a faster
# cross-check (see Validation below for why it reads lower than the full set)
sbatch slurm/eval_subset.sbatch
```

`eval_random.sbatch` (random-init sanity baseline) exists in `slurm/` but was
not run for this submission.

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
| Smoke test (100 images) runs without error | completes, prints top-1/top-5 | Pass -- job 6114790: 0 missing / 0 unexpected checkpoint keys, top1 97.00%, top5 100.00%, 72s wall. Not an accuracy signal: the first 100 CSV rows aren't class-balanced, so this checks the pipeline only. |
| **Full val top-1 accuracy (84,939 images)** | 62.65% +/- 1 pt (61.65%-63.65%) | **PASS -- 62.65%** (job 6115115, `results/metrics.json`). top5 85.79%, 271.7s eval / 278s wall, ~1.84 GB peak GPU mem, 0 missing / 0 unexpected checkpoint keys. Matches the upstream-reported 62.65% exactly. |
| Stratified subset top-1 accuracy (4,950 images, secondary) | n/a -- informal cross-check only | 59.45% (job 6114806, `results/subset5000/metrics.json`), top5 83.03%, 46s wall. Lower than the primary full-val result because `prepare_csv.py --subset` draws a class-stratified sample (~80/class) rather than val's natural class distribution. |
| Random-init baseline top-1 | ~1/62 = 1.6% | Not run (`eval_random.sbatch` was not submitted for this submission). |
| n_images evaluated | full val split count (84,939) | full: 84,939 (job 6115115, primary); subset: 4,950 (job 6114806, secondary); smoke: 100 (job 6114790). |

See `RUN_LOG.md` for the full job-by-job history, including the failures
that preceded the passing full-val run.

## Compute resources used

Runs use the `coc-gpu` partition on Georgia Tech ICE (account `coc`, QOS
`coc-ice`) with 1x NVIDIA L40S (`--gres=gpu:l40s:1`, as set in `slurm/*.sbatch`).
1x NVIDIA A100 (`--gres=gpu:a100:1`) is available as an alternative on the
same partition if L40S nodes are busy.

**Avoid the AMD MI210 GPUs also present on `coc-gpu`.** This codebase
(PyTorch + CUDA, `rasterio`, etc.) requires CUDA; MI210 nodes are ROCm-only
and will not run this code. Do not set `--gres=gpu:mi210:1`.

- Cluster: Georgia Tech ICE (partition `coc-gpu`, account `coc`, QOS `coc-ice`)
- GPU: 1x NVIDIA L40S-46GB (confirmed via `nvidia-smi` in `logs/*_nvidia-smi.txt`, e.g. `logs/smoke_6114790_nvidia-smi.txt`)
- Wall time: full (84,939 images) 278s (job 6115115); subset (4,950 images) 46s (job 6114806); smoke (100 images) 72s (job 6114790)
- Peak GPU memory: ~1.84 GB (full, `results/metrics.json`); ~1.84 GB (subset, `results/subset5000/metrics.json`); ~1.17 GB (smoke, `results/smoke/metrics.json`)
- Env build: CPU-only job on `coc-cpu`, 7m44s (job 6106735)
- Data download: CPU-only job on `coc-cpu`, ~5h13m at ~3.9 MB/s for the 77.5 GB archive (job 6107078)

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
- **`timm` `container_abcs` ImportError (distinct from the `util/misc.py`
  issue above):** `timm==0.3.2`'s `timm/models/layers/helpers.py` does
  `from torch._six import container_abcs`, which fails even under the
  pinned torch 1.12.1 (`ImportError: cannot import name 'container_abcs'
  from 'torch._six'`, hit during env build job 6106735). Fixed with
  `scripts/patch_timm.sh`, which rewrites that line to
  `import collections.abc as container_abcs`. Run it once after creating
  the env (see Environment setup).
- **`mkdir: Permission denied` in a Slurm job (job 6114779):** the sbatch
  scripts originally resolved the repo path from `${BASH_SOURCE[0]}`, which
  inside a Slurm job points at Slurm's own spooled copy of the submission
  script, not your checkout -- so `cd`/`mkdir` landed in a directory you
  don't own. Fixed by using `REPO_ROOT="${SLURM_SUBMIT_DIR}"` instead, which
  is why every job must be submitted with `sbatch` from the repo root.
- **`MKL_INTERFACE_LAYER: unbound variable` (job 6114783):** with
  `set -euo pipefail`, conda's own activation hook
  (`libblas_mkl_activate.sh`) trips `set -u` because it references
  `$MKL_INTERFACE_LAYER` without a default. Fixed by wrapping the `conda
  activate` call in `set +u; conda activate ...; set -u`, as done in
  `slurm/*.sbatch`.
- **`Invalid qos specification`:** happened when submitting with
  `-A coe-gpu`; `coe-gpu` isn't this account's association. The correct
  values, confirmed via `sacctmgr`/`sinfo`, are account `coc`, partition
  `coc-gpu`, QOS `coc-ice` -- see `RUN_LOG.md` (2026-10-08 entry).
- **OOM on the full validation set (job 6114969):** evaluating all 84,939
  images with `--mem=64G` got a DataLoader worker killed by SIGKILL and 1
  `oom_kill` event on the host. Fixed by raising to `--mem=96G` in
  `slurm/eval_full.sbatch` (rerun as job 6115115).

## Cleanup

```bash
rm -rf data/ ~/scratch/envs/satmae ~/scratch/conda_pkgs
```

`data/`, `weights/`, `*.pth`, and other large artifacts are gitignored and
safe to delete locally at any time; re-run the download scripts (and
`slurm/build_env.sbatch` + `scripts/patch_timm.sh`) to restore them. `logs/`
and `results/` stay tracked.

## License notes

- **Code:** upstream SatMAE (`third_party/SatMAE`) is licensed CC BY-NC 4.0
  -- non-commercial use only. This repo's own scripts inherit that
  restriction where they import upstream code.
- **Data:** fMoW-Sentinel is derived from the Functional Map of the World
  (fMoW) dataset and Sentinel-2 imagery; both carry their own usage terms
  (fMoW license + Copernicus/Sentinel-2 terms) that you accept when
  downloading from the Stanford Digital Repository. Do not redistribute the
  imagery; this repo only stores download scripts, not the data itself.
