# Run Log

Record every cluster run here, one entry per job. Copy the template below for each new row.

## Template

- **Date:** YYYY-MM-DD
- **Cluster:** TBD (e.g. AI Makerspace / PACE ICE)
- **GPU:** TBD (e.g. 1x A100-40GB, from `nvidia-smi` in logs/)
- **Job ID:** TBD (SLURM `$SLURM_JOB_ID`)
- **Command:** TBD (exact `sbatch ...` invocation)
- **Result:** TBD (top-1 / top-5 / n_images from results/metrics.json)
- **Issues:** TBD (anything that went wrong, deviated, or needed a workaround)

---

## Entries

### 2026-10-08 -- first ICE login / account setup

- **Date:** 2026-10-08
- **Cluster:** Georgia Tech ICE
- **GPU:** n/a (no job run yet)
- **Job ID:** n/a
- **Command:** n/a -- initial login and account check (`sacctmgr`, `sinfo`)
- **Result:** Home quota 30 GB; scratch quota 300 GB at
  `/storage/ice1/5/8/dpatel878`. Association confirmed: account `coc`, QOS
  `coc-ice`.
- **Issues:** Tried `-A coe-gpu`, rejected by Slurm with
  "Invalid qos specification". Correct values for my association are
  account `coc`, partition `coc-gpu`, QOS `coc-ice` -- now hard-coded in
  `slurm/*.sbatch`. Because home is only 30 GB, the conda env is created
  under `~/scratch/envs/satmae` instead of the default location.

### 2026-10-08 -- environment build

- **Date:** 2026-10-08
- **Cluster:** Georgia Tech ICE, partition `coc-cpu`, account `coc`, QOS `coc-ice`
- **GPU:** n/a (CPU-only build job)
- **Job ID:** 6106735
- **Command:** `sbatch slurm/build_env.sbatch`
- **Result:** Succeeded in 7m44s. Resolved `torch 1.12.1`, `CUDA 11.3`,
  `timm 0.3.2`; lock files written to `env/environment.lock.yml` and
  `env/pip-freeze.txt`.
- **Issues:** Building interactively on the login node was killed by login
  node resource limits, hence the batch job. The job's own import
  smoke-check failed with `ImportError: cannot import name 'container_abcs'
  from 'torch._six'` (`timm==0.3.2`'s `timm/models/layers/helpers.py`).
  Fixed afterward with `scripts/patch_timm.sh` (rewrites that import to
  `collections.abc`).

### 2026-10-08 -- data download

- **Date:** 2026-10-08
- **Cluster:** Georgia Tech ICE, partition `coc-cpu`, account `coc`, QOS `coc-ice`
- **GPU:** n/a (CPU-only download job)
- **Job ID:** 6107078
- **Command:** `sbatch slurm/download_data.sbatch` (runs `scripts/download_data.sh`)
- **Result:** fMoW-Sentinel from Stanford SDR (DOI 10.25740/vg497cb6002,
  Version 1). Downloaded `fmow-sentinel.tar.gz`, 77,497,587,986 bytes, in
  ~5h13m at ~3.9 MB/s. Extracted only `val/` (92,263 `.tif` files); `val.csv`
  lists 84,939 images, all of which resolved correctly via
  `scripts/prepare_csv.py`. Archive deleted after extraction to stay within
  the 300 GB scratch quota.
- **Issues:** None.

### 2026-10-08 -- smoke test (failed attempt 1)

- **Date:** 2026-10-08
- **Cluster:** Georgia Tech ICE, partition `coc-gpu`, account `coc`, QOS `coc-ice`
- **GPU:** 1x NVIDIA L40S
- **Job ID:** 6114779
- **Command:** `sbatch slurm/smoke_test.sbatch`
- **Result:** Failed immediately: `mkdir: cannot create directory 'logs': Permission denied` (and same for `results`).
- **Issues:** The sbatch script resolved its repo path from
  `${BASH_SOURCE[0]}`, which inside a Slurm job points at Slurm's spooled
  copy of the submission script, not the actual repo checkout. Fixed by
  switching to `REPO_ROOT="${SLURM_SUBMIT_DIR}"` in all `slurm/*.sbatch`
  scripts -- meaning jobs must always be submitted with `sbatch` from the
  repo root.

### 2026-10-08 -- smoke test (failed attempt 2)

- **Date:** 2026-10-08
- **Cluster:** Georgia Tech ICE, partition `coc-gpu`, account `coc`, QOS `coc-ice`
- **GPU:** 1x NVIDIA L40S
- **Job ID:** 6114783
- **Command:** `sbatch slurm/smoke_test.sbatch`
- **Result:** Failed during environment activation:
  `/home/hice1/dpatel878/scratch/envs/satmae/etc/conda/activate.d/libblas_mkl_activate.sh: line 1: MKL_INTERFACE_LAYER: unbound variable`.
- **Issues:** `set -euo pipefail` plus conda's own activation hook
  referencing an undefined variable. Fixed by wrapping activation in
  `set +u; conda activate ...; set -u`.

### 2026-10-08 -- smoke test (pass)

- **Date:** 2026-10-08
- **Cluster:** Georgia Tech ICE, partition `coc-gpu`, account `coc`, QOS `coc-ice`
- **GPU:** 1x NVIDIA L40S
- **Job ID:** 6114790
- **Command:** `sbatch slurm/smoke_test.sbatch`
- **Result:** Pass. 100 images, checkpoint loaded with 0 missing / 0
  unexpected keys. top1 97.00%, top5 100.00%, 31.1s eval / 72s wall,
  ~1.17 GB peak GPU memory. See `results/smoke/metrics.json`.
- **Issues:** None (pipeline check only -- the first 100 CSV rows are not
  class-balanced, so this is not an accuracy signal).

### 2026-10-08 -- stratified subset eval

- **Date:** 2026-10-08
- **Cluster:** Georgia Tech ICE, partition `coc-gpu`, account `coc`, QOS `coc-ice`
- **GPU:** 1x NVIDIA L40S
- **Job ID:** 6114806
- **Command:** `sbatch slurm/eval_subset.sbatch` (`data/val_subset5000.csv`, class-stratified)
- **Result:** 4,950 images, top1 59.45%, top5 83.03%, 46s wall, ~1.84 GB
  peak GPU memory. See `results/subset5000/metrics.json`. ~3pt below the
  upstream 62.65%, attributable to the stratified sample overweighting
  rare/harder classes relative to val's natural class distribution.
- **Issues:** None.

### 2026-10-08 -- full validation eval (OOM)

- **Date:** 2026-10-08
- **Cluster:** Georgia Tech ICE, partition `coc-gpu`, account `coc`, QOS `coc-ice`
- **GPU:** 1x NVIDIA L40S
- **Job ID:** 6114969
- **Command:** `sbatch slurm/eval_full.sbatch` (`--mem=64G`, full 84,939-image val set)
- **Result:** Failed: DataLoader worker killed by SIGKILL, 1 `oom_kill`
  event reported by Slurm.
- **Issues:** Host RAM limit too low for the full-size run. Fixed by
  raising `--mem` to `96G` in `slurm/eval_full.sbatch`.

### 2026-10-08 -- full validation eval (rerun, PASS -- primary result)

- **Date:** 2026-10-08
- **Cluster:** Georgia Tech ICE, partition `coc-gpu`, account `coc`, QOS `coc-ice`
- **GPU:** 1x NVIDIA L40S
- **Job ID:** 6115115
- **Command:** `sbatch slurm/eval_full.sbatch` (`--mem=96G`, full 84,939-image val set)
- **Result:** Checkpoint loaded cleanly (0 missing / 0 unexpected keys).
  n_images=84,939, top1=62.65%, top5=85.79%, runtime 271.7s / 278s wall,
  ~1.84 GB peak GPU memory. Exactly matches the 62.65% top-1 reported
  upstream -- **PASS**. See `results/metrics.json`. This is the primary
  validation result for this submission; the stratified-subset result
  above (job 6114806, 59.45%) stands as a secondary cross-check only.
- **Issues:** None -- this rerun (with `--mem=96G`) completed cleanly
  after the OOM in job 6114969.
