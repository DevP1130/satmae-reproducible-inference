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

### Smoke test

- **Date:** TBD
- **Cluster:** TBD
- **GPU:** TBD
- **Job ID:** TBD
- **Command:** `sbatch slurm/smoke_test.sbatch`
- **Result:** TBD
- **Issues:** TBD

### Random-init baseline

- **Date:** TBD
- **Cluster:** TBD
- **GPU:** TBD
- **Job ID:** TBD
- **Command:** `sbatch slurm/eval_random.sbatch`
- **Result:** TBD
- **Issues:** TBD

### Full validation eval

- **Date:** TBD
- **Cluster:** TBD
- **GPU:** TBD
- **Job ID:** TBD
- **Command:** `sbatch slurm/eval_full.sbatch`
- **Result:** TBD
- **Issues:** TBD
