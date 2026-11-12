# IAD official-CFLOW matrix on Berzelius

This directory contains a reproducible Slurm-array workflow for the first
Berzelius experiment: the official CFLOW-AD architecture on IAD.

## What is in the matrix

The locked config is `configs/iad_cflow_official_matrix.json`:

- input sizes: 64, 128, 256;
- modulations: 16QAM, CHIRP, GMSK, QPSK;
- seeds: 2026--2030;
- total: 60 independent cells, one GPU per cell.

Task IDs are stable:

| Task IDs | Input | Modulation/seed ordering |
|---|---:|---|
| 0--19 | 64 | 16QAM, CHIRP, GMSK, QPSK; five seeds each |
| 20--39 | 128 | same ordering |
| 40--59 | 256 | same ordering |

The 256 input, batch 32 row is the upstream default input/batch setting. The
64 and 128 rows are IAD resolution ablations. Batch 64 is necessary at input
64 because the WRN50-2 layer4 grid is 2x2 and the official flow fiber contains
256 patches.

The implementation follows the official repository at commit
`b2ebf9e673a0aa46992a3b18367ec066a57bba89`: WRN50-2 layer2/layer3/layer4,
three conditional FrEIA heads, eight AllInOne blocks per head, condition
dimension 128, clamp 1.9, fiber 256, 25 meta epochs x 8 sub epochs, cosine LR,
and the official two-meta-epoch warm-up.

The upstream BSD-3-Clause notice is retained in `berzelius/CFLOW_LICENSE.txt`.

IAD-specific protocol changes are explicit in every `run.json`:

- hash-grouped normal fit/calibration split;
- no validation-set tuning and no test-based checkpoint selection;
- test labels are used only after scores are fixed;
- the upstream test-set likelihood maximum is replaced by a fixed normal
  calibration maximum, so score scaling does not depend on the test set.

## 1. Transfer code and data

Example from the local machine (replace all angle-bracket values):

```bash
rsync -a --exclude outputs --exclude .git --exclude tmp \
  IAD_Publication/ \
  <user>@berzelius.nsc.liu.se:/proj/<project>/users/<user>/IAD_Publication/

rsync -a Datasets/IAD/ \
  <user>@berzelius.nsc.liu.se:/proj/<project>/users/<user>/Datasets/IAD/
```

Use `berzelius-hopper.nsc.liu.se` instead when the allocation is on Hopper.
Do not put the environment or dataset in the 20 GB home directory.

## 2. Create the environment once

On a Berzelius login node:

```bash
cd /proj/<project>/users/$USER/IAD_Publication
bash berzelius/prepare_cflow_env.sh
```

The environment and PyTorch cache are created under the project directory.
The setup script also downloads WRN50-2 weights once, before array jobs start.

## 3. Run a GPU smoke cell first

```bash
export BERZELIUS_ACCOUNT=<slurm-project-account>
export IAD_DATA_DIR=/proj/<project>/users/$USER/Datasets/IAD
export CFLOW_ARRAY_SPEC=40
export CFLOW_SMOKE=1
bash berzelius/submit_cflow_matrix.sh
```

Task 40 checks the official 256 pipeline using one meta/sub epoch and small
subsets. Its result is written below `outputs/iad_cflow_official_matrix/smoke`
and is always marked as a smoke-only result.

Monitor it with:

```bash
squeue -u $USER
jobgraph -j <job-id>
```

Unset smoke mode before the real matrix:

```bash
unset CFLOW_SMOKE CFLOW_ARRAY_SPEC
```

## 4. Submit the real matrix

Ampere/A100 thin nodes (default and lowest A100 GPU-hour cost):

```bash
export BERZELIUS_ACCOUNT=<slurm-project-account>
export IAD_DATA_DIR=/proj/<project>/users/$USER/Datasets/IAD
export BERZELIUS_SYSTEM=ampere
export CFLOW_MAX_CONCURRENT=4
bash berzelius/submit_cflow_matrix.sh
```

For Hopper, submit from a Hopper login node and use:

```bash
export BERZELIUS_SYSTEM=hopper
bash berzelius/submit_cflow_matrix.sh
```

Useful partial submissions:

```bash
# 64 only
CFLOW_ARRAY_SPEC='0-19%4' bash berzelius/submit_cflow_matrix.sh

# 128 only
CFLOW_ARRAY_SPEC='20-39%4' bash berzelius/submit_cflow_matrix.sh

# upstream-default 256 only
CFLOW_ARRAY_SPEC='40-59%4' bash berzelius/submit_cflow_matrix.sh

# retry selected failed cells
CFLOW_ARRAY_SPEC='3,17,42' bash berzelius/submit_cflow_matrix.sh
```

Completed cells are skipped by `--resume`. In-progress cells checkpoint every
five meta epochs. A successful cell deletes its large optimizer checkpoint
after atomically writing `run.json` and `scores.npz`.

Each job copies the four IAD pickle files from project storage to its isolated
`/scratch/local/$SLURM_JOB_ID` directory. The authoritative results remain in
the project directory because node-local scratch is erased after the job.

## 5. Aggregate and verify

Partial status can be collected at any time:

```bash
bash berzelius/collect_cflow_results.sh
```

Require all 60 cells before producing the final handoff:

```bash
bash berzelius/collect_cflow_results.sh --require-complete
```

The collector writes:

- `summary.csv`: one row per cell;
- `aggregate.json`: per-modulation and macro mean/std across seeds;
- `status.json`: completed count and missing task IDs.

## Resource and reproducibility notes

- The batch script requests one GPU and 48 hours. Berzelius currently permits
  at most 72 hours; override `--time` at submission if measured 256 jobs need
  a different limit.
- A100 thin nodes have 40 GB VRAM, well above the 2.1 GB observed in the local
  64 pilot. Actual 128/256 memory and wall time remain to be measured on the
  cluster.
- Use `jobgraph` to verify GPU utilization. The high-dimensional FrEIA soft
  permutations have a CPU-heavy initialization phase before training begins.
- The config hash, dataset SHA-256, source Git revision, package versions,
  GPU, Slurm IDs, timings, and peak memory are recorded per cell.

Current NSC references:

- https://www.nsc.liu.se/support/systems/berzelius-gpu/
- https://www.nsc.liu.se/support/systems/berzelius-getting-started/
- https://www.nsc.liu.se/support/systems/berzelius-software/berzelius-conda-mamba/
