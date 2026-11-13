# IAD Spectrogram Anomaly Detection

Reproducible normal-only anomaly detection on the ISM Anomaly Detection (IAD)
spectrogram dataset. The benchmark compares four detector paradigms:

| Method | Paradigm | Input |
| --- | --- | ---: |
| PatchCore | pretrained feature memory | 64 x 64 |
| EfficientAD | student--teacher | 256 x 256 |
| CFLOW-AD | feature-density estimation | 64 x 64 |
| VAE-PER | reconstruction | 64 x 64 |

All methods fit only normal training samples. The main protocol uses the full
normal reference pool, a fixed normal test set, four unseen anomaly conditions,
and five independent fitting seeds. Test labels are used only for evaluation.

This repository contains code, configurations, tests, and portable launch tools.
Datasets, model weights, run outputs, manuscript sources, and generated figures
are intentionally not versioned.

## Repository layout

```text
configs/                 versioned experiment configurations
core/                    shared PatchCore and metric utilities
dataio/iad/              safe IAD loading, discovery, and content hashing
iad_detection/           detector implementations and protocol utilities
iad_labeling/            optional local annotation and QC tools
scripts/                 benchmark, collection, audit, and plotting commands
licenses/                retained third-party license texts
tests/                    unit and protocol-integrity tests
outputs/                  generated run artifacts (ignored)
```

The older SAS/open-world modules remain in the repository for compatibility
with the parent thesis project but are not required by the IAD benchmark.

## Dataset

Download IAD using the links supplied by the dataset authors:

<https://github.com/QXSLAB/vae_ism_ano/blob/main/dataset.md>

Place the four released pickle files together in one directory. The loader
discovers the 16QAM, CHIRP, GMSK, and QPSK sources from their filenames and
validates their contents before use. Raw dataset files are not included here.

## Local setup

Python 3.11 is recommended.

```bash
conda env create -f environment.yml
conda activate iad-anomaly-detection
```

The checked-in `requirements.txt` contains the dependencies used by the IAD
code. CFLOW-AD requires `FrEIA`; it is included in that file. If the Conda
environment does not select the appropriate GPU-enabled PyTorch build for your
system, install the build matching your CUDA runtime before running the matrix.

Run the test suite:

```bash
python -m pytest -q
```

EfficientAD additionally requires checksum-verified teacher weights and the
ImageNette auxiliary dataset. Prepare them once with:

```bash
python scripts/prepare_iad_assets.py --asset-root .cache/iad_assets
```

## Quick smoke run

Task 0 is PatchCore with seed 2026. The command below uses reduced subsets and
training to verify data loading, fitting, scoring, and result serialization:

```bash
python scripts/run_iad_benchmark.py \
  --config configs/iad_benchmark.json \
  --task-kind gpu \
  --task-id 0 \
  --dataset-dir /path/to/IAD \
  --asset-root .cache/iad_assets \
  --output-root outputs/iad_benchmark \
  --device cuda \
  --workers 0 \
  --smoke
```

List the complete method-by-seed task map without starting training:

```bash
python scripts/run_iad_benchmark.py \
  --config configs/iad_benchmark.json \
  --task-kind gpu \
  --list-tasks
```

## Complete benchmark matrix

The portable launcher executes the configured method-by-seed cells sequentially
and resumes completed cells safely:

```bash
python scripts/run_iad_matrix.py \
  --config configs/iad_benchmark.json \
  --dataset-dir /path/to/IAD \
  --asset-root .cache/iad_assets \
  --output-root outputs/iad_benchmark \
  --device cuda \
  --workers 8 \
  --resume
```

Use `--task-ids 0 1 2` to run only selected cells. On a multi-GPU workstation
or a managed cluster, launch disjoint task-ID sets in separate processes and
assign each process to one GPU using the platform's standard mechanism. The
benchmark runner itself has no dependency on Slurm or a specific cluster.

Collect and validate a completed matrix:

```bash
python scripts/collect_iad_full_training_results.py \
  --config configs/iad_benchmark.json \
  --result-root outputs/iad_benchmark \
  --output-dir outputs/iad_benchmark_summary
```

## Reproducibility and outputs

Each completed cell records the configuration hash, source-data hashes, code
snapshot hash, seed, package and device information, timings, metrics, and raw
sample scores. Writes are atomic, and `--resume` accepts only matching run
identities. Generated artifacts are written under `outputs/` and are ignored
by Git.

## Licensing and attribution

Project-owned code is released under the MIT License. Adapted third-party
components retain their original notices; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The IAD dataset is obtained
separately and remains subject to its authors' terms.
