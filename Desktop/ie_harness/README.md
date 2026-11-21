# IRIS-Harness

**IRIS-Harness: Harness-level Recursive Self-Improvement for Image Restoration with Self-Improvement Memory**

IRIS-Harness is a harness-optimization framework for low-level vision restoration
(dehazing, low-light enhancement, underwater enhancement, denoising). The restoration
backbone is kept **frozen**; only the surrounding **harness** (pre-processing, backend
selection, post-processing, inference TTA, and evaluation scheduling) is evolved
recursively. A structured **Self-Improvement Memory (SIM)** accumulates both successful
and failed edits across rounds and degradation types, so that past experience transfers
across backends and avoids repeated trials.

The paradigm follows Continual Recursive Self-Improvement (CRSI) with a SIM memory,
adapted from the agent-harness setting to the image restoration domain.

## Highlights

- **Frozen backbone, evolvable harness**: no weight update; the search acts on pre/post-processing, backend choice, and inference hyperparameters.
- **Structured SIM memory**: each edit records root cause, rationale, predicted and observed outcomes, verdict, and provenance; failures are retained for pruning.
- **Closed-loop Proposer–Selector–Evaluator**: root-cause-driven candidate generation, SIM-based pruning, hard-subset evaluation with a composite score and full-set rollback protection.
- **Plug-in backends**: lightweight deep models (Zero-DCE, FUnIE-GAN, AOD-Net) and classical operators (gray-world WB, CLAHE, MSR, bilateral, NLM) behind one unified interface.
- **Validated on real paired benchmarks**: LOL, a RESIDE-style dehazing set, EUVP paired underwater, and Kodak24 with synthetic noise.

## Repository Structure

```
ie_harness/
├── configs/harness_default.yaml      # Per-task baseline harness configs (evolution start)
├── ie_harness/
│   ├── backends/                     # Pluggable frozen backends
│   │   ├── base.py                   #   Unified interface: enhance(uint8 RGB) -> uint8 RGB
│   │   ├── zero_dce.py               #   Low-light Zero-DCE (79.4k params)
│   │   ├── funie_gan.py              #   Underwater FUnIE-GAN (7.02M params)
│   │   ├── aodnet.py                 #   Dehazing AOD-Net (1.8k params)
│   │   ├── classical.py              #   WB / CLAHE / MSR / bilateral / NLM
│   │   └── registry.py               #   Registry + instance cache
│   ├── pipeline.py                   # Configured pipeline: pre -> backend -> post
│   ├── degradations.py               # Synthetic haze / low-light / underwater / noise
│   ├── metrics.py                    # PSNR / SSIM / channel shift / contrast / UIQM
│   ├── proposer.py                   # Proposer: root-cause-driven candidate edits
│   ├── selector.py                   # Selector: SIM-history pruning (cross-backend)
│   ├── evaluator.py                  # Evaluator: batched metrics + verdict
│   ├── runner.py                     # CRSI closed loop + full-set rollback guard
│   └── sim/memory.py                 # SIM structured memory (JSON persistence)
├── scripts/
│   ├── prepare_data.py               # Build synthetic paired data
│   ├── run_experiment.py             # Closed loop on synthetic data
│   ├── build_report.py               # Synthetic-data Markdown report
│   ├── prepare_real_data.py          # Download/extract real paired samples
│   ├── run_real_experiment.py        # Closed loop on real benchmarks
│   └── build_real_report.py          # Real-benchmark report + metrics + figures
├── weights/                          # Frozen checkpoints (see download table below)
├── data/{gt,degraded}                # Synthetic paired data
├── data/real/<task>/{input,gt}       # Real paired samples
├── sim_store/                        # SIM memory stores (resumable)
├── results/real/                     # Real-benchmark figures, CSV, JSON, report
└── paper/                            # Single-column paper source and compiled PDF
```

## Installation

```bash
conda create -n iris python=3.10 -y && conda activate iris
pip install torch torchvision opencv-python-headless pillow pyyaml \
            scikit-image scipy numpy
```

## Quick Start

### Real benchmarks (main results)

```bash
cd ie_harness
python3 scripts/prepare_real_data.py                      # 1. Fetch/extract real paired samples
python3 scripts/run_real_experiment.py --rounds 2 --fresh # 2. Two-round closed loop (--fresh resets SIM)
python3 scripts/build_real_report.py                      # 3. Report, CSV, and comparison figures
```

Drop `--fresh` to resume and accumulate experience in the existing SIM store.

### Synthetic prototype

```bash
python3 scripts/prepare_data.py
python3 scripts/run_experiment.py --rounds 2 --fresh
python3 scripts/build_report.py
```

## Real Datasets

| Task | Dataset | Pairs | Source |
|---|---|---|---|
| Low-light | LOL test (paired) | 15 | HF `ishicode/low-light-dataset` |
| Dehazing | RESIDE-style paired set | 100 | HF `linxin020826/dehazing_test` |
| Underwater | EUVP paired (256x256) | 40 | HF `Ken1053/EUVP` (parquet) |
| Denoising | Kodak24 clean + Gaussian noise sigma=25 | 24 | HF `msdkhairi/kodak` |

## Backend Weights

Checkpoints are not bundled in git; download the verified URLs below into `weights/`.

| Backend | URL |
|---|---|
| Zero-DCE | `https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth` |
| FUnIE-GAN | `https://raw.githubusercontent.com/xahidbuffon/funie-gan/master/PyTorch/models/funie_generator.pth` |
| AOD-Net | `https://raw.githubusercontent.com/zhuleyang/aodnet/master/model_pretrained/final_model.pth` |

## Closed-Loop Logic

1. **Proposer** analyzes failure root causes (dark / bright / low-contrast / color cast / haze residual) and emits single-factor edits from the task candidate space, keeping changes attributable.
2. **Selector** queries SIM: previously `regressed` `(edit key, value)` pairs are pruned directly; historically `improved` pairs are prioritized. Experience transfers across backends.
3. **Evaluator** measures candidates on the hard subset (worst 50% by PSNR) with a composite score and hard-regression guards; the winning harness is re-evaluated on the full set, with rollback protection against hard-subset overfitting.
4. Every outcome, including failures, is written back to SIM.

## Main Results (full set, mean +/- std)

| Task | Backend (B->E) | PSNR B | PSNR E | dPSNR | SSIM B | SSIM E | dSSIM | Shift B | Shift E | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Low-light | zero_dce | 14.94 | 18.67 | +3.73 | 0.653 | 0.702 | +0.049 | 45.64 | 25.84 | Harness evolved (gain+gamma) |
| Dehazing | aodnet | 21.08 | 21.07 | -0.01 | 0.897 | 0.909 | +0.011 | 12.87 | 10.68 | Harness evolved (gamma + channel gain) |
| Underwater | funie_gan | 19.45 | 20.21 | +0.77 | 0.752 | 0.803 | +0.051 | 13.16 | 13.23 | Harness evolved (TTA flip) |
| Denoising | nlm | 24.80 | 24.80 | +0.00 | 0.647 | 0.647 | +0.000 | 1.38 | 1.38 | Baseline retained |

## Adding a New Backend

1. Add a class under `backends/` inheriting `EnhancerBackend`, implementing `enhance(img) -> img` (uint8 HWC RGB).
2. Register the name and task in `registry.py`.
3. Configure the baseline in `configs/harness_default.yaml`, or add switchable candidates in `proposer.py`.

## Limitations

- Deep backends are lightweight legacy models; the ceiling of some tasks is set by the frozen backbone.
- Classical denoising/dehazing alternatives are weak; stronger frozen backends (RIDNet, lightweight Transformer) can be plugged in for comparison.
- Metrics do not yet include LPIPS / CLIP-IQA or subjective scoring; the candidate space is single-factor plus a few two-factor combinations.
- EUVP uses a 40-pair sample; full EUVP and UIEB / SID / SOTS benchmarks are future work.

## Citation

```bibtex
@article{zram2026iris,
  title  = {IRIS-Harness: Harness-level Recursive Self-Improvement for Image Restoration with Self-Improvement Memory},
  author = {zram},
  year   = {2026}
}
```

## License

Released under the MIT License. See `LICENSE`.
