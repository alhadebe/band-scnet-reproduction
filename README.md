# Band-SCNet independent reproduction

A from-paper PyTorch research prototype for potential Mixxx stem preparation. The Phase 1 benchmark uses random weights; do not use those outputs as real stems. No upstream checkout, weights, or dataset needed for benchmarking. Default architecture has 2,591,062 parameters. Read [DESIGN.md](DESIGN.md) for source references, assumptions, and local test results.

## Phase 2: train on music

See [TRAINING.md](TRAINING.md) for dataset setup, first GPU smoke run, checkpoint/resume, TensorBoard, listening tests, museval evaluation and trained ONNX export. Training is experimental; no usable checkpoint or SDR result is provided yet. Keep MUSDB18-HQ audio and its research checkpoints private until rights are resolved.

## Run on Linux with Docker Compose

Clone the repository and enter the directory:

```bash
mkdir -p ~/dockerapps
cd ~/dockerapps
git clone https://github.com/alhadebe/band-scnet-reproduction.git
cd band-scnet-reproduction
```

CPU first (Intel or AMD):

```bash
docker compose --profile cpu run --build --rm cpu audit --seconds 1
docker compose --profile cpu run --rm cpu bench --seconds 11 --threads 1 --profile-flops
docker compose --profile cpu run --rm cpu bench --seconds 11 --threads 4
```

NVIDIA GPU (requires NVIDIA Container Toolkit and driver supporting CUDA 12.8; PyTorch cu128 and ORT 1.22 CUDA12/cuDNN9 are paired):

```bash
docker compose --profile gpu run --build --rm gpu audit --seconds 1
docker compose --profile gpu run --rm gpu bench --seconds 11
```

The build downloads dependencies. Subsequent runs need no upstream repository or network access for inference. Model initialization is random, seed 42. Benchmark measures median of three runs after one warmup, synchronizes CUDA before/after inference and separates network-only from FFT/inference/iFFT times. It does not include disk or track decoding; do not directly compare it with the full-track JavaScript SCNet benchmark.

## ONNX export and benchmark

Export CPU first. This generates a fixed-length real-valued spectrogram graph, runs ONNX checker and checks numerical parity with PyTorch. The 1-second graph is just a smoke test; the 11-second graph is the main benchmark.

```bash
docker compose --profile cpu run --rm cpu export --seconds 1
docker compose --profile cpu run --rm cpu export --seconds 11
docker compose --profile cpu run --rm cpu onnx --seconds 11 --threads 1
docker compose --profile gpu run --rm gpu onnx --seconds 11
```

ONNX files persist in `models/`. JSON reports appear in `output/`. CPU and GPU use the same exported file. ONNX CPU and CUDA inference have been exercised on the i9-9900K/RTX 3090 host. Trained checkpoint export and parity still require testing. The GPU runner verifies that CUDA is the first active provider but unsupported operators may still run on CPU. ORT uses host-side audio transforms and tensor transfers; PyTorch GPU benchmark keeps these tensors on GPU. A direct comparison requires acknowledging that difference.

## Training readiness smoke test

```bash
docker compose --profile cpu run --rm cpu train-smoke --seconds 0.25
docker compose --profile gpu run --rm gpu train-smoke --seconds 1
```

This only tests backward propagation and a single optimizer update on synthetic random signals. It does not train a useful separator. For real music training, follow [TRAINING.md](TRAINING.md).

## Project files

- `model.py`: independent encoder, bottleneck, CSA, decoder and audio transforms.
- `run.py`: architecture audit, benchmark, synthetic training test and ONNX export/parity.
- `Dockerfile` and `compose.yaml`: CPU and NVIDIA profiles, pinned direct dependencies.
- `DESIGN.md`: architecture decisions, source references and measured local results.
- `models/`, `output/`: persistent artifacts, excluded from Git.

See the paper: https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf
