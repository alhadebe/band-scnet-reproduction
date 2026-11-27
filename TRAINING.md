# Phase 2: research training

## Scope and data rights

Yang et al., Interspeech 2025 (https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf), sections 4.1–4.2: MUSDB18-HQ, stereo 44.1 kHz, overlapping 11-second segments, source remixing, scaling, complex spectrogram RMSE, museval SDR. The paper does not specify optimizer, batch size, augmentation probabilities, overlap amount, learning-rate schedule or epochs. Our choices below are experimental, not claimed as the paper's exact settings.

MUSDB18-HQ: https://zenodo.org/records/3338373 (DOI 10.5281/zenodo.3338373). Educational use only; commercial use of the material requires permission. Do not publish dataset audio or research checkpoints/ONNX files until rights for redistribution and downstream use have been reviewed. The MIT repository licence covers code, not the dataset or resulting checkpoint rights.

We use the standard MUSDB 86/14 validation split from https://github.com/sigsep/sigsep-mus-db/blob/master/musdb/configs/mus.yaml. The separate 50-song test split is not used for training or model selection.

## Dataset setup (on host)

Review the dataset terms first. The archive is 22.7 GB. Allow roughly 50 GB or more for archive plus extracted WAVs, in addition to Docker and checkpoint storage. Download via the official Zenodo page or run:

```bash
mkdir -p dataset/musdb18hq
curl -fL -C - https://zenodo.org/records/3338373/files/musdb18hq.zip -o dataset/musdb18hq.zip
echo '12d4f2ecd55245a4688754dd76363103  dataset/musdb18hq.zip' | md5sum -c -
unzip dataset/musdb18hq.zip -d dataset/musdb18hq
```

Ensure the dataset root contains train/ and test/ directly, each containing song directories with mixture.wav, vocals.wav, bass.wav, drums.wav, other.wav. If the archive extracts an extra folder, set MUSDB_ROOT in .env to the directory containing train/ and test/, or adjust the folder. Do not move validation tracks out of train/. The loader makes that split internally. Dataset is mounted read-only.

## Build and validate

```bash
docker compose --profile gpu build gpu
docker compose --profile gpu run --rm --entrypoint 'python /app/train.py' gpu --check-data
```

The check verifies WAV metadata for all 150 tracks, sample rates, channels and aligned lengths. It does not decode every sample. Next run a short full-length real-data training and validation test (a distinct checkpoint directory):

```bash
docker compose --profile gpu run --rm --entrypoint 'python /app/train.py' gpu --run /checkpoints/smoke --epochs 1 --steps 2 --accum 1 --val-clips 1 --precision fp32 --save-every 1
```

Expect 2 updates and 14 validation clips, finite loss, latest.pt and best.pt. This is not a useful trained model. Do not start a long run until the smoke test succeeds.

## Start training

```bash
docker compose --profile gpu run --rm --entrypoint 'python /app/train.py' gpu --epochs 100 --precision bf16
```

Defaults: 11 seconds, batch 1, accumulate 4 microbatches per update, Adam LR 0.0005, 500 updates/epoch, gradient norm clip 5, plateau LR reduction (factor 0.5, patience 5), 50% source remix chance, +/-6 dB random gains, 2 deterministic validation clips per song, save every 100 updates and at epoch end. Random crops can overlap. Training mixture is the sum of augmented targets; validation uses the actual mixture.wav and aligned sources. These are chosen defaults; change flags if needed. BF16 is optional on Ampere; FP32 is the reference and fallback. If out of memory, try FP32 with a shorter segment for debugging, or BF16 and batch 1; do not silently compare shorter training segments to the paper. No runtime estimate until actual update timing is measured.

Checkpoints are stored locally under checkpoints/bandscnet/. latest.pt contains model, Adam, LR scheduler, AMP scaler, configuration, epoch, completed update count and RNG states. best.pt is selected by validation RMSE. Atomic saves occur at completed optimizer boundaries. Ctrl-C requests a graceful checkpoint at the next boundary; if the container is forcibly killed, resume from the last completed save. To resume the normal run:

```bash
docker compose --profile gpu run --rm --entrypoint 'python /app/train.py' gpu --epochs 100 --precision bf16 --resume
```

Keep all training flags identical except total epochs, worker count, thread count and checkpoint frequency. Use a new --run directory for changes to batch size, precision, steps, segment length or augmentation. Do not run two trainers writing to the same directory simultaneously. Only load trusted checkpoints. Resumption preserves optimizer state and sample order; exact bitwise GPU reproducibility is not guaranteed. Changing hardware/kernel versions can change results.

## TensorBoard

In a separate terminal on the server:

```bash
docker compose --profile gpu run --rm --no-deps -p 127.0.0.1:6006:6006 --entrypoint tensorboard gpu --logdir /checkpoints --host 0.0.0.0 --port 6006
```

If accessing from another machine, use an SSH tunnel (replace fish with your SSH host):

```bash
ssh -L 6006:127.0.0.1:6006 fish
```

Open http://localhost:6006 locally. Track train/validation RMSE, learning rate, gradient norm, update duration and per-source clip SDR proxy. Clip SDR is a monitoring signal, NOT the paper's BSS Eval SDR.

## Listening and full-track evaluation

After real training, use the best checkpoint to write the four stems for a validation song (or provide --track with its exact name). For your own 44.1kHz stereo WAV, mount via ./audio and pass --input /audio/test.wav.

```bash
docker compose --profile gpu run --rm --entrypoint 'python /app/evaluate.py' gpu listen --checkpoint /checkpoints/bandscnet/best.pt --split val
docker compose --profile gpu run --rm --entrypoint 'python /app/evaluate.py' gpu evaluate --checkpoint /checkpoints/bandscnet/best.pt --split val --limit 1
```

The evaluator uses 11-second chunks, 2-second linear crossfades and normalization of overlap weights. These are reproduction choices; the paper does not specify its precise overlap or aggregation. The full-track museval BSS Eval v4 runs on CPU and can be slow. After selecting the model on validation data, evaluate the full test split once:

```bash
docker compose --profile gpu run --rm --entrypoint 'python /app/evaluate.py' gpu evaluate --checkpoint /checkpoints/bandscnet/best.pt --split test
```

Results in output/evaluation/test-sdr.json include track medians, mean/median across tracks and the mean of stem medians. Compare cautiously with the paper's 7.79 dB. No SDR guarantee until training and evaluation.

## Export a trained checkpoint to ONNX

Build CPU image once if needed. This will preserve the random-weight graph and use a distinct name/directory. Use the same checkpoint and flags for benchmarking.

```bash
docker compose --profile cpu run --build --rm cpu export --seconds 11 --checkpoint /checkpoints/bandscnet/best.pt --model-dir /models/trained
docker compose --profile gpu run --rm gpu onnx --seconds 11 --checkpoint /checkpoints/bandscnet/best.pt --model-dir /models/trained
```

Export checks numerical parity in CPU FP32. To export a differently trained model, use a unique --model-dir. A benchmark with --checkpoint assumes the ONNX file was exported from that exact checkpoint; re-export after changing it. The graph expects 471 frames for 11 seconds. This is research-only until data rights are resolved.

## Verification state

Phase 1 CPU/CUDA PyTorch and ONNX inference verified on the user's host. Phase 2 training, AMP, real-data validation, museval and trained export are newly implemented and require target-machine testing. No music has been downloaded, no training started, no useful checkpoint produced yet.
