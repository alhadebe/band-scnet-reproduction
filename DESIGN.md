# Architecture and reproduction notes

This is a new implementation based on the published description and diagrams, not a port of the unofficial Band-SCNet GitHub code. No upstream source files or checkpoints are bundled or fetched. It is a research reproduction, not an official release or a demonstrated equivalent trained separator.

## Sources

- Yang et al., Band-SCNet, Interspeech 2025, DOI 10.21437/Interspeech.2025-448: https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf (Sections 2–4, Figures 2–3, Table 1).
- Tong et al., SCNet, ICASSP 2024: https://arxiv.org/html/2401.13276v1 (encoder, convolution module, inherited STFT and sparse compression design). The official SCNet configuration also documents band kernels 3/4/16: https://github.com/starrytong/SCNet/blob/main/scnet/SCNet.py.

## Reported versus implemented

| Item | Source | Implementation |
|---|---|---|
| FFT/hop, sample rate | Band-SCNet §4 | 4096/1024, 44100 Hz |
| Encoder widths, split ratios, strides, block depths | §3.2, §4.2 | 4/32/64/128; .175/.392/.433; 1/4/16; 3/2/1 |
| Down/up kernels | inherited SCNet config | 3/4/16; exact symmetric frequency padding/cropping, floor low/mid split |
| Encoder convolution module | inherited SCNet design | causal bottleneck GLU, depthwise temporal convolution, cumulative normalization, pointwise projection and residual; details inferred |
| Bottleneck | §3.3 | frequency-grouped convolution, per-channel full-band linear, frequency-grouped convolution; temporal attention and temporal grouped FFN, residual at each submodule |
| C', C'', groups, kernel sizes | §4.2 | 8, 256, 8, frequency 3/time 5 |
| CSA | §3.4 | Q/K channels per head ceil(1024/F); V channels per head C/4; causal temporal attention across flattened frequency-feature vectors |
| CSA linear gate | §4.2 | frequency down/up only at first (highest-frequency) decoder level; stride/kernel 2; pad odd frequency length and crop |
| CSA projection and GLU kernels | not fully specified | pointwise Q/K/V/output; causal 3x3 GLU convolution, pointwise final convolution |
| Number of separation pairs L | unspecified | 4 by default, configurable; chosen as plausible hypothesis, matches rounded reported size |
| Normalization | paper specifies LN/GN and causal adaptation | per-frame channel LN in bottleneck/fusion; cumulative moments in encoder and FFN group normalization to avoid future leakage; exact axes/epsilon inferred |
| Dropout | rate unspecified | 0 by default, configurable at model constructor |
| Window and scaling | inherited SCNet; Band-SCNet does not fully specify | rectangular, normalized, center=False, tail pad to frame boundary |
| Sources | Fig. 2 | vocals, bass, drums, other; real/imag interleaved per stereo channel |
| Loss | §4.2 | RMSE over real/imag spectral targets for synthetic gradient smoke test; exact reduction convention assumed |

We intentionally do not claim checkpoint compatibility, reproduction of the paper's SDR, or exact equivalence from matching parameter count. No real dataset training or held-out evaluation has been performed. The causal test verifies a finite prefix perturbation at the spectrogram level, not a stateful audio streaming implementation or 92 ms system latency. Attention currently runs over the whole segment, with causal masking and no KV cache. For Mixxx load-time processing, use overlapping chunks and stitch outputs later; that full-track engine is not in this phase.

## Local validation (CPU, PyTorch 2.10.0+cpu)

- Default four-pair configuration: 2,591,062 trainable parameters (paper: 2.59M rounded).
- Encoder frequency sizes: 2049 -> 615 -> 185 -> 56.
- One-second input: [1,4,41,2049] -> [1,16,41,2049] -> [1,4,2,44100]. All outputs finite.
- Perturb future input frames: maximum absolute change in preceding output frames = 0.0 in this test.
- Audio transform roundtrip max absolute error: 8.94e-8 (random waveform scaled by 0.1).
- Synthetic complex spectral loss: one Adam step with finite gradients for all parameters, passed.
- 11-second input, one CPU thread, one measured run, no warmup: network 5.485 s, full FFT/inference/iFFT 5.606 s (RTF 0.510). Hosted CPU environment is not the paper's i7 or the user's i9. Do not compare speeds as a controlled hardware benchmark.
- `torch.utils.flop_counter.FlopCounterMode`: 17.4476 counted GFLOPs per audio second for 11 s, close to paper's 17.59. Counter omits certain operators; not a complete FLOP accounting. Small discrepancies can reflect architecture choices or framing.
- ONNX and ONNX Runtime are unavailable in this local environment; package installation failed. Docker and NVIDIA execution not locally tested. Export code is experimental and should be validated on the target machines.

## Future work

1. Confirm missing design settings with authors, especially L and causal normalization.
2. Run 11 s CPU/CUDA benchmarks and ONNX parity on target hardware.
3. Add licensed dataset ingestion, real training, validation split, checkpoint/resume, and museval SDR. No need to download MUSDB until performance and licensing have been reviewed.
4. Add overlap-add track processing, profiling, and optional non-causal experimental branch, then integrate with Mixxx.
