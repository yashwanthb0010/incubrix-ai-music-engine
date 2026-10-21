# Sources, Models, Dependencies, and Commercial Rights

This document provides a comprehensive inventory of all machine learning models, third-party libraries, datasets, and API specifications utilized in the **IncuBrix AI Background Music Engine**, including license terms, commercial viability, and operational constraints.

---

## 1. AI Music Generation Models

### Primary Intended Backend: `stabilityai/stable-audio-3-small-music`
* **Repository / Model Card:** [`https://huggingface.co/stabilityai/stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music)
* **Architecture:** Diffusion Transformer (DiT) operating on the SAME-S (Semantically-Aligned Music Autoencoder - Small) latent space. Parameter count ~0.6B.
* **Native Audio Output:** 44.1 kHz, stereo 2-channel.
* **Native Duration Window:** Variable, up to 120.0 seconds in a single pass. Durations beyond 120s up to 300s are handled by IncuBrix via segmented inpainting/crossfade extension.
* **Gating & Authentication:** Gated model repository. Access requires a Hugging Face user account, acceptance of terms on the model page, and setting the `HF_TOKEN` environment variable.
* **License:** **Stability AI Community License**
* **Commercial Rights & Limitations:**
  - **Permitted Free Commercial Use:** Organizations and individuals with annual gross revenue below **$1,000,000 USD** may generate and deploy commercial audio outputs freely without royalties.
  - **Enterprise Restriction:** Any commercial deployment by entities with annual revenue exceeding **$1,000,000 USD** requires an Enterprise Agreement directly with Stability AI.
  - **Outputs Ownership:** Users retain ownership of generated audio tracks subject to compliance with acceptable use policies (no impersonation of living artists or training derivative foundational base models).

### Deterministic Mock Engine: `incubrix-mock-synth-v1`
* **Source:** Internal procedural harmonic synthesizer (`incubrix.providers.mock_provider`).
* **Architecture:** Multi-oscillator additive harmonic synthesis with beat-synchronized rhythm accumulators, exponential pitch envelopes (percussion), and tempo/key modulation.
* **Output:** 48 kHz stereo, 5 to 300 seconds.
* **License:** MIT License. 100% free for all testing, development, and benchmarking with zero cloud costs or API keys.

---

## 2. Music Source Separation Models

### Meta AI Demucs v4: `htdemucs` / `htdemucs_6s`
* **Repository:** [`https://github.com/facebookresearch/demucs`](https://github.com/facebookresearch/demucs)
* **Architecture:** Hybrid Transformer Demucs (HTDemucs), combining bi-directional multi-scale spectrogram transformers and time-domain convolutional U-Nets.
* **Extracted Stems:**
  - 4-stem model (`htdemucs`): `drums`, `bass`, `other`, `vocals`.
  - 6-stem model (`htdemucs_6s`): `drums`, `bass`, `other`, `vocals`, `guitar`, `piano`.
* **Dataset:** Trained on MUSDB-HQ (150 full-length multi-track music recordings).
* **License:** **MIT License** for the code and model weights released by Meta AI research.
* **Commercial Viability:**
  - MIT license allows commercial use, modification, and integration into backend pipelines.
  - In background music workloads, vocals are non-existent or suppressed; `drums`, `bass`, and `other` constitute genuine, acoustically isolated instrument parts.
* **Stem Reconstruction Notice:** Stem separation is lossy by definition; phase cancellation and masking across overlapping instrument spectra lead to minor reconstruction residuals (typically measured at 25–40 dB SNR). The IncuBrix manifest records the exact signal-to-noise ratio and RMS error for every completed job.

---

## 3. Core Software Libraries & Dependencies

| Library | Version | License | Primary Purpose in IncuBrix |
| :--- | :--- | :--- | :--- |
| **Python** | 3.11+ / 3.14 | PSF License | Runtime language environment |
| **FastAPI** | >= 0.110.0 | MIT | Asynchronous REST HTTP API server, OpenAPI 3.1 generation |
| **Pydantic** | >= 2.6.0 | MIT | Strict request, response, and manifest data validation |
| **SoundFile** | >= 0.12.1 | BSD 3-Clause | Uncompressed 16-bit / 24-bit PCM WAV audio read/write |
| **NumPy** | >= 1.26.0 | BSD 3-Clause | Vectorized waveform manipulation, panning, mixing |
| **SciPy** | >= 1.12.0 | BSD 3-Clause | Polyphase audio resampling, moving RMS envelope filtering |
| **pyloudnorm** | >= 0.1.1 | MIT | ITU-R BS.1770-4 integrated loudness (LUFS) compliance meter |
| **FFmpeg** | 7.x / imageio-ffmpeg | LGPL 2.1+ / GPL | Conversion of master mix to high-bitrate MP3 |
| **MinIO Python SDK** | >= 7.2.0 | Apache 2.0 | S3-compatible cloud object storage integration |
| **Redis** | >= 5.0.0 | MIT | Distributed job queue and pub/sub message broker |
| **SQLite3** | Built-in | Public Domain | Lightweight ACID metadata and idempotency store |
| **httpx** | >= 0.27.0 | BSD 3-Clause | SSRF-validated asynchronous HTTP client for narration fetching |
| **Pytest** | >= 8.0.0 | MIT | Test runner for unit, integration, and e2e test suite |

---

## 4. Compliance and Commercial Rights Summary

1. **Zero Proprietary Locked APIs:** The system does not depend on closed-source black-box music generation APIs.
2. **Transparent Licensing:** All code developed for this assessment is licensed under the MIT License.
3. **No Unverified Claims:** The system explicitly records whether stems were generated via source separation or multi-track synthesis, logs reconstruction error, and stamps all output manifests with an unambiguous `ai_generated_media_label`.
