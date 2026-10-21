# AI Assistance and Tool Usage Disclosure

This document details the usage of AI-assisted engineering tools during the design, development, and testing of the **IncuBrix AI Background Music Engine**.

---

## 1. AI Coding Assistants Used
* **Model / Assistant:** Google Antigravity (Gemini 3.8 Flash model family)
* **Environment:** Antigravity IDE paired with automated terminal tool invocation, ripgrep code inspection, and file editing systems.

---

## 2. Methodology & Contributions

### Architecture & System Design
* **Interface Specification:** Antigravity was directed to enforce strict separation of concerns between music generation (`BaseMusicProvider`) and stem separation (`BaseStemSeparator`), preventing architectural coupling.
* **Master Prompt Execution:** The assistant verified gated model repositories (`stabilityai/stable-audio-3-small-music`) and licensing terms before writing inference adapters.

### Audio Engineering & Digital Signal Processing (DSP)
* **Loudness Compliance:** Developed ITU-R BS.1770 compliant normalization routines using `pyloudnorm` paired with soft-knee polynomial transient limiting to guarantee `-16.0 LUFS (±1.0 LU)` while maintaining `True Peak <= -1.0 dBTP`.
* **Speech Envelope Ducking:** Designed a dynamic voice activity detection (VAD) filter applying asymmetric attack (50ms) and release (250ms) curves to lower music by 10 dB without ducking distortion or pumping.
* **Phase Alignment & Stem Reconstruction:** Implemented SNR (Signal-to-Noise Ratio) and RMS difference error computations to mathematically verify that the sum of extracted stems reconstructs the master mix within documented tolerances.

### Testing & Verification
* **Test Suite Authoring:** Synthesized an automated test harness covering all 18 mandatory evaluation criteria, including duration limits (5s to 300s), duplicate idempotency checks, SSRF attack prevention, non-English prompts, and stem isolation checks.
* **Debugging & Refinement:** Diagnosed and corrected test database lifecycle race conditions and numpy boolean type assertions to achieve a 100% test pass rate across all 40 unit and integration tests.

---

## 3. Human Oversight & Engineering Verification

* All generated code was inspected and validated against real execution outputs.
* Docker Compose files, environment configurations, and audio file outputs were tested locally.
* Licensing conditions were independently verified against Stability AI and Meta AI official repositories.
