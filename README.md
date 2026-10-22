# IncuBrix AI Background Music Engine for Video

Production-oriented, asynchronous AI background music generation engine with genuine instrument stem separation, section mood transitions, and narration ducking. Built for the IncuBrix Engineering Assessment 1.

![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)
![Tests](https://img.shields.io/badge/Tests-40%2F40%20Passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## Key Capabilities

- **Instrumental Music Generation:** Supports text prompts, tempo (BPM), key, durations from 5 to 300 seconds, and seamless looping.
- **Genuine Instrument Stems:** Extracts at least three distinct, independently playable stems (`drums`, `bass`, `other`/synths) rather than arbitrary frequency filters.
- **Stem Reconstruction Audit:** Mathematically measures reconstruction fidelity (RMS error and SNR) between the reconstructed sum of stems and the final mix.
- **Broadcast Audio Quality:** Guarantees 48 kHz stereo WAV, optional MP3, integrated loudness of **-16.0 LUFS (±1.0 LU)**, True Peak **≤ -1.0 dBTP**, and duration accuracy within **0.05s**.
- **Section-Aware Transitions:** Follows timestamped section mood changes aligned within 0.5 seconds of requested boundaries.
- **Narration Ducking:** Speech-envelope detection automatically attenuates music by 8–15 dB during voiceover, exporting clean `mix_ducked.wav`.
- **Complete Manifest:** Every completed job produces `music_manifest.json` detailing loudness measurements, true peak, stem metrics, and licensing audit.
- **Zero-Dependency Mock Engine:** Deterministic, beat-synced harmonic mock synthesizer runs all 40 tests and batch jobs offline without external API keys or GPU.
- **Interactive Audio Studio Web UI:** Browser interface for real-time generation, waveform playback, stem inspection, and manifest visualization.

---

## Architecture

```
[ Client Request ] ────► [ POST /api/v1/music/jobs ] (Idempotency-Key enforced)
                                  │
                                  ▼
                         [ FastAPI Gateway ] ──► HTTP 202 Accepted (with job_id)
                                  │
                                  ▼
                        [ Async Worker Queue ]
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
[ Music Provider ]       [ Audio Processing DSP ]  [ Stem Separator ]
• Stable Audio 3 Small   • Polyphase Resample      • Meta Demucs v4
• Deterministic Mock     • -16 LUFS Normalization  • Harmonic Synthesis
                         • Soft Peak Limiter       • Aligns & Measures
                         • Musical Boundary Fade     Reconstruction SNR
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[ Narration Ducking ]                             [ Storage & Manifest ]
• Speech Envelope VAD                             • MinIO S3 / Local
• 10 dB Attenuation                               • mix.wav, mix.mp3, stems
• mix_ducked.wav export                           • music_manifest.json
```

---

## Quickstart

### 1. Local Development (Zero External Dependencies)

1. Clone or navigate to the repository:
   ```bash
   cd "incubrix 1"
   ```

2. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:
   ```bash
   uvicorn incubrix.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. Open your browser:
   - **Interactive Studio UI:** [http://localhost:8000/](http://localhost:8000/)
   - **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **OpenAPI 3.1 YAML:** [http://localhost:8000/openapi.yaml](http://localhost:8000/openapi.yaml)

---

### 2. Docker & Docker Compose

To start the API, background worker, MinIO S3 object storage, and Redis:
```bash
docker compose up --build
```
- API is available at `http://localhost:8000`
- MinIO S3 Console at `http://localhost:9001` (user: `minioadmin`, pass: `minioadmin`)

---

## API Reference

### 1. Submit Music Job
`POST /api/v1/music/jobs`  
**Headers:** `Idempotency-Key: <unique-uuid-or-token>` (Required)

```bash
curl -X POST http://localhost:8000/api/v1/music/jobs \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: job_req_001" \
  -d '{
    "prompt": "Warm, optimistic acoustic pop with upbeat guitars for a product launch",
    "duration_s": 45.0,
    "tempo_bpm": 112,
    "key": "C major",
    "seed": 42,
    "sections": [
      {"start_s": 0.0, "mood": "calm intro"},
      {"start_s": 15.0, "mood": "energetic build"},
      {"start_s": 30.0, "mood": "uplifting finish"}
    ],
    "output": {
      "sample_rate": 48000,
      "target_lufs": -16.0,
      "formats": ["wav", "mp3"]
    }
  }'
```

**Response (HTTP 202 Accepted):**
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "queued",
  "status_url": "/api/v1/jobs/job_a1b2c3d4e5f6",
  "created_at": "2026-10-21T11:00:00Z",
  "warnings": []
}
```

### 2. Poll Job Status
`GET /api/v1/jobs/{job_id}`

**Response (HTTP 200 OK):**
```json
{
  "job_id": "job_a1b2c3d4e5f6",
  "status": "succeeded",
  "progress": 100,
  "created_at": "2026-10-21T11:00:00Z",
  "updated_at": "2026-10-21T11:00:02Z",
  "completed_at": "2026-10-21T11:00:02Z",
  "warnings": [],
  "assets": {
    "mix_wav": "/api/v1/jobs/job_a1b2c3d4e5f6/download/mix.wav",
    "mix_mp3": "/api/v1/jobs/job_a1b2c3d4e5f6/download/mix.mp3",
    "stems": {
      "drums": "/api/v1/jobs/job_a1b2c3d4e5f6/download/stem_drums.wav",
      "bass": "/api/v1/jobs/job_a1b2c3d4e5f6/download/stem_bass.wav",
      "other": "/api/v1/jobs/job_a1b2c3d4e5f6/download/stem_other.wav"
    },
    "mix_ducked_wav": null,
    "manifest": "/api/v1/jobs/job_a1b2c3d4e5f6/manifest"
  }
}
```

### 3. Retrieve Manifest
`GET /api/v1/jobs/{job_id}/manifest`

Returns the complete `music_manifest.json` containing measured LUFS, True Peak, stem reconstruction SNR, ducking parameters, and license audit.

### 4. Download Audio
`GET /api/v1/jobs/{job_id}/download/{asset_name}`

Streams or downloads `mix.wav`, `mix.mp3`, `stem_drums.wav`, `stem_bass.wav`, `stem_other.wav`, etc.

### 5. Health & Readiness
- `GET /health` — Basic process health (`{"status": "healthy"}`).
- `GET /ready` — Deep health check inspecting database, storage, queue status, music provider, and stem separator.

---

## Structured JSON Error Codes

All API errors return standardized JSON responses with uppercase snake-case codes:

```json
{
  "error": {
    "code": "MISSING_IDEMPOTENCY_KEY",
    "message": "The 'Idempotency-Key' HTTP header is required for creating a music job.",
    "retryable": false,
    "details": {}
  }
}
```

| Error Code | HTTP Status | Description |
| :--- | :--- | :--- |
| `VALIDATION_ERROR` | 400 | Invalid payload (e.g. duration out of 5–300s range, unsupported audio format). |
| `INVALID_SECTION_TIMESTAMPS` | 400 | Section timestamp exceeds or equals requested track duration. |
| `MISSING_IDEMPOTENCY_KEY` | 400 | The required `Idempotency-Key` header was omitted from a POST request. |
| `IDEMPOTENCY_CONFLICT` | 409 | The same `Idempotency-Key` was submitted with a conflicting payload. |
| `SSRF_ATTEMPT_BLOCKED` | 400 | Narration URL targeted localhost, internal IP, or private cloud metadata. |
| `NARRATION_FETCH_FAILED` | 400 | Could not fetch or decode speech narration audio from provided URL or upload. |
| `UNSUPPORTED_FORMAT` | 400 | Audio file format or requested export extension is unsupported. |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication token when `DEMO_ACCESS_TOKEN` is configured. |
| `JOB_NOT_FOUND` | 404 | Specified `job_id` does not exist in the database. |
| `ASSET_NOT_FOUND` | 404 | Requested audio file or manifest has not been generated or was removed. |
| `RATE_LIMIT_EXCEEDED` | 429 | Client exceeded rate limits (default: 60 req/min). |
| `PROVIDER_TIMEOUT` | 504 | Upstream music generation model timed out before completing audio synthesis. |
| `MODEL_PROVIDER_UNAVAILABLE`| 503 | Configured AI generation model or inference library cannot be reached. |
| `MISSING_MODEL_CREDENTIALS` | 503 | `HF_TOKEN` was not provided for gated Hugging Face weights. |
| `STEM_SEPARATION_FAILED` | 500 | Stem separation model failed during audio tensor decomposition. |
| `AUDIO_PROCESSING_ERROR` | 500 | DSP normalization, LUFS filtering, or format transcoding failed. |
| `STORAGE_FAILURE` | 500 | Could not save asset to MinIO S3 or local filesystem storage. |
| `WORKER_FAILURE` | 500 | Background asynchronous worker task failed during pipeline execution. |
| `INTERNAL_SERVER_ERROR` | 500 | An unhandled server exception occurred. |

---

## Model Provider Configuration: Stable Audio 3 Small Music

To enable the real `stabilityai/stable-audio-3-small-music` backend:
1. Log into your Hugging Face account and accept terms at: [`https://huggingface.co/stabilityai/stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music)
2. Generate an Access Token on Hugging Face.
3. Configure your `.env`:
   ```bash
   MUSIC_PROVIDER=stable_audio
   STABLE_AUDIO_MODEL_ID=stabilityai/stable-audio-3-small-music
   HF_TOKEN=hf_your_actual_token_here
   TORCH_DEVICE=cuda
   ```
4. Install PyTorch and the inference package:
   ```bash
   pip install torch torchaudio stable-audio-3
   ```
*If `HF_TOKEN` is missing or model terms are unaccepted, the engine cleanly reports `MISSING_MODEL_CREDENTIALS` or `MODEL_PROVIDER_UNAVAILABLE` rather than failing silently.*

---

## Running the Automated Test Suite

Run the full pytest suite (covering all 18 mandatory assessment scenarios):
```bash
python -m pytest -v
```

Output:
```
tests/test_api.py::test_health_endpoint PASSED                           [  2%]
tests/test_api.py::test_readiness_endpoint PASSED                        [  5%]
tests/test_api.py::test_01_normal_45s_music_request PASSED               [  7%]
...
tests/test_api.py::test_18_loopable_audio PASSED                         [ 72%]
tests/test_audio_processor.py::test_loudness_and_true_peak_normalization PASSED [ 80%]
tests/test_providers.py::test_stable_audio_provider_health_and_credentials_handling PASSED [100%]

============================= 40 passed in 42.07s =============================
```

---

## Command-Line Batch Runner

To execute a batch test pack and download all generated audio and manifests:
```bash
python -m incubrix.batch_runner --input tests/batch_samples.json --output results/
```
The runner creates:
- `{job_id}_mix.wav` for each test request.
- `{job_id}_manifest.json` for each test request.
- `results/batch_summary.json` consolidating execution status, latency, and quality audits.
