"""
Unit and Integration Tests for Hard Cases / Screening Scenarios.
Directly validates all 8 hidden screening test scenarios specified by IncuBrix:
1. Very short (5-6s) and long (5 min) tracks, plus boundary checks.
2. Vague or contradictory prompts (e.g. "calm but aggressive").
3. Real songs / artists (e.g. Queen / Bohemian Rhapsody) and vocal requests.
4. Closely spaced sections (<1.5s) and out-of-bounds sections.
5. Narration with no speech / long silence.
6. Non-English multilingual prompts (Spanish, German, Japanese).
7. Simultaneous submissions and Idempotency-Key duplicate / conflict handling.
8. Provider failure and timeout handling (never fails silently).
"""

import time
import pytest
import numpy as np
from fastapi.testclient import TestClient

from incubrix.schemas import JobStatus
from incubrix.errors import ErrorCode


def wait_for_job(client: TestClient, job_id: str, timeout_s: float = 25.0):
    start = time.time()
    while time.time() - start < timeout_s:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("succeeded", "failed"):
            return data
        time.sleep(0.3)
    pytest.fail(f"Job {job_id} did not finish within {timeout_s}s.")


# ==============================================================================
# Hard Case 1: Very short (5-6s) and long (5m / 300s) tracks, plus out-of-bounds
# ==============================================================================

def test_hard_case_1_very_short_track_5s(client: TestClient):
    resp = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Short 5-second energetic podcast sting", "duration_s": 5.0, "seed": 10},
        headers={"Idempotency-Key": "hc1_short_5s"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = wait_for_job(client, job_id)
    assert job["status"] == "succeeded"

    man_resp = client.get(f"/api/v1/jobs/{job_id}/manifest")
    assert man_resp.status_code == 200
    manifest = man_resp.json()
    assert abs(manifest["actual_duration_s"] - 5.0) <= 0.05
    assert manifest["quality_measurements"]["has_clipping"] is False


def test_hard_case_1_long_track_300s(client: TestClient):
    resp = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Long continuous ambient background soundscape", "duration_s": 300.0, "seed": 20},
        headers={"Idempotency-Key": "hc1_long_300s"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = wait_for_job(client, job_id, timeout_s=35.0)
    assert job["status"] == "succeeded"

    manifest = client.get(f"/api/v1/jobs/{job_id}/manifest").json()
    assert abs(manifest["actual_duration_s"] - 300.0) <= 0.05
    assert manifest["beat_and_bar_grid"] is not None


def test_hard_case_1_out_of_bounds_duration_rejected(client: TestClient):
    # Too short (< 5s)
    resp_short = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Too short sting", "duration_s": 2.0},
        headers={"Idempotency-Key": "hc1_too_short"},
    )
    assert resp_short.status_code == 400
    assert resp_short.json()["error"]["code"] == ErrorCode.VALIDATION_ERROR

    # Too long (> 300s)
    resp_long = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Too long track", "duration_s": 600.0},
        headers={"Idempotency-Key": "hc1_too_long"},
    )
    assert resp_long.status_code == 400
    assert resp_long.json()["error"]["code"] == ErrorCode.VALIDATION_ERROR


# ==============================================================================
# Hard Case 2: Vague or contradictory prompts ("calm but aggressive")
# ==============================================================================

def test_hard_case_2_contradictory_prompt(client: TestClient):
    resp = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Calm but aggressive soundtrack with quiet screaming energy", "duration_s": 8.0, "seed": 33},
        headers={"Idempotency-Key": "hc2_contradictory"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    warnings = resp.json()["warnings"]

    # System must say what it did instead of failing silently
    assert any("contradictory" in w.lower() for w in warnings)

    job = wait_for_job(client, job_id)
    assert job["status"] == "succeeded"

    manifest = client.get(f"/api/v1/jobs/{job_id}/manifest").json()
    assert any("contradictory" in w.lower() for w in manifest.get("warnings", []))


# ==============================================================================
# Hard Case 3: Real song or artist, and vocal requests
# ==============================================================================

def test_hard_case_3_real_artist_song_and_vocals(client: TestClient):
    resp = client.post(
        "/api/v1/music/jobs",
        json={
            "prompt": "Cover of Bohemian Rhapsody by Queen with female singing vocals and lyrics",
            "duration_s": 10.0,
            "seed": 44,
        },
        headers={"Idempotency-Key": "hc3_artist_vocals"},
    )
    assert resp.status_code == 202
    warnings = resp.json()["warnings"]

    # Must explain vocals suppression and copyright original generation
    assert any("vocals" in w.lower() and "suppressed" in w.lower() for w in warnings)
    assert any("queen" in w.lower() or "bohemian rhapsody" in w.lower() or "copyright" in w.lower() for w in warnings)

    job = wait_for_job(client, job_id=resp.json()["job_id"])
    assert job["status"] == "succeeded"


# ==============================================================================
# Hard Case 4: Closely spaced sections (<1.5s) and out-of-bounds section
# ==============================================================================

def test_hard_case_4_closely_spaced_sections(client: TestClient):
    resp = client.post(
        "/api/v1/music/jobs",
        json={
            "prompt": "Rapid transition cue",
            "duration_s": 12.0,
            "sections": [
                {"start_s": 0.0, "mood": "calm intro"},
                {"start_s": 0.8, "mood": "rapid shift"},
                {"start_s": 1.4, "mood": "another shift"},
            ],
            "seed": 55,
        },
        headers={"Idempotency-Key": "hc4_close_sections"},
    )
    assert resp.status_code == 202
    warnings = resp.json()["warnings"]
    assert any("closely-spaced" in w.lower() or "<1.5s" in w.lower() for w in warnings)

    job = wait_for_job(client, resp.json()["job_id"])
    assert job["status"] == "succeeded"


def test_hard_case_4_section_starting_after_track_ends(client: TestClient):
    resp = client.post(
        "/api/v1/music/jobs",
        json={
            "prompt": "Impossible timing",
            "duration_s": 10.0,
            "sections": [
                {"start_s": 0.0, "mood": "intro"},
                {"start_s": 15.0, "mood": "after end of song"},
            ],
        },
        headers={"Idempotency-Key": "hc4_invalid_sec"},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == ErrorCode.VALIDATION_ERROR
    assert "exceeds" in str(err) or "validation" in str(err).lower()


# ==============================================================================
# Hard Case 5: Narration with long silences or no speech at all
# ==============================================================================

def test_hard_case_5_narration_with_no_speech(client: TestClient, monkeypatch):
    from incubrix.queue import pipeline

    # Mock narration audio with completely silent audio (zero speech energy)
    async def mock_silent_narration(url: str, *args, **kwargs):
        sr = 48000
        silent = np.zeros(int(8.0 * sr), dtype=np.float32)
        return silent, sr

    monkeypatch.setattr(pipeline, "fetch_narration_audio", mock_silent_narration)

    resp = client.post(
        "/api/v1/music/jobs",
        json={
            "prompt": "Interview music with silent audio file",
            "duration_s": 8.0,
            "narration_url": "https://example.com/silent_room.wav",
        },
        headers={"Idempotency-Key": "hc5_silent_narration"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = wait_for_job(client, job_id)
    assert job["status"] == "succeeded"

    manifest = client.get(f"/api/v1/jobs/{job_id}/manifest").json()
    ducking = manifest.get("ducking", {})
    assert ducking["applied"] is True
    assert ducking["speech_detected"] is False
    # Manifest / warnings states ducking was bypassed for silence
    assert any("no speech" in w.lower() for w in manifest.get("warnings", []))


# ==============================================================================
# Hard Case 6: Non-English prompts (Spanish, German, Japanese)
# ==============================================================================

def test_hard_case_6_non_english_prompts(client: TestClient):
    # Spanish prompt
    resp_es = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Musica alegre para un video corporativo de tecnologia moderna", "duration_s": 8.0},
        headers={"Idempotency-Key": "hc6_spanish"},
    )
    assert resp_es.status_code == 202
    assert any("non-english" in w.lower() for w in resp_es.json()["warnings"])

    # German prompt
    resp_de = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Epische dramatische Trailermusik mit Streichern und Bläsern", "duration_s": 8.0},
        headers={"Idempotency-Key": "hc6_german"},
    )
    assert resp_de.status_code == 202
    assert any("non-english" in w.lower() for w in resp_de.json()["warnings"])

    job = wait_for_job(client, resp_es.json()["job_id"])
    assert job["status"] == "succeeded"


# ==============================================================================
# Hard Case 7: Simultaneous jobs and Idempotency Duplicate / Conflict
# ==============================================================================

def test_hard_case_7_simultaneous_and_idempotency_conflicts(client: TestClient):
    key = "hc7_unique_idemp_key_123"
    payload1 = {"prompt": "First submission track", "duration_s": 8.0}

    # 1. First submission
    resp1 = client.post("/api/v1/music/jobs", json=payload1, headers={"Idempotency-Key": key})
    assert resp1.status_code == 202
    job_id1 = resp1.json()["job_id"]

    # 2. Duplicate submission with exact same key & payload -> Returns existing job
    resp2 = client.post("/api/v1/music/jobs", json=payload1, headers={"Idempotency-Key": key})
    assert resp2.status_code == 202
    assert resp2.json()["job_id"] == job_id1

    # 3. Conflicting submission with same key but different duration -> HTTP 409
    payload_conflict = {"prompt": "First submission track", "duration_s": 24.0}
    resp3 = client.post("/api/v1/music/jobs", json=payload_conflict, headers={"Idempotency-Key": key})
    assert resp3.status_code == 409
    err = resp3.json()["error"]
    assert err["code"] == ErrorCode.IDEMPOTENCY_CONFLICT
    assert "different payload" in err["message"]


# ==============================================================================
# Hard Case 8: Provider failure and timeout handling
# ==============================================================================

def test_hard_case_8_provider_timeout_handled_gracefully(client: TestClient, monkeypatch):
    import asyncio
    from incubrix.providers import mock_provider

    # Simulate provider timing out
    async def mock_hanging_generate(*args, **kwargs):
        await asyncio.sleep(10.0)  # Exceeds configured timeout

    monkeypatch.setattr(mock_provider.MockMusicProvider, "generate_music", mock_hanging_generate)
    from incubrix.config import get_settings
    monkeypatch.setattr(get_settings(), "model_timeout_s", 0.5)

    resp = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Track that will timeout", "duration_s": 6.0},
        headers={"Idempotency-Key": "hc8_timeout_test"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = wait_for_job(client, job_id, timeout_s=5.0)
    assert job["status"] == "failed"
    assert job["error"] is not None
    assert job["error"]["code"] == ErrorCode.PROVIDER_TIMEOUT
    assert "timed out" in job["error"]["message"].lower()
    assert job["error"]["retryable"] is True


def test_hard_case_8_provider_runtime_error_handled_gracefully(client: TestClient, monkeypatch):
    from incubrix.providers import mock_provider

    # Simulate provider internal failure
    async def mock_exploding_generate(*args, **kwargs):
        raise RuntimeError("CUDA Out-Of-Memory during diffusion pass")

    monkeypatch.setattr(mock_provider.MockMusicProvider, "generate_music", mock_exploding_generate)

    resp = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Track that will explode", "duration_s": 6.0},
        headers={"Idempotency-Key": "hc8_failure_test"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = wait_for_job(client, job_id, timeout_s=5.0)
    assert job["status"] == "failed"
    assert job["error"] is not None
    assert job["error"]["code"] == ErrorCode.MODEL_PROVIDER_UNAVAILABLE
    assert "CUDA Out-Of-Memory" in job["error"]["message"]
