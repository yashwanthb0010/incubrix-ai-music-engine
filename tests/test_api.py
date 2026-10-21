"""
Comprehensive API and Integration Tests for IncuBrix AI Background Music Engine.
Covers all 18 mandated assessment test scenarios.
"""

import time
import pytest
from fastapi.testclient import TestClient


def wait_for_job_completion(client: TestClient, job_id: str, timeout_s: float = 20.0):
    start = time.time()
    while time.time() - start < timeout_s:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("succeeded", "failed"):
            return data
        time.sleep(0.3)
    pytest.fail(f"Job {job_id} did not finish within {timeout_s} seconds.")


# --- Health & Readiness ---

def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_readiness_endpoint(client: TestClient):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True
    checks = resp.json()["checks"]
    assert "database" in checks
    assert "queue" in checks
    assert "music_provider" in checks


# --- 1. Normal 45-second music request ---

def test_01_normal_45s_music_request(client: TestClient):
    payload = {
      "prompt": "Warm, optimistic acoustic pop for a product launch",
      "duration_s": 45.0,
      "tempo_bpm": 110,
      "key": "C major",
      "seed": 42
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_normal_45s"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "succeeded"
    assert job["progress"] == 100
    assert job["assets"]["mix_wav"] is not None
    assert "drums" in job["assets"]["stems"]
    assert "bass" in job["assets"]["stems"]
    assert "other" in job["assets"]["stems"]

    # Download mix
    mix_resp = client.get(f"/api/v1/jobs/{job_id}/download/mix.wav")
    assert mix_resp.status_code == 200
    assert len(mix_resp.content) > 1000

    # Manifest check
    man_resp = client.get(f"/api/v1/jobs/{job_id}/manifest")
    assert man_resp.status_code == 200
    manifest = man_resp.json()
    assert abs(manifest["actual_duration_s"] - 45.0) <= 0.05
    assert abs(manifest["quality_measurements"]["integrated_lufs"] - (-16.0)) <= 1.0


# --- 2. Short 5–6-second request ---

def test_02_short_5s_request(client: TestClient):
    payload = {
        "prompt": "Punchy 5-second energetic podcast sting",
        "duration_s": 5.0,
        "seed": 101,
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_short_5s"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "succeeded"

    man_resp = client.get(f"/api/v1/jobs/{job_id}/manifest")
    manifest = man_resp.json()
    assert abs(manifest["actual_duration_s"] - 5.0) <= 0.05


# --- 3. Long 300-second request ---

def test_03_300s_request(client: TestClient):
    payload = {
        "prompt": "Continuous ambient study music",
        "duration_s": 300.0,
        "seed": 202,
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_long_300s"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = wait_for_job_completion(client, job_id, timeout_s=30.0)
    assert job["status"] == "succeeded"

    man_resp = client.get(f"/api/v1/jobs/{job_id}/manifest")
    manifest = man_resp.json()
    assert abs(manifest["actual_duration_s"] - 300.0) <= 0.05


# --- 4. Ten distinct style prompts ---

@pytest.mark.parametrize(
    "style,prompt",
    [
        ("acoustic", "Acoustic folk song with fingerpicked guitar"),
        ("cinematic", "Dramatic cinematic trailer soundtrack with brass"),
        ("corporate", "Modern clean tech corporate presentation theme"),
        ("lofi", "Chill lo-fi study beat with mellow electric piano"),
        ("electronic", "Uplifting future bass synth electronic track"),
        ("tense", "Dark suspenseful mystery thriller background pulse"),
        ("playful", "Quirky pizzicato comedy theme with light marimba"),
        ("ambient", "Dreamy ambient space music with lush reverb pads"),
        ("rock", "Instrumental energetic indie rock guitar groove"),
        ("jazz", "Sophisticated smooth jazz lounge quartet groove"),
    ],
)
def test_04_ten_distinct_styles(client: TestClient, style: str, prompt: str):
    payload = {"prompt": prompt, "duration_s": 10.0, "seed": 42}
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": f"test_style_{style}"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "succeeded"


# --- 5. Contradictory or vague prompt ---

def test_05_contradictory_prompt(client: TestClient):
    payload = {
        "prompt": "Fast slow loud quiet happy sad purple square sound",
        "duration_s": 10.0,
        "seed": 99,
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_vague_prompt"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "succeeded"


# --- 6. Request naming real song or artist ---

def test_06_real_artist_warning(client: TestClient):
    payload = {
        "prompt": "Rock track in the style of Queen and Michael Jackson",
        "duration_s": 12.0,
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_artist_warning"},
    )
    assert resp.status_code == 202
    warnings = resp.json()["warnings"]
    assert any("Queen" in w or "copyright" in w.lower() for w in warnings)


# --- 7. Request asking for vocals ---

def test_07_vocal_request_warning(client: TestClient):
    payload = {
        "prompt": "Catchy pop song with female vocals and singing melody",
        "duration_s": 12.0,
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_vocal_request"},
    )
    assert resp.status_code == 202
    warnings = resp.json()["warnings"]
    assert any("vocals" in w.lower() for w in warnings)


# --- 8. Closely spaced sections ---

def test_08_closely_spaced_sections(client: TestClient):
    payload = {
        "prompt": "Rapid fire video intro",
        "duration_s": 15.0,
        "sections": [
            {"start_s": 0.0, "mood": "start"},
            {"start_s": 1.0, "mood": "fast switch"},
            {"start_s": 2.2, "mood": "another switch"},
        ],
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_close_sections"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "succeeded"


# --- 9. Section starting after track ends ---

def test_09_section_starting_after_track_ends(client: TestClient):
    payload = {
        "prompt": "Invalid track",
        "duration_s": 10.0,
        "sections": [
            {"start_s": 0.0, "mood": "intro"},
            {"start_s": 15.0, "mood": "invalid end"},
        ],
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_invalid_section"},
    )
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "VALIDATION_ERROR"


# --- 10 & 11. Narration handling ---

def test_10_missing_idempotency_key(client: TestClient):
    payload = {"prompt": "Music test", "duration_s": 10.0}
    # No Idempotency-Key header
    resp = client.post("/api/v1/music/jobs", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "MISSING_IDEMPOTENCY_KEY"


def test_11_ssrf_attempt_blocked(client: TestClient):
    payload = {
        "prompt": "Music with exploit",
        "duration_s": 10.0,
        "narration_url": "http://127.0.0.1:8000/internal-data.wav",
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_ssrf_key"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "SSRF_ATTEMPT_BLOCKED"


# --- 12. Non-English prompt ---

def test_12_non_english_prompt(client: TestClient):
    payload = {
        "prompt": "Musica alegre para un video corporativo de tecnologia moderna",
        "duration_s": 12.0,
        "seed": 888,
    }
    resp = client.post(
        "/api/v1/music/jobs",
        json=payload,
        headers={"Idempotency-Key": "test_non_english"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = wait_for_job_completion(client, job_id)
    assert job["status"] == "succeeded"


# --- 13. Two simultaneous jobs ---

def test_13_two_simultaneous_jobs(client: TestClient):
    resp1 = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "First simultaneous track", "duration_s": 8.0},
        headers={"Idempotency-Key": "test_simultaneous_1"},
    )
    resp2 = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Second simultaneous track", "duration_s": 8.0},
        headers={"Idempotency-Key": "test_simultaneous_2"},
    )
    assert resp1.status_code == 202
    assert resp2.status_code == 202

    id1 = resp1.json()["job_id"]
    id2 = resp2.json()["job_id"]

    job1 = wait_for_job_completion(client, id1)
    job2 = wait_for_job_completion(client, id2)
    assert job1["status"] == "succeeded"
    assert job2["status"] == "succeeded"


# --- 14. Idempotency Key Handling ---

def test_14_idempotency_duplicate_and_conflict(client: TestClient):
    key = "idemp_test_key_abc"
    payload = {"prompt": "Idempotent music request", "duration_s": 10.0}

    # First submission
    resp1 = client.post("/api/v1/music/jobs", json=payload, headers={"Idempotency-Key": key})
    assert resp1.status_code == 202
    job_id1 = resp1.json()["job_id"]

    # Identical submission with same key -> Returns existing job
    resp2 = client.post("/api/v1/music/jobs", json=payload, headers={"Idempotency-Key": key})
    assert resp2.status_code == 202
    assert resp2.json()["job_id"] == job_id1

    # Conflicting submission with same key but different payload -> HTTP 409 Conflict
    conflicting_payload = {"prompt": "Completely different music request", "duration_s": 25.0}
    resp3 = client.post("/api/v1/music/jobs", json=conflicting_payload, headers={"Idempotency-Key": key})
    assert resp3.status_code == 409
    assert resp3.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


# --- 15. Nonexistent job status & asset 404 ---

def test_15_nonexistent_job_and_asset(client: TestClient):
    resp = client.get("/api/v1/jobs/job_nonexistent_123")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "JOB_NOT_FOUND"

    resp_asset = client.get("/api/v1/jobs/job_nonexistent_123/download/mix.wav")
    assert resp_asset.status_code == 404


# --- 16. Audio length and loudness validation ---

def test_16_audio_length_and_loudness_verification(client: TestClient):
    req_dur = 14.5
    resp = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Precision loudness and duration test", "duration_s": req_dur},
        headers={"Idempotency-Key": "test_precision_loudness"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    wait_for_job_completion(client, job_id)

    man_resp = client.get(f"/api/v1/jobs/{job_id}/manifest")
    manifest = man_resp.json()
    qm = manifest["quality_measurements"]

    # Duration within 0.05s
    assert qm["duration_error_s"] <= 0.05
    # Integrated loudness within 1 LU of -16 LUFS
    assert qm["lufs_error_lu"] <= 1.0
    # True peak <= -1.0 dBTP
    assert qm["true_peak_dbtp"] <= -1.0
    assert qm["has_clipping"] is False


# --- 17. Stem reconstruction and distinct stems verification ---

def test_17_stem_reconstruction_verification(client: TestClient):
    resp = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Stem reconstruction test", "duration_s": 15.0},
        headers={"Idempotency-Key": "test_stem_reconstruction"},
    )
    job_id = resp.json()["job_id"]
    wait_for_job_completion(client, job_id)

    man_resp = client.get(f"/api/v1/jobs/{job_id}/manifest")
    manifest = man_resp.json()
    sr = manifest["stem_reconstruction"]

    # Must have at least 3 genuine stems
    assert sr["stem_count"] >= 3
    assert "drums" in sr["stem_identities"]
    assert "bass" in sr["stem_identities"]
    assert "other" in sr["stem_identities"]
    # Reconstructed mix fidelity SNR should be high
    assert sr["reconstruction_snr_db"] > 25.0
    assert sr["tolerances_met"] is True


# --- 18. Loopable audio generation ---

def test_18_loopable_audio(client: TestClient):
    resp = client.post(
        "/api/v1/music/jobs",
        json={"prompt": "Seamless background loop", "duration_s": 12.0, "loopable": True},
        headers={"Idempotency-Key": "test_loopable"},
    )
    job_id = resp.json()["job_id"]
    wait_for_job_completion(client, job_id)

    man_resp = client.get(f"/api/v1/jobs/{job_id}/manifest")
    manifest = man_resp.json()
    assert manifest["loopable"] is True
    assert manifest["loop_boundary_checked"] is True
    assert manifest["loop_max_boundary_discontinuity"] is not None
    assert manifest["loop_max_boundary_discontinuity"] < 0.01
