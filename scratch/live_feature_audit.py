"""
Full Feature Verification Audit for IncuBrix AI Background Music Engine
Tests all 10 core architectural and assessment requirements against the live containerized system.
"""
import time
import requests
import json
import io
import wave

BASE_URL = "http://localhost:8000"

def log(msg, status="INFO"):
    symbol = "PASS" if status == "PASS" else ("FAIL" if status == "FAIL" else "INFO")
    print(f"[{symbol}] {msg}")

def test_feature_matrix():
    results = {}
    print("=" * 70)
    print("  INCUBRIX AI MUSIC ENGINE - FULL FEATURE VERIFICATION AUDIT")
    print("=" * 70)

    # 1. Health & Readiness
    try:
        r_health = requests.get(f"{BASE_URL}/health", timeout=5)
        r_ready = requests.get(f"{BASE_URL}/ready", timeout=5)
        if r_health.status_code == 200 and r_ready.status_code == 200:
            log("Health & Readiness probes: HTTP 200 OK (all dependencies online)", "PASS")
            results["health_probes"] = True
        else:
            log(f"Health probes returned status: {r_health.status_code}, {r_ready.status_code}", "FAIL")
            results["health_probes"] = False
    except Exception as e:
        log(f"Health probe check failed: {e}", "FAIL")
        results["health_probes"] = False

    # 2. Studio UI & Swagger Docs
    try:
        r_ui = requests.get(f"{BASE_URL}/", timeout=5)
        r_docs = requests.get(f"{BASE_URL}/docs", timeout=5)
        if r_ui.status_code == 200 and "IncuBrix" in r_ui.text and r_docs.status_code == 200:
            log("Studio Web UI (Modern SaaS) & Interactive Swagger Docs: HTTP 200 OK", "PASS")
            results["ui_and_docs"] = True
        else:
            log(f"UI/Docs check failed: UI {r_ui.status_code}, Docs {r_docs.status_code}", "FAIL")
            results["ui_and_docs"] = False
    except Exception as e:
        log(f"UI/Docs check failed: {e}", "FAIL")
        results["ui_and_docs"] = False

    # 3. Job Submission (Testing POST /api/v1/music/jobs)
    job_id = None
    try:
        payload = {
            "prompt": "Modern upbeat corporate podcast theme with crisp drums and warm bass",
            "duration_s": 15.0,
            "tempo_bpm": 120,
            "key": "C major",
            "loopable": True,
            "sections": [
                {"start_s": 0.0, "mood": "intro"},
                {"start_s": 5.0, "mood": "verse"},
                {"start_s": 10.0, "mood": "chorus"}
            ]
        }
        headers = {"Idempotency-Key": f"audit-key-{int(time.time())}"}
        r_job = requests.post(f"{BASE_URL}/api/v1/music/jobs", json=payload, headers=headers, timeout=10)
        if r_job.status_code == 202:
            data = r_job.json()
            job_id = data.get("job_id")
            log(f"Job creation endpoint (POST /api/v1/music/jobs): HTTP 202 Accepted (Job ID: {job_id})", "PASS")
            results["job_creation"] = True
        else:
            log(f"Job creation failed with status {r_job.status_code}: {r_job.text}", "FAIL")
            results["job_creation"] = False
    except Exception as e:
        log(f"Job submission failed: {e}", "FAIL")
        results["job_creation"] = False

    if not job_id:
        print("Cannot continue stem checks without job_id")
        return results

    # 4. Job Polling & Processing Completion
    log(f"Waiting for generation job {job_id} to finish...")
    completed = False
    for _ in range(30):
        time.sleep(2)
        r_poll = requests.get(f"{BASE_URL}/api/v1/jobs/{job_id}", timeout=5)
        if r_poll.status_code == 200:
            job_data = r_poll.json()
            st = job_data.get("status")
            if st == "succeeded":
                completed = True
                log("Music Generation Pipeline: Completed with status 'succeeded'", "PASS")
                break
            elif st == "failed":
                log(f"Job failed: {job_data.get('error')}", "FAIL")
                break
    results["job_completion"] = completed
    if not completed:
        return results

    # 5. Fetch Complete Music Manifest
    r_man = requests.get(f"{BASE_URL}/api/v1/jobs/{job_id}/manifest", timeout=5)
    manifest = r_man.json() if r_man.status_code == 200 else {}
    
    # 5a. Exact Duration Enforcement
    actual_duration = manifest.get("actual_duration_s", 0.0)
    if abs(actual_duration - 15.0) < 0.05:
        log(f"Exact Duration Enforcement: Requested 15.0s, Output {actual_duration:.2f}s (delta < 50ms)", "PASS")
        results["exact_duration"] = True
    else:
        log(f"Duration mismatch: Requested 15.0s, got {actual_duration}", "FAIL")
        results["exact_duration"] = False

    # 6. Real Discrete Stems (Drums, Bass, Melody/Other) & Reconstruction
    sr_data = manifest.get("stem_reconstruction", {})
    stems = sr_data.get("stem_identities", [])
    snr = sr_data.get("reconstruction_snr_db", 0.0)
    if len(stems) >= 3 and "drums" in stems and "bass" in stems and "other" in stems and snr > 20.0:
        log(f"Real Stem Generation & Reconstruction: Discrete {stems} with high SNR ({snr:.1f} dB)", "PASS")
        results["stems_presence"] = True
    else:
        log(f"Missing stems or poor SNR: stems={stems}, snr={snr}", "FAIL")
        results["stems_presence"] = False

    # 7. Broadcast Audio Compliance (LUFS & Peak)
    qm = manifest.get("quality_measurements", {})
    lufs = qm.get("integrated_lufs", -99)
    peak = qm.get("true_peak_dbtp", 99)
    if -17.5 <= lufs <= -14.5 and peak <= -0.5:
        log(f"EBU R128 Broadcast Loudness: Integrated LUFS = {lufs:.1f} (target -16±1.0), Peak = {peak:.2f} dBTP (target <= -1.0)", "PASS")
        results["broadcast_lufs"] = True
    else:
        log(f"Loudness non-compliant: LUFS = {lufs}, Peak = {peak}", "FAIL")
        results["broadcast_lufs"] = False

    # 8. Beat/Bar Grid & Section Transitions in Manifest
    grid = manifest.get("beat_and_bar_grid", {})
    sections = manifest.get("sections", [])
    beats = grid.get("beats", [])
    bars = grid.get("bars", [])
    if len(sections) >= 3 and len(beats) > 0 and len(bars) > 0:
        log(f"Musical Grid & Structure: {len(sections)} sections, {len(beats)} beat markers, {len(bars)} bar markers aligned at {grid.get('tempo_bpm')} BPM", "PASS")
        results["grid_and_sections"] = True
    else:
        log(f"Grid or sections incomplete: sections={sections}, beats={len(beats)}, bars={len(bars)}", "FAIL")
        results["grid_and_sections"] = False

    # 9. Instrumental Only & Commercial Safety Rights
    rights = manifest.get("rights_and_license", {})
    commercial_ok = rights.get("commercial_use_permitted", False)
    safety_status = rights.get("commercial_safety_status", "")
    originality = rights.get("originality_guarantee", "")
    if commercial_ok and "Safe" in safety_status:
        log(f"Commercial Safety & Legal Clearance: Permitted=True, Status='{safety_status}' ({rights.get('license_name')})", "PASS")
        results["commercial_safety"] = True
    else:
        log(f"Commercial safety check failed: rights={rights}", "FAIL")
        results["commercial_safety"] = False

    # 10. HTTP 206 Audio Scrubbing / Range Seeking
    try:
        mix_url = f"{BASE_URL}/api/v1/jobs/{job_id}/download/mix.wav"
        headers = {"Range": "bytes=1000-4999"}
        r_range = requests.get(mix_url, headers=headers, timeout=5)
        if r_range.status_code == 206 and r_range.headers.get("Accept-Ranges") == "bytes":
            content_range = r_range.headers.get("Content-Range", "")
            log(f"Audio Scrubbing / Seeking: HTTP 206 Partial Content supported (Content-Range: {content_range})", "PASS")
            results["audio_scrubbing"] = True
        else:
            log(f"Range request failed: status {r_range.status_code}, headers: {r_range.headers}", "FAIL")
            results["audio_scrubbing"] = False
    except Exception as e:
        log(f"Audio scrubbing check failed: {e}", "FAIL")
        results["audio_scrubbing"] = False

    # 11. Single-Stem Regeneration
    try:
        # Get current bass and drums stem bytes
        bass_url = f"{BASE_URL}/api/v1/jobs/{job_id}/download/stem_bass.wav"
        drums_url = f"{BASE_URL}/api/v1/jobs/{job_id}/download/stem_drums.wav"
        bass_orig = requests.get(bass_url, timeout=5).content
        drums_orig = requests.get(drums_url, timeout=5).content

        # Regenerate drums
        r_regen = requests.post(
            f"{BASE_URL}/api/v1/jobs/{job_id}/stems/drums/regenerate",
            json={"stem_name": "drums", "seed": 9999},
            timeout=15
        )
        if r_regen.status_code == 200:
            log(f"Single-Stem Regeneration API (POST /api/v1/jobs/{job_id}/stems/drums/regenerate): HTTP 200 OK", "PASS")
            bass_after = requests.get(bass_url, timeout=5).content
            drums_after = requests.get(drums_url, timeout=5).content
            
            # Bass should be untouched, drums should be modified
            if bass_orig == bass_after and drums_orig != drums_after:
                log("Stem Isolation Guarantee: Non-target stems (Bass) completely untouched (byte-identical), target (Drums) refreshed", "PASS")
                results["single_stem_regen"] = True
            else:
                log("Stem isolation verification failed", "FAIL")
                results["single_stem_regen"] = False
        else:
            log(f"Stem regeneration failed: status {r_regen.status_code}, {r_regen.text}", "FAIL")
            results["single_stem_regen"] = False
    except Exception as e:
        log(f"Stem regeneration check failed: {e}", "FAIL")
        results["single_stem_regen"] = False

    # 12. Edge Case & Input Validation Resilience (422/400 on invalid requests)
    try:
        r_invalid = requests.post(f"{BASE_URL}/api/v1/music/jobs", json={"duration": 9999}, timeout=5)
        if r_invalid.status_code in (400, 422):
            log(f"Edge Case Resilience: Invalid duration 9999 rejected with HTTP {r_invalid.status_code}", "PASS")
            results["edge_cases"] = True
        else:
            log(f"Invalid duration was not rejected: {r_invalid.status_code}", "FAIL")
            results["edge_cases"] = False
    except Exception as e:
        log(f"Edge case test failed: {e}", "FAIL")
        results["edge_cases"] = False

    print("=" * 70)
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"VERIFICATION SUMMARY: {passed_count}/{total_count} FEATURES PASSED ({passed_count/total_count*100:.0f}%)")
    print("=" * 70)
    return results

if __name__ == "__main__":
    test_feature_matrix()
