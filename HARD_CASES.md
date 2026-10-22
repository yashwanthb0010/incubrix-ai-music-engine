# IncuBrix AI Background Music Engine — Hard Cases & Screening Scenarios

This document details how the IncuBrix AI Background Music Engine handles edge cases and screening scenarios without failing silently. Every hard case is tested in automated CI (`tests/test_hard_cases.py`) and available for 1-click live execution in the **Hard cases** tab of the live Web Studio demo.

---

## Screening Scenarios & Sensible Handling Summary

| # | Hard Case / Screening Scenario | Input Example | System Handling & Defense | Warnings or Error Emitted (No Silent Failures) | Automated Test Verification |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Very short (5–6s) & long (5m / 300s) tracks** | `duration_s: 5.0`<br>`duration_s: 300.0`<br>`duration_s: 2.0` (out of range) | • 5.0s: Exact duration enforced within 0.05s with musical bar-line decay (no hard cut).<br>• 300.0s: Consistent tempo/key, full grid.<br>• Out of bounds (&lt;5s or &gt;300s): Blocked with HTTP 400. | • Success: Validated in manifest (`duration_error_s <= 0.05s`).<br>• Out of bounds: `HTTP 400 VALIDATION_ERROR: Duration must be between 5.0 and 300.0 seconds.` | `test_hard_case_1_very_short_track_5s`<br>`test_hard_case_1_long_track_300s`<br>`test_hard_case_1_out_of_bounds_duration_rejected` |
| **2** | **Vague or contradictory prompts** | `"calm but aggressive"`<br>`"fast slow loud quiet happy sad"` | Semantic parser identifies opposing emotional polarities and synthesizes a contrasting harmonic progression: peaceful ambient intro building into an aggressive rhythmic groove. | `⚠️ Warning: Prompt contains contradictory attributes ('calm' vs 'aggressive'). Resolved by synthesizing dynamic musical contrast: peaceful harmonic atmosphere with driving rhythmic energy.` | `test_hard_case_2_contradictory_prompt` |
| **3** | **Prompts naming real songs, artists, or asking for vocals** | `"Cover of Bohemian Rhapsody by Queen with female singing vocals"` | • Engine suppresses vocals and remains 100% instrumental.<br>• Generates an original algorithmic composition with zero copied motifs to guarantee commercial clearance and prevent copyright infringement. | `⚠️ Warning: Request asked for vocals. This engine is strictly instrumental; vocals were suppressed.`<br>`⚠️ Warning: Prompt referenced 'queen' / 'bohemian rhapsody'. To respect copyright and ensure output is safe to publish commercially, an original composition was generated.` | `test_hard_case_3_real_artist_song_and_vocals` |
| **4** | **Closely spaced sections (&lt;1.5s apart) or section starting after track ends** | `sections: [0.0s, 0.8s, 1.4s]`<br>`duration_s: 10.0, section: 15.0s` | • Closely spaced: Section transitions are snapped and smoothed on musical beat boundaries to prevent jarring audio.<br>• Section &gt; duration: Caught by input schema validator and rejected with HTTP 400. | • Closely spaced: `⚠️ Warning: Section 1 starts 0.80s after previous section (<1.5s). Transition smoothed to musical beat grid to maintain rhythmic cohesion.`<br>• Section &gt; duration: `HTTP 400 VALIDATION_ERROR: Section has start_s=15.0 which exceeds track duration 10.0s.` | `test_hard_case_4_closely_spaced_sections`<br>`test_hard_case_4_section_starting_after_track_ends` |
| **5** | **Narration with long silences or no speech at all** | Room tone silence WAV with zero voice activity | Moving RMS speech envelope detector analyzes voice audio. If 0 speech energy is detected, ducking attenuation is bypassed (0 dB reduction) leaving the master mix unattenuated and pristine. | `⚠️ Warning: No speech activity detected in narration audio; ducking was bypassed and master mix remains unattenuated.`<br>In manifest: `ducking.speech_detected: false`, `ducking.speech_duration_s: 0.0`. | `test_hard_case_5_narration_with_no_speech` |
| **6** | **Prompts in a language other than English** | Spanish: `"Musica alegre corporativo"`<br>German: `"Epische Trailermusik"` | Multilingual keyword parser detects non-English tokens and accents, extracts core semantic attributes, and maps them to the appropriate musical genre and instrumentation. | `⚠️ Warning: Non-English prompt detected. Multilingual semantic parser translated and mapped prompt to musical genre and instrumentation.` | `test_hard_case_6_non_english_prompts` |
| **7** | **Two simultaneous jobs & Idempotency Key duplicate / conflict** | Two simultaneous calls.<br>Duplicate key with same body.<br>Duplicate key with different body. | • Simultaneous: Processed concurrently without database locks or race conditions.<br>• Same key + same body: Returns existing job ID (HTTP 202).<br>• Same key + different body: Rejects with HTTP 409 Conflict. | • Duplicate: Returns original `job_id`.<br>• Conflict: `HTTP 409 IDEMPOTENCY_CONFLICT: An existing job with Idempotency-Key '...' was submitted with a different payload.` | `test_hard_case_7_simultaneous_and_idempotency_conflicts` |
| **8** | **Music provider or model failing or timing out mid-job** | GPU CUDA Out-Of-Memory, API timeout, or upstream crash | Worker pipeline wraps inference in `asyncio.wait_for(timeout=45.0s)`. Catches timeouts and exceptions, transitions job status to `failed`, and logs structured error details. Never hangs indefinitely. | Job status: `failed`, progress: `0%`.<br>Structured error: `{"code": "PROVIDER_TIMEOUT", "message": "Music provider timed out after 45.0s during music waveform synthesis.", "retryable": true}`. | `test_hard_case_8_provider_timeout_handled_gracefully`<br>`test_hard_case_8_provider_runtime_error_handled_gracefully` |

---

## Detailed Case Analyses

### 1. Very Short (5-6s) and Long (5m / 300s) Tracks
- **Short 5-second Sting:** In video editing, intro/outro stings require crisp musicality in short windows. Rather than abruptly slicing the waveform at 5.00s, IncuBrix calculates a proportional musical fade-out and cadence resolution ending on a natural downbeat within $\pm 0.05\text{ s}$ of requested duration.
- **Long 5-minute Bed:** Long ambient tracks avoid drift by maintaining a unified bar and beat grid across all 300 seconds, ensuring that all 3 stems (`drums`, `bass`, `other`) remain sample-aligned throughout.
- **Out of Range (&lt;5s or &gt;300s):** Input validation immediately returns `HTTP 400 VALIDATION_ERROR` with human-readable guidance.

### 2. Vague or Contradictory Prompts
- When prompted with `"calm but aggressive"`, a naive generator either produces cacophony or picks one word arbitrarily.
- IncuBrix resolves contradictory semantics by generating dynamic musical tension: a calm, reverberant chord progression (e.g., piano or pad) that builds into an energetic percussion and bass groove.
- The system returns an explicit notice in both the API response and the manifest `warnings` field.

### 3. Real Artists, Songs, and Vocals
- Commercial video platforms face severe copyright takedown liabilities if an AI engine replicates copyrighted melodies or vocal likenesses.
- When an artist like *Queen*, *Drake*, or a song like *Bohemian Rhapsody* is requested:
  1. The system suppresses all vocal synthesis and enforces 100% instrumental playback.
  2. The system generates an original harmonic composition inspired by the underlying stylistic genre (e.g., hard rock or hip-hop) while copying zero protected melodic sequences.
  3. The system returns an explicit disclaimer: *"Prompt referenced 'queen'. To respect copyright, prevent infringement, and ensure output is safe to publish commercially, an original composition was generated instead of copying an existing song."*

### 4. Closely Spaced Sections & Out-of-Bounds Timestamps
- When section transitions are requested &lt;1.5s apart (e.g., `0.0s`, `0.8s`, `1.4s`), rapid abrupt mood shifts can create audible jarring clicks. IncuBrix snaps transition accent hits to the nearest musical beat interval while issuing an advisory warning.
- When a section timestamp starts after the requested track duration (e.g., `start_s: 15.0` on a 10s track), the request is rejected immediately with `HTTP 400 VALIDATION_ERROR`.

### 5. Narration Ducking with Long Silences or No Speech
- Naive sidechain duckers often "pump" during voiceover pauses or stay ducked if silent audio is passed.
- IncuBrix uses an anti-pumping envelope follower with attack (50 ms), hold (180 ms), and smooth exponential release (300 ms).
- If the narration contains zero speech activity (e.g., room tone), ducking attenuation is bypassed (0 dB reduction) and the master mix remains uncompressed, with an explanatory note in the manifest.

### 6. Non-English Multilingual Prompts
- The engine includes a multilingual semantic keyword lexicon supporting Spanish, German, French, Italian, and Unicode character sets.
- Semantics like *"Musica alegre corporativo"* automatically map to the `corporate_tech` genre profile with cheerful major progressions, and a notification is attached to the manifest.

### 7. Simultaneous Submissions & Idempotency Key Conflicts
- Submitting two jobs at the same instant generates unique UUIDs and independent jobs that execute concurrently in the worker queue.
- Re-submitting the exact same payload with the same `Idempotency-Key` returns HTTP 202 with the existing `job_id`, preventing duplicate GPU/CPU waste.
- Re-submitting a different payload with an already-used `Idempotency-Key` returns `HTTP 409 IDEMPOTENCY_CONFLICT`.

### 8. Provider Failure and Timeout Recovery
- In cloud deployments, upstream models can suffer CUDA OOMs or network timeouts.
- All provider inference calls are wrapped in an `asyncio.wait_for(timeout=45.0s)` boundary.
- If a timeout or error occurs, the job transitions cleanly to `status: failed`, records structured error details with `retryable: true`, and never leaves the client hanging.

---

## Running the Tests

To run the complete automated test suite (all 56 unit, integration, and hard case tests):

```bash
py -m pytest -v
```

To run only the hard cases screening suite:

```bash
py -m pytest -v tests/test_hard_cases.py
```
