"""
Tests for Music Providers and Stem Separators.
"""

import pytest
import numpy as np
from incubrix.schemas import CreateMusicJobRequest, Section
from incubrix.providers.mock_provider import MockMusicProvider
from incubrix.providers.stable_audio import StableAudioProvider
from incubrix.stems.mock_separator import MockStemSeparator
from incubrix.errors import IncubrixError, ErrorCode


@pytest.mark.asyncio
async def test_mock_music_provider_seed_reproducibility():
    provider = MockMusicProvider(sample_rate=48000)
    req1 = CreateMusicJobRequest(
        prompt="Acoustic corporate background music",
        duration_s=10.0,
        seed=12345,
    )
    req2 = CreateMusicJobRequest(
        prompt="Acoustic corporate background music",
        duration_s=10.0,
        seed=12345,
    )
    req3 = CreateMusicJobRequest(
        prompt="Acoustic corporate background music",
        duration_s=10.0,
        seed=99999,
    )

    res1 = await provider.generate_music(req1)
    res2 = await provider.generate_music(req2)
    res3 = await provider.generate_music(req3)

    # Identical seed -> exactly identical audio
    np.testing.assert_array_equal(res1.audio, res2.audio)
    # Different seed -> distinct audio
    assert not np.array_equal(res1.audio, res3.audio)


@pytest.mark.asyncio
async def test_mock_music_provider_sections_and_warnings():
    provider = MockMusicProvider(sample_rate=48000)
    req = CreateMusicJobRequest(
        prompt="Upbeat pop with vocals in the style of Taylor Swift",
        duration_s=20.0,
        sections=[
            Section(start_s=0.0, mood="soft intro"),
            Section(start_s=8.0, mood="energetic build"),
            Section(start_s=15.0, mood="climax finish"),
        ],
        seed=42,
    )

    res = await provider.generate_music(req)
    assert len(res.section_measurements) == 3
    for m in res.section_measurements:
        assert m["aligned_within_0_5s"] is True
        assert m["transition_deviation_s"] <= 0.5

    # Check warnings
    warnings_str = " ".join(res.warnings).lower()
    assert "vocals" in warnings_str
    assert "taylor swift" in warnings_str


@pytest.mark.asyncio
async def test_mock_stem_separator_produces_distinct_stems():
    provider = MockMusicProvider(sample_rate=48000)
    req = CreateMusicJobRequest(
        prompt="Electronic track with synth and drums",
        duration_s=10.0,
        seed=55,
    )
    gen_result = await provider.generate_music(req)

    separator = MockStemSeparator()
    sep_result = await separator.separate_stems(gen_result.audio, 48000, provider_ref=provider)

    assert "drums" in sep_result.stems
    assert "bass" in sep_result.stems
    assert "other" in sep_result.stems

    drums_audio, _ = sep_result.stems["drums"]
    bass_audio, _ = sep_result.stems["bass"]
    other_audio, _ = sep_result.stems["other"]

    # Verify stems are genuinely different signals
    assert not np.array_equal(drums_audio, bass_audio)
    assert not np.array_equal(drums_audio, other_audio)


@pytest.mark.asyncio
async def test_stable_audio_provider_health_and_credentials_handling():
    # Instantiate without token
    provider = StableAudioProvider(
        model_id="stabilityai/stable-audio-3-small-music",
        hf_token=None,
    )

    info = provider.get_provider_info()
    assert info["model_name"] == "stabilityai/stable-audio-3-small-music"
    assert info["is_gated"] is True
    assert "Community License" in info["license"]

    # Health check should indicate missing token
    ok, msg = await provider.health_check()
    assert ok is False
    assert "HF_TOKEN" in msg

    # Calling generate_music without token should raise structured IncubrixError
    req = CreateMusicJobRequest(prompt="Test prompt", duration_s=10.0)
    with pytest.raises(IncubrixError) as exc_info:
        await provider.generate_music(req)

    assert exc_info.value.code == ErrorCode.MISSING_MODEL_CREDENTIALS
    assert exc_info.value.status_code == 503
