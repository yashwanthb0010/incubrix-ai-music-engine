"""
Pydantic schemas and validation models for IncuBrix AI Background Music Engine.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Section(BaseModel):
    start_s: float = Field(..., ge=0.0, description="Start time of the section in seconds")
    mood: str = Field(..., min_length=1, max_length=100, description="Description or mood of the section")


class OutputConfig(BaseModel):
    sample_rate: int = Field(default=48000, description="Audio sample rate (standard 48000 Hz)")
    target_lufs: float = Field(default=-16.0, description="Target integrated loudness in LUFS (standard -16 LUFS)")
    formats: List[str] = Field(default=["wav", "mp3"], description="Target export audio formats (e.g. wav, mp3)")

    @model_validator(mode="after")
    def validate_formats(self) -> "OutputConfig":
        valid = {"wav", "mp3"}
        for fmt in self.formats:
            if fmt.lower() not in valid:
                raise ValueError(f"Unsupported format '{fmt}'. Supported formats are: {sorted(list(valid))}")
        return self


class CreateMusicJobRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1500, description="Text description of desired background music")
    duration_s: float = Field(..., ge=5.0, le=300.0, description="Target track duration in seconds (5 to 300s)")
    sections: Optional[List[Section]] = Field(default=None, description="Optional timestamped mood/energy sections")
    tempo_bpm: Optional[int] = Field(default=None, ge=30, le=260, description="Musical tempo in beats per minute")
    key: Optional[str] = Field(default=None, max_length=50, description="Musical key/scale (e.g. 'C major', 'A minor')")
    loopable: bool = Field(default=False, description="Whether to produce a seamless loopable track")
    narration_url: Optional[str] = Field(default=None, description="Optional audio URL of speech track for ducking")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed for reproducible generation")
    output: Optional[OutputConfig] = Field(default_factory=OutputConfig, description="Output audio specification")

    @model_validator(mode="after")
    def validate_sections_within_duration(self) -> "CreateMusicJobRequest":
        if self.sections:
            # Check for timestamps exceeding duration
            for idx, sec in enumerate(self.sections):
                if sec.start_s >= self.duration_s:
                    raise ValueError(
                        f"Section at index {idx} has start_s={sec.start_s} which exceeds or equals track duration {self.duration_s}s."
                    )
            # Ensure sorted order
            sorted_sections = sorted(self.sections, key=lambda s: s.start_s)
            if [s.start_s for s in self.sections] != [s.start_s for s in sorted_sections]:
                self.sections = sorted_sections
        return self


class StemInfo(BaseModel):
    name: str = Field(..., description="Stem identifier: e.g. drums, bass, other, melodic")
    filename: str = Field(..., description="WAV filename")
    gain: float = Field(default=1.0, description="Linear gain applied during mix reconstruction")
    production_method: str = Field(..., description="Method used: source-separation (Demucs) or synthesized stem decomposition")
    sample_rate: int = Field(default=48000)
    duration_s: float = Field(...)
    download_url: Optional[str] = None


class QualityMeasurements(BaseModel):
    measured_duration_s: float = Field(..., description="Actual duration in seconds")
    duration_error_s: float = Field(..., description="Absolute difference between requested and measured duration")
    sample_rate: int = Field(default=48000)
    integrated_lufs: float = Field(..., description="Measured integrated loudness in LUFS")
    target_lufs: float = Field(default=-16.0)
    lufs_error_lu: float = Field(..., description="Difference between target and measured LUFS")
    true_peak_dbtp: float = Field(..., description="Maximum true peak in dBFS/dBTP")
    has_clipping: bool = Field(default=False)
    has_unexpected_gaps: bool = Field(default=False)
    ending_style: str = Field(default="musical_cadence_or_fade", description="Description of track ending")


class StemReconstruction(BaseModel):
    stem_count: int = Field(..., description="Number of distinct stems")
    stem_identities: List[str] = Field(..., description="List of stem names")
    reconstruction_snr_db: float = Field(..., description="Signal-to-noise / fidelity ratio in dB")
    reconstruction_rms_error: float = Field(..., description="RMS difference between reconstructed mix and final mix")
    difference_max_db: float = Field(..., description="Peak error in dB")
    tolerances_met: bool = Field(default=True)
    limitations: Optional[str] = None


class DuckingManifest(BaseModel):
    applied: bool = Field(default=False)
    narration_source: Optional[str] = None
    attenuation_db: float = Field(default=10.0, description="Ducking reduction in dB")
    speech_detected: bool = Field(default=False)
    speech_duration_s: float = Field(default=0.0)
    attack_ms: float = Field(default=50.0)
    release_ms: float = Field(default=250.0)
    volume_curve_summary: Dict[str, Any] = Field(default_factory=dict)


class MusicManifest(BaseModel):
    manifest_version: str = Field(default="1.0.0")
    job_id: str
    created_at: str
    completed_at: str
    ai_generated_media_label: str = Field(default="AI-Generated Instrumental Background Music (IncuBrix Engine)")
    
    # Request summary
    request: Dict[str, Any]
    
    # Generation Settings
    model_provider: str
    model_name: str
    model_version: str
    seed: Optional[int] = None
    reproducibility: str = Field(default="deterministic_when_seeded")
    
    # Musical features
    requested_duration_s: float
    actual_duration_s: float
    tempo_bpm: Optional[int] = None
    key: Optional[str] = None
    sections: Optional[List[Dict[str, Any]]] = None
    section_transitions_measured: Optional[List[Dict[str, Any]]] = None
    loopable: bool = False
    loop_boundary_checked: Optional[bool] = None
    loop_max_boundary_discontinuity: Optional[float] = None
    
    # Audio outputs and quality
    sample_rate: int
    output_formats: List[str]
    quality_measurements: QualityMeasurements
    
    # Stems & Reconstruction
    stems: List[StemInfo]
    stem_reconstruction: StemReconstruction
    
    # Ducking
    ducking: DuckingManifest
    
    # Operations, Cost, Limitations
    processing_time_s: float
    estimated_cost_usd: float
    warnings: List[str] = Field(default_factory=list)
    known_limitations: List[str] = Field(default_factory=list)
    rights_and_license: Dict[str, Any] = Field(default_factory=dict)
    
    # Value type audit
    audit: Dict[str, str] = Field(
        default_factory=lambda: {
            "duration": "measured",
            "integrated_lufs": "measured",
            "true_peak_dbtp": "measured",
            "reconstruction_snr": "measured",
            "cost": "estimated",
            "tempo_bpm": "provider_or_requested",
        }
    )


class JobAssets(BaseModel):
    mix_wav: Optional[str] = None
    mix_mp3: Optional[str] = None
    stems: Dict[str, str] = Field(default_factory=dict)
    mix_ducked_wav: Optional[str] = None
    manifest: Optional[str] = None


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    status_url: str
    created_at: datetime
    warnings: List[str] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    assets: JobAssets = Field(default_factory=JobAssets)
