from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from YAML then overridden by env vars."""

    model_config = {
        "env_prefix": "SI_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "forbid",
    }

    # Paths
    data_dir: Path = Path("data")
    config_dir: Path = Path("config")

    # Target
    target_name: str = "helinsilver"
    target_base_url: str = "https://helinsilver.com"
    target_languages: list[str] = ["en", "ja"]

    # Source
    source_type: str = "http"
    source_local_path: Optional[Path] = None
    source_git_url: Optional[str] = None
    source_ftp_host: Optional[str] = None
    source_ftp_user: Optional[str] = None
    source_ftp_password: Optional[str] = None
    source_ftp_port: int = 21
    source_ftp_use_tls: bool = True

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: int = 120

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout: int = 120

    # Free Image APIs (optional — improves rate limits for article image search)
    unsplash_api_key: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""

    # Optional AI fallback for article images. Search providers are always tried first.
    openai_api_key: str = ""
    image_generation_enabled: bool = False
    image_generation_model: str = "gpt-image-2"
    article_image_count: int = 4

    # Crawling
    crawl_rate: float = 0.5
    crawl_max_concurrent: int = 3
    crawl_timeout: int = 30
    crawl_max_retries: int = 3
    crawl_max_pages: int = 200
    crawl_user_agent: str = "SiteInspector/1.0 (+https://site-inspector.local)"

    # Lighthouse
    lighthouse_path: str = "lighthouse"
    lighthouse_flags: str = "--chrome-flags='--headless --no-sandbox'"

    # Priority weights
    priority_impact_weight: float = Field(default=0.40, alias="priority_impact_weight")
    priority_severity_weight: float = 0.35
    priority_fix_roi_weight: float = 0.25

    # Auto-fix
    auto_fix_sandbox_diff_auto: float = 0.05
    auto_fix_sandbox_diff_review: float = 0.15
    auto_fix_max_per_scan: int = 50

    # Verification
    verification_observation_days: int = 14
    verification_checkpoints: list[int] = [3, 7, 14]
    verification_degradation_threshold: float = 0.10

    # Git
    git_author_name: str = "Site Inspector Bot"
    git_author_email: str = "bot@site-inspector.local"
    git_platform: str = "github"
    git_default_branch: str = "main"
    git_auto_merge: bool = False

    # Web security. Repository writes stay disabled unless explicitly enabled.
    web_allow_repo_writes: bool = False
    web_max_active_scans: int = 1

    # Google APIs
    google_credentials_path: Optional[Path] = None
    gsc_property: Optional[str] = None
    ga4_property_id: Optional[str] = None

    # Reporting
    report_recipients: list[str] = []
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None

    # Scheduler
    daily_scan_time: str = "02:00"
    weekly_report_day: str = "mon"
    weekly_report_time: str = "08:00"

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Settings":
        path = config_path or Path("config/defaults.yaml")
        yaml_values: dict = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                yaml_values = yaml.safe_load(f) or {}

        # Map the public YAML layout to field names. Keeping this explicit
        # prevents renamed or misspelled options from being silently ignored.
        field_map = {
            ("ollama", "base_url"): "ollama_base_url",
            ("ollama", "model"): "ollama_model",
            ("ollama", "embed_model"): "ollama_embed_model",
            ("ollama", "timeout"): "ollama_timeout",
            ("crawling", "rate"): "crawl_rate",
            ("crawling", "max_concurrent"): "crawl_max_concurrent",
            ("crawling", "timeout_seconds"): "crawl_timeout",
            ("crawling", "max_retries"): "crawl_max_retries",
            ("crawling", "max_pages"): "crawl_max_pages",
            ("crawling", "user_agent"): "crawl_user_agent",
            ("lighthouse", "path"): "lighthouse_path",
            ("lighthouse", "flags"): "lighthouse_flags",
            ("priority_weights", "impact_scope"): "priority_impact_weight",
            ("priority_weights", "severity"): "priority_severity_weight",
            ("priority_weights", "fix_roi"): "priority_fix_roi_weight",
            ("auto_fix", "sandbox_diff_threshold_auto"): "auto_fix_sandbox_diff_auto",
            ("auto_fix", "sandbox_diff_threshold_review"): "auto_fix_sandbox_diff_review",
            ("auto_fix", "max_auto_fixes_per_scan"): "auto_fix_max_per_scan",
            ("verification", "observation_days"): "verification_observation_days",
            ("verification", "checkpoints"): "verification_checkpoints",
            ("verification", "degradation_threshold"): "verification_degradation_threshold",
            ("git", "author_name"): "git_author_name",
            ("git", "author_email"): "git_author_email",
            ("git", "platform"): "git_platform",
            ("git", "auto_merge"): "git_auto_merge",
            ("report", "recipients"): "report_recipients",
            ("report", "smtp_host"): "smtp_host",
            ("report", "smtp_port"): "smtp_port",
            ("scheduler", "daily_scan_time"): "daily_scan_time",
            ("scheduler", "weekly_report_day"): "weekly_report_day",
            ("scheduler", "weekly_report_time"): "weekly_report_time",
            ("web", "allow_repo_writes"): "web_allow_repo_writes",
            ("web", "max_active_scans"): "web_max_active_scans",
        }

        flat: dict = {}
        for section, values in yaml_values.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    key = field_map.get((section, k))
                    if key is None:
                        raise ValueError(f"Unknown configuration option: {section}.{k}")
                    flat[key] = v
            else:
                flat[section] = values

        return cls(**flat)

    @classmethod
    def load_target(cls, target_name: str) -> dict:
        """Load per-target configuration from targets.yaml."""
        path = Path("config/targets.yaml")
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("targets", {}).get(target_name, {})
