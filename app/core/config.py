from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./image_factory.db"
    redis_url: str = "redis://localhost:6379/0"
    storage_root: str = "./data"
    image_analysis_provider: str = "mock"
    qa_provider: str = "mock"
    image_generation_provider: str = "mock"
    generation_max_jobs_per_run: int = 5
    visual_strategy: str = "safe_material_hero"
    pipeline_log_dir: str = "data/logs"
    qa_min_total_score: int = 80
    qa_min_risk_score: int = 16
    qa_min_product_accuracy_score: int = 16
    qa_min_material_realism_score: int = 16
    qa_min_photorealism_score: int = 16
    qa_min_structure_preservation_score: int = 16
    qa_policy_version: str = "qa_policy_v2_safe_material"
    stage_run_lease_seconds: int = 900
    stage_max_inflight_default: int = 8
    stage_max_inflight_analysis: int = 16
    stage_max_inflight_generation: int = 4
    stage_max_inflight_qa: int = 16
    stage_max_inflight_publish: int = 4
    openai_max_retries: int = 3
    openai_request_timeout_seconds: int = 180
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_image_model: str = "gpt-image-2"
    openai_image_size: str = "1024x1024"
    ecommerce_image_size: str = "1024x1024"
    ecommerce_image_fit: str = "cover"
    openai_text_model: str = "gpt-5.5"
    openai_text_reasoning_effort: str | None = "xhigh"
    openai_store: bool = False
    color_card_catalog_path: str = (
        "data/catalogs/deekus_new_vinyl/deekus_new_vinyl_color_card.json"
    )
    color_material_qa_enabled: bool = True
    ai_watermark_detector_provider: str = "mock"
    ai_watermark_check_visible: bool = True
    ai_watermark_check_invisible: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
