import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

# Load variables from a local .env file (if present) into os.environ.
# Real environment variables take precedence over .env values.
load_dotenv()


class Settings(BaseModel):
    """Application settings, validated by Pydantic."""

    app_name: str
    app_version: str
    environment: Literal["development", "staging", "production"]


# Read from environment variables, falling back to sensible defaults.
# An invalid value (e.g. ENVIRONMENT=prod) fails fast at startup.
settings = Settings.model_validate(
    {
        "app_name": os.getenv("APP_NAME", "Enterprise AI Knowledge Assistant"),
        "app_version": os.getenv("APP_VERSION", "1.0.0"),
        "environment": os.getenv("ENVIRONMENT", "development"),
    }
)
