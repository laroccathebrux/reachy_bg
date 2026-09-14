"""Test-wide setup: never read the developer's real .env, so tests are deterministic."""

import os

os.environ["REACHY_BG_DOTENV"] = ""  # must be set before src.config is imported anywhere
