"""Minimal provider route shared by prediction and legacy execution code.

This module deliberately contains no benchmark, split, population, scorer, or
judge imports.  Provider SDK imports remain behind the call boundary so
provider-free planning and replay can import the route without loading an SDK.
"""

from __future__ import annotations

import ssl
from typing import Any


DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_TERRA_GATEWAY_MODEL = "codex_sdk/gpt-5.6-terra"
DEFAULT_TERRA_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-terra"
DEFAULT_API_KEY_ENV = "LITELLM_KEY"


def make_provider_client(api_key: str, gateway_url: str) -> Any:
    """Build the truststore-backed, zero-retry OpenAI-compatible client."""

    import httpx
    import truststore
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=gateway_url,
        http_client=httpx.Client(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ),
        max_retries=0,
    )
