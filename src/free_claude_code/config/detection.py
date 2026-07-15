"""Provider type detection from base URL hostname.

When a user adds a custom provider with only a ``base_url``, this module
inspects the hostname to guess which provider profile applies. Unknown
hosts fall through to ``None`` → generic OpenAI-compatible.
"""

from urllib.parse import urlsplit

# Hostname → provider profile id (matches keys in OPENAI_CHAT_PROFILES
# or special factory ids). Only hosts whose profile differs from
# generic OpenAI-compatible behavior need explicit entries.
HOST_PROFILE_MAP: dict[str, str] = {
    "api.deepseek.com": "deepseek",
    "api.mistral.ai": "mistral",
    "codestral.mistral.ai": "mistral_codestral",
    "api.groq.com": "groq",
    "api.cerebras.ai": "cerebras",
    "api.sambanova.ai": "sambanova",
    "api.fireworks.ai": "fireworks",
    "api.cohere.ai": "cohere",
    "api.moonshot.ai": "kimi",
    "api.minimax.io": "minimax",
    "pass.wafer.ai": "wafer",
    "api.z.ai": "zai",
    "generativelanguage.googleapis.com": "gemini",
    "openrouter.ai": "open_router",
    "integrate.api.nvidia.com": "nvidia_nim",
    "router.huggingface.co": "huggingface",
    "ai-gateway.vercel.sh": "vercel",
    "models.github.ai": "github_models",
    "opencode.ai": "opencode",
    "ollama.com": "ollama_cloud",
}


def detect_provider_profile(base_url: str) -> str | None:
    """Return a known provider profile id for *base_url*, or None.

    Parses the hostname from *base_url* and looks it up in
    ``HOST_PROFILE_MAP``. Returns ``None`` for unknown hosts
    (caller should apply the generic OpenAI-compatible fallback).
    """
    if not base_url:
        return None
    host = _extract_host(base_url)
    if not host:
        return None
    host_lower = host.lower()

    # Exact match first
    if host_lower in HOST_PROFILE_MAP:
        return HOST_PROFILE_MAP[host_lower]

    # Subdomain / suffix match: e.g. "api.openai.com" not in map → None
    for known_host, profile_id in HOST_PROFILE_MAP.items():
        if host_lower == known_host or host_lower.endswith("." + known_host):
            return profile_id

    return None


def generate_provider_id(base_url: str) -> str:
    """Generate a stable custom provider id from a base URL hostname.

    Examples:
        ``https://api.my-llm.com/v1`` → ``custom_my_llm_com``
        ``http://localhost:8000/v1`` → ``custom_localhost_8000``
    """
    host = _extract_host(base_url) or "unknown"
    normalized = (
        host.lower().replace(".", "_").replace("-", "_").replace(":", "_").strip("_")
    )
    return f"custom_{normalized}"


def _extract_host(url: str) -> str | None:
    """Return the hostname (including port) from *url*, or None."""
    try:
        parsed = urlsplit(url)
        if parsed.hostname:
            host = parsed.hostname.lower()
            if parsed.port:
                return f"{host}:{parsed.port}"
            return host
    except Exception:
        pass
    # Fallback: strip scheme and path manually
    cleaned = url.strip()
    if "://" in cleaned:
        cleaned = cleaned.split("://", 1)[1]
    host_part = cleaned.split("/", 1)[0].split("?")[0].split("#")[0]
    return host_part.strip().lower() or None
