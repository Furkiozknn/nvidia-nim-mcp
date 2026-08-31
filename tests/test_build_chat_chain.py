"""_build_chat_chain: NVIDIA models first, extra providers only when their
API key is actually set in the environment."""
import nvidia_image


def _clear_all_provider_keys(monkeypatch):
    for provider in nvidia_image.EXTRA_PROVIDERS:
        monkeypatch.delenv(provider["env"], raising=False)


def test_nvidia_models_come_first_in_declared_order(monkeypatch):
    _clear_all_provider_keys(monkeypatch)

    chain = nvidia_image._build_chat_chain(["nvidia/model-a", "nvidia/model-b"])

    assert len(chain) == 2
    assert chain[0]["model"] == "openai/nvidia/model-a"
    assert chain[1]["model"] == "openai/nvidia/model-b"
    for entry in chain:
        assert entry["api_base"] == "https://integrate.api.nvidia.com/v1"
        assert entry["api_key"] == nvidia_image.API_KEY


def test_no_extra_providers_when_no_keys_set(monkeypatch):
    _clear_all_provider_keys(monkeypatch)

    chain = nvidia_image._build_chat_chain(["nvidia/model-a"])

    assert len(chain) == 1
    assert chain[0]["model"] == "openai/nvidia/model-a"


def test_only_configured_providers_are_appended(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    # MISTRAL_API_KEY and CEREBRAS_API_KEY deliberately left unset.

    chain = nvidia_image._build_chat_chain(["nvidia/model-a"])

    models_after_nvidia = [entry["model"] for entry in chain[1:]]
    assert models_after_nvidia == ["groq/openai/gpt-oss-120b", "gemini/gemini-flash-latest"]
    assert "mistral/mistral-small-latest" not in models_after_nvidia
    assert "cerebras/gpt-oss-120b" not in models_after_nvidia


def test_extra_providers_preserve_declared_order(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    # Set every key; the chain order should follow EXTRA_PROVIDERS, not
    # environment insertion order.
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    chain = nvidia_image._build_chat_chain(["nvidia/model-a"])

    models_after_nvidia = [entry["model"] for entry in chain[1:]]
    assert models_after_nvidia == [
        "groq/openai/gpt-oss-120b",
        "mistral/mistral-small-latest",
        "gemini/gemini-flash-latest",
        "cerebras/gpt-oss-120b",
    ]


def test_each_extra_provider_carries_its_own_key(monkeypatch):
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "the-groq-key")

    chain = nvidia_image._build_chat_chain(["nvidia/model-a"])

    groq_entry = chain[-1]
    assert groq_entry["api_key"] == "the-groq-key"
