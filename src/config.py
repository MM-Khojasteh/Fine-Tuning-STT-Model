"""Centralized configuration loading with environment-variable overrides.

``config/config.yaml`` defines the public configuration schema with safe,
generic defaults. Every operational value -- secrets, dataset locations,
tuned hyperparameters, diagnostic thresholds and feature toggles -- can be
overridden through environment variables, typically supplied via a local
``.env`` file that is never committed (see ``.env.example`` for the full
documented matrix).

Precedence (highest wins):
    1. Real environment variables exported in the shell.
    2. Values from the local ``.env`` file (python-dotenv, no clobbering).
    3. Defaults from ``config/config.yaml``.
"""

import os
import re
from typing import Any, Callable, Dict, Optional, Tuple

import yaml
from dotenv import load_dotenv

from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = "config/config.yaml"

# Freely available public checkpoint used as the default base model.
# This is a well-known Hugging Face model, not project IP.
DEFAULT_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"

# Numeric string patterns accepted by convert_numeric_values (YAML 1.1 parses
# bare exponent notation such as `1e-6` as a string, so we normalize here).
_INT_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
_NULL_TOKENS = {"", "none", "null"}


# ---------------------------------------------------------------------------
# Environment variable parsers (strict: fail fast with a clear error)
# ---------------------------------------------------------------------------

def _parse_str(raw: str, env_var: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("must not be empty")
    return value


def _parse_bool(raw: str, env_var: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"invalid boolean for {env_var}: {raw!r} "
        f"(expected one of: {', '.join(sorted(_TRUTHY | _FALSY))})"
    )


def _parse_int(raw: str, env_var: str) -> int:
    value = raw.strip()
    if not _INT_RE.fullmatch(value):
        raise ValueError(f"invalid integer for {env_var}: {raw!r}")
    return int(value)


def _parse_float(raw: str, env_var: str) -> float:
    value = raw.strip()
    if not _FLOAT_RE.fullmatch(value):
        raise ValueError(f"invalid number for {env_var}: {raw!r}")
    return float(value)


def _parse_optional_str(raw: str, env_var: str) -> Optional[str]:
    value = raw.strip()
    return None if value.lower() in _NULL_TOKENS else value


def _parse_optional_int(raw: str, env_var: str) -> Optional[int]:
    value = raw.strip()
    if value.lower() in _NULL_TOKENS:
        return None
    return _parse_int(value, env_var)


def _parse_optional_float(raw: str, env_var: str) -> Optional[float]:
    value = raw.strip()
    if value.lower() in _NULL_TOKENS:
        return None
    return _parse_float(value, env_var)


# ---------------------------------------------------------------------------
# Override matrix: (env var, config section, config key, parser)
# Every entry is documented in .env.example. Keep the two files in sync.
# ---------------------------------------------------------------------------

ENV_OVERRIDES: Tuple[Tuple[str, str, str, Callable[[str, str], Any]], ...] = (
    # Base model / checkpoint
    ("WAV2VEC2_MODEL_NAME", "model", "name", _parse_str),

    # Dataset locations & split
    ("WAV2VEC2_AUDIO_FILES_DIR", "data", "audio_files_dir", _parse_str),
    ("WAV2VEC2_JSON_INFO_DIR", "data", "json_info_dir", _parse_str),
    ("WAV2VEC2_TEST_AUDIO_DIR", "data", "test_audio_dir", _parse_str),
    ("WAV2VEC2_TRAIN_VAL_SPLIT", "data", "train_val_split", _parse_float),

    # Training matrix
    ("WAV2VEC2_BATCH_SIZE", "training", "batch_size", _parse_int),
    ("WAV2VEC2_NUM_EPOCHS", "training", "num_epochs", _parse_int),
    ("WAV2VEC2_LEARNING_RATE", "training", "learning_rate", _parse_float),
    ("WAV2VEC2_WEIGHT_DECAY", "training", "weight_decay", _parse_float),
    ("WAV2VEC2_WARMUP_STEPS", "training", "warmup_steps", _parse_int),
    ("WAV2VEC2_GRADIENT_ACCUMULATION_STEPS", "training", "gradient_accumulation_steps", _parse_int),
    ("WAV2VEC2_MAX_AUDIO_LENGTH", "training", "max_audio_length", _parse_optional_float),
    ("WAV2VEC2_NUM_WORKERS", "training", "num_workers", _parse_int),
    ("WAV2VEC2_USE_REDUCE_LR_ON_PLATEAU", "training", "use_reduce_lr_on_plateau", _parse_bool),
    ("WAV2VEC2_REDUCE_LR_PATIENCE", "training", "reduce_lr_patience", _parse_int),
    ("WAV2VEC2_REDUCE_LR_FACTOR", "training", "reduce_lr_factor", _parse_float),
    ("WAV2VEC2_REDUCE_LR_MIN", "training", "reduce_lr_min", _parse_float),
    ("WAV2VEC2_EARLY_STOPPING_PATIENCE", "training", "early_stopping_patience", _parse_optional_int),
    ("WAV2VEC2_EARLY_STOPPING_MIN_DELTA", "training", "early_stopping_min_delta", _parse_float),
    ("WAV2VEC2_GRADIENT_CLIP_NORM", "training", "gradient_clip_norm", _parse_float),
    ("WAV2VEC2_RANDOM_SEED", "training", "random_seed", _parse_int),

    # Inference behaviour
    ("WAV2VEC2_CHUNK_DURATION", "inference", "chunk_duration", _parse_int),
    ("WAV2VEC2_SKIP_SHORT_CHUNKS", "inference", "skip_short_chunks", _parse_bool),
    ("WAV2VEC2_MIN_CHUNK_SAMPLES", "inference", "min_chunk_samples", _parse_int),
    ("WAV2VEC2_BLANK_TOKEN_THRESHOLD", "inference", "blank_token_threshold", _parse_float),

    # Output & logging
    ("WAV2VEC2_OUTPUT_DIR", "output", "output_dir", _parse_str),
    ("WAV2VEC2_SAVE_EVERY_N_EPOCHS", "output", "save_every_n_epochs", _parse_int),
    ("WAV2VEC2_LOG_FILE", "output", "log_file", _parse_optional_str),
    ("WAV2VEC2_LOG_LEVEL", "output", "log_level", _parse_str),

    # Device / runtime toggles
    ("WAV2VEC2_USE_CUDA", "device", "use_cuda", _parse_bool),
    ("WAV2VEC2_MIXED_PRECISION", "device", "mixed_precision", _parse_bool),
    ("WAV2VEC2_EMPTY_CACHE_AFTER_VAL", "device", "empty_cache_after_val", _parse_bool),
)


def convert_numeric_values(obj: Any) -> Any:
    """Recursively convert numeric string values to proper numeric types.

    YAML 1.1 resolves bare exponent notation such as ``1e-6`` to a string.
    This normalizes any numeric-looking string (including scientific
    notation) to int/float so downstream consumers always receive the
    correct type.

    Args:
        obj: Object to process (dict, list, or scalar).

    Returns:
        Object with numeric strings converted to int/float.
    """
    if isinstance(obj, dict):
        return {key: convert_numeric_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numeric_values(item) for item in obj]
    elif isinstance(obj, str):
        value = obj.strip()
        if _INT_RE.fullmatch(value):
            return int(value)
        if _FLOAT_RE.fullmatch(value):
            return float(value)
    return obj


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to a loaded configuration.

    Empty/whitespace-only variables are treated as unset. Invalid values
    raise immediately (fail fast) with the offending variable name.
    Override *names* are logged, never their values.

    Args:
        config: Configuration dictionary loaded from YAML.

    Returns:
        The same dictionary with overrides applied in place.
    """
    for env_var, section, key, parser in ENV_OVERRIDES:
        raw = os.getenv(env_var)
        if raw is None or not raw.strip():
            continue
        try:
            value = parser(raw, env_var)
        except ValueError as exc:
            raise ValueError(f"Invalid environment variable: {exc}") from exc
        config.setdefault(section, {})[key] = value
        logger.info(f"Config override applied: {env_var} -> {section}.{key}")
    return config


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML, then apply environment overrides.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary with numeric values properly typed and
        environment overrides applied.
    """
    # Load local .env (does not clobber real environment variables).
    load_dotenv()

    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = convert_numeric_values(config)
    config = apply_env_overrides(config)

    logger.debug("Configuration loaded and processed successfully")
    return config


def get_hf_token() -> Optional[str]:
    """Return the Hugging Face access token from the environment, if set.

    Reads ``HUGGINGFACE_TOKEN`` (typically from the local ``.env`` file).
    The token is a secret: it must never be hardcoded or committed.

    Returns:
        The stripped token string, or None if not configured.
    """
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token and token.strip():
        return token.strip()
    return None


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an optional string environment variable with a fallback."""
    load_dotenv()
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Read an optional integer environment variable with a fallback."""
    raw = env_str(name)
    if raw is None:
        return default
    try:
        return _parse_int(raw, name)
    except ValueError as exc:
        logger.warning(f"{exc}; falling back to default {default!r}")
        return default
