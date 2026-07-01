"""Utilities del framework ONEPAI."""

from .checksum import sha256_bytes, sha256_file, hmac_sign, hmac_verify, CRCRecord
from .compression import compress, decompress, compress_ratio
from .config import ONEPAIConfig, load_config, save_config, get_default_config
from .decorators import timer, validate_ndarray, retry, requires_numpy
from .invisible_math import (
    entropy, kl_divergence, js_divergence, cosine_similarity,
    silence_score, void_depth, activation_sparsity, gini_coefficient,
    softmax, normalize,
)
from .logger import get_logger, configure_logging
from .shadow_utils import (
    compute_shadow_matrix, find_shadow_clusters, shadow_overlap,
    activation_to_shadow_map, merge_shadow_maps, shadow_diff,
)

__all__ = [
    "sha256_bytes", "sha256_file", "hmac_sign", "hmac_verify", "CRCRecord",
    "compress", "decompress", "compress_ratio",
    "ONEPAIConfig", "load_config", "save_config", "get_default_config",
    "timer", "validate_ndarray", "retry", "requires_numpy",
    "entropy", "kl_divergence", "js_divergence", "cosine_similarity",
    "silence_score", "void_depth", "activation_sparsity", "gini_coefficient",
    "softmax", "normalize",
    "get_logger", "configure_logging",
    "compute_shadow_matrix", "find_shadow_clusters", "shadow_overlap",
    "activation_to_shadow_map", "merge_shadow_maps", "shadow_diff",
]
