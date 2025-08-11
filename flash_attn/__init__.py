__version__ = "2.8.2"

# Try to import the compiled CUDA interface, but gracefully skip if unavailable.
# This makes the package import safe by default for Cute-DSL-only usage.
try:  # noqa: SIM105
    from flash_attn.flash_attn_interface import (  # type: ignore
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
except Exception:  # Compiled extension not present; top-level shortcuts unavailable
    pass
