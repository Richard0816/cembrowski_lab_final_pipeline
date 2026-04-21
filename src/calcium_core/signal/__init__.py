"""Pure signal processing — no plotting, no file I/O."""
from __future__ import annotations

from .filters import lowpass_causal_1d, sg_first_derivative_1d
from .normalize import mad_z, robust_df_over_f_1d
from .spikes import hysteresis_onsets

__all__ = [
    "lowpass_causal_1d",
    "sg_first_derivative_1d",
    "robust_df_over_f_1d",
    "mad_z",
    "hysteresis_onsets",
]
