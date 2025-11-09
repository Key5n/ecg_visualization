import seaborn as sns

ABNORMAL_INTERVAL_COLOR = "#f4a261"
EXTREME_INTERVAL_COLOR = "#2a9d8f"

_CUSTOM_PARAMS = {
    "lines.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "axes.linewidth": 0.5,
    "xtick.labelsize": 6,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
}


def apply_default_style() -> None:
    """Configure Seaborn/matplotlib rcParams for ECG figures."""
    sns.set_theme(context="paper", style="whitegrid", rc=_CUSTOM_PARAMS)
