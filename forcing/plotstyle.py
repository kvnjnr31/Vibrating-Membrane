"""Shared matplotlib style for the experiment figures."""
import matplotlib

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# fixed colour per entity (never reassigned by plotting order)
DRIVE_COLOR = {"pll": "#4a3aa7", "pll+gain": "#008300", "pll+agc": "#eda100",
               "table": "#1baf7a", "table+gain": "#e87ba4",
               "fixed": "#2a78d6", "naive": "#eb6834", "self": "#4a3aa7"}
# velocity feedback shares the loop's hue (both are feedback) but is drawn dashed
DRIVE_LINESTYLE = {"self": (0, (5, 2))}
DRIVE_LABEL = {"pll": "Phase-locked loop",
               "pll+gain": "Phase-locked loop + feed-forward gain",
               "pll+agc": "Phase-locked loop + amplitude loop",
               "table": "Table lookup (baseline)",
               "table+gain": "Table lookup + feed-forward gain (baseline)",
               "fixed": "Fixed frequency",
               "naive": "Direct substitution, cos(w(t) t)",
               "self": "Velocity feedback (self-excited)"}
REFERENCE = INK2                     # predictions / expected values (dashed)
GRAIN = "#2a78d6"
SEQUENTIAL = "Blues"


def apply():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150,
    })
