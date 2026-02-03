"""
Visual themes for Monte Carlo π visualizations.

Provides multiple color schemes optimized for:
- Dark mode displays
- Publication (print-friendly)
- Colorblind accessibility
- High contrast presentations
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Theme:
    """Color theme for visualizations."""
    
    name: str
    
    # Core colors
    background: str
    panel: str
    text: str
    text_dim: str
    grid: str
    
    # Data colors
    inside: str
    outside: str
    accent: str
    highlight: str
    
    # Accuracy gradient (from excellent to poor)
    accuracy_excellent: str
    accuracy_good: str
    accuracy_moderate: str
    accuracy_poor: str
    accuracy_bad: str
    
    # Additional accents
    theoretical: str
    confidence_band: str
    
    @property
    def accuracy_colors(self) -> List[Tuple[float, str, str]]:
        """Returns (threshold, color, label) for accuracy levels."""
        return [
            (0.001, self.accuracy_excellent, "Exceptional"),
            (0.01, self.accuracy_excellent, "Excellent"),
            (0.05, self.accuracy_good, "Very Good"),
            (0.10, self.accuracy_moderate, "Good"),
            (0.20, self.accuracy_poor, "Moderate"),
            (float('inf'), self.accuracy_bad, "Poor"),
        ]
    
    def get_accuracy_color(self, error: float) -> str:
        """Get color based on estimation error."""
        for threshold, color, _ in self.accuracy_colors:
            if error < threshold:
                return color
        return self.accuracy_bad
    
    def get_accuracy_label(self, error: float) -> str:
        """Get label based on estimation error."""
        for threshold, _, label in self.accuracy_colors:
            if error < threshold:
                return label
        return "Poor"


# ═══════════════════════════════════════════════════════════════════════════════
# THEME DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

THEME_MIDNIGHT = Theme(
    name="midnight",
    background="#0a0a0f",
    panel="#12121a",
    text="#e8e8f0",
    text_dim="#6b6b80",
    grid="#2a2a3a",
    inside="#00d4aa",
    outside="#ff4d6a",
    accent="#4d9fff",
    highlight="#ffd93d",
    accuracy_excellent="#00ff88",
    accuracy_good="#88ff00",
    accuracy_moderate="#ffcc00",
    accuracy_poor="#ff8800",
    accuracy_bad="#ff3355",
    theoretical="#9d6eff",
    confidence_band="#00d4aa",
)

THEME_CYBERPUNK = Theme(
    name="cyberpunk",
    background="#0d0221",
    panel="#1a0533",
    text="#f0e6ff",
    text_dim="#8866aa",
    grid="#3d1a5c",
    inside="#00ffcc",
    outside="#ff0066",
    accent="#ff6600",
    highlight="#ffff00",
    accuracy_excellent="#00ffcc",
    accuracy_good="#66ff66",
    accuracy_moderate="#ffcc00",
    accuracy_poor="#ff6600",
    accuracy_bad="#ff0066",
    theoretical="#cc66ff",
    confidence_band="#00ffcc",
)

THEME_OCEAN = Theme(
    name="ocean",
    background="#001524",
    panel="#002233",
    text="#e0f4ff",
    text_dim="#5588aa",
    grid="#003344",
    inside="#00d4ff",
    outside="#ff6b8a",
    accent="#00ffaa",
    highlight="#ffdd44",
    accuracy_excellent="#00ffcc",
    accuracy_good="#44ddaa",
    accuracy_moderate="#ffcc44",
    accuracy_poor="#ff8844",
    accuracy_bad="#ff5566",
    theoretical="#aa88ff",
    confidence_band="#00d4ff",
)

THEME_FOREST = Theme(
    name="forest",
    background="#0a1209",
    panel="#121a11",
    text="#e8f0e6",
    text_dim="#6b806a",
    grid="#2a3a28",
    inside="#44dd66",
    outside="#dd6644",
    accent="#66aadd",
    highlight="#ddcc44",
    accuracy_excellent="#22ff66",
    accuracy_good="#88dd44",
    accuracy_moderate="#ddcc22",
    accuracy_poor="#dd8822",
    accuracy_bad="#dd4444",
    theoretical="#aa66dd",
    confidence_band="#44dd66",
)

THEME_PUBLICATION = Theme(
    name="publication",
    background="#ffffff",
    panel="#f8f8f8",
    text="#1a1a1a",
    text_dim="#666666",
    grid="#dddddd",
    inside="#2e7d32",
    outside="#c62828",
    accent="#1565c0",
    highlight="#f57c00",
    accuracy_excellent="#2e7d32",
    accuracy_good="#558b2f",
    accuracy_moderate="#f9a825",
    accuracy_poor="#ef6c00",
    accuracy_bad="#c62828",
    theoretical="#7b1fa2",
    confidence_band="#2e7d32",
)

THEME_COLORBLIND = Theme(
    name="colorblind",
    background="#1a1a2e",
    panel="#16213e",
    text="#e8e8f0",
    text_dim="#8888aa",
    grid="#2a2a4a",
    inside="#0077bb",   # Blue
    outside="#ee7733",  # Orange
    accent="#009988",   # Teal
    highlight="#cc3311", # Vermillion
    accuracy_excellent="#0077bb",
    accuracy_good="#33bbee",
    accuracy_moderate="#ee7733",
    accuracy_poor="#cc3311",
    accuracy_bad="#aa3377",
    theoretical="#009988",
    confidence_band="#0077bb",
)

THEME_NEON = Theme(
    name="neon",
    background="#000000",
    panel="#0a0a0a",
    text="#ffffff",
    text_dim="#888888",
    grid="#222222",
    inside="#39ff14",
    outside="#ff073a",
    accent="#00f0ff",
    highlight="#fff01f",
    accuracy_excellent="#39ff14",
    accuracy_good="#adff2f",
    accuracy_moderate="#fff01f",
    accuracy_poor="#ff6600",
    accuracy_bad="#ff073a",
    theoretical="#bf00ff",
    confidence_band="#39ff14",
)

THEME_SUNSET = Theme(
    name="sunset",
    background="#1a0a0a",
    panel="#2a1515",
    text="#ffe8e0",
    text_dim="#aa8888",
    grid="#3a2525",
    inside="#ff9966",
    outside="#6699ff",
    accent="#ff6699",
    highlight="#ffcc66",
    accuracy_excellent="#66ff99",
    accuracy_good="#99ff66",
    accuracy_moderate="#ffcc66",
    accuracy_poor="#ff9966",
    accuracy_bad="#ff6666",
    theoretical="#cc99ff",
    confidence_band="#ff9966",
)

# Theme registry
THEMES: Dict[str, Theme] = {
    "midnight": THEME_MIDNIGHT,
    "cyberpunk": THEME_CYBERPUNK,
    "ocean": THEME_OCEAN,
    "forest": THEME_FOREST,
    "publication": THEME_PUBLICATION,
    "colorblind": THEME_COLORBLIND,
    "neon": THEME_NEON,
    "sunset": THEME_SUNSET,
}

DEFAULT_THEME = THEME_MIDNIGHT


def get_theme(name: str = "midnight") -> Theme:
    """Get theme by name. Returns default if not found."""
    return THEMES.get(name.lower(), DEFAULT_THEME)


def list_themes() -> List[str]:
    """List all available theme names."""
    return list(THEMES.keys())
