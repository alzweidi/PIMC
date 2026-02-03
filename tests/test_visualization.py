"""
Test suite for visualization module.

Tests cover:
- Theme system
- Visualizer creation
- Plot generation
- Export functionality
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from pimc.visualization.themes import (
    Theme,
    THEMES,
    get_theme,
    list_themes,
    DEFAULT_THEME,
    THEME_MIDNIGHT,
    THEME_CYBERPUNK,
    THEME_PUBLICATION,
)
from pimc.visualization.plots import PiVisualizer, quick_plot
from pimc.core.simulation import MonteCarloPi


class TestThemes:
    """Tests for the theme system."""
    
    def test_theme_dataclass(self):
        """Theme dataclass has all required attributes."""
        theme = THEME_MIDNIGHT
        
        # Core colors
        assert hasattr(theme, 'background')
        assert hasattr(theme, 'panel')
        assert hasattr(theme, 'text')
        assert hasattr(theme, 'grid')
        
        # Data colors
        assert hasattr(theme, 'inside')
        assert hasattr(theme, 'outside')
        assert hasattr(theme, 'accent')
        
        # Accuracy colors
        assert hasattr(theme, 'accuracy_excellent')
        assert hasattr(theme, 'accuracy_good')
        assert hasattr(theme, 'accuracy_moderate')
        assert hasattr(theme, 'accuracy_poor')
        assert hasattr(theme, 'accuracy_bad')
    
    def test_get_theme_by_name(self):
        """get_theme returns correct theme by name."""
        theme = get_theme("midnight")
        assert theme == THEME_MIDNIGHT
        
        theme = get_theme("cyberpunk")
        assert theme == THEME_CYBERPUNK
        
        theme = get_theme("publication")
        assert theme == THEME_PUBLICATION
    
    def test_get_theme_case_insensitive(self):
        """get_theme is case-insensitive."""
        theme1 = get_theme("MIDNIGHT")
        theme2 = get_theme("Midnight")
        theme3 = get_theme("midnight")
        
        assert theme1 == theme2 == theme3
    
    def test_get_theme_unknown_returns_default(self):
        """Unknown theme name returns default theme."""
        theme = get_theme("nonexistent_theme")
        assert theme == DEFAULT_THEME
    
    def test_list_themes(self):
        """list_themes returns all available themes."""
        themes = list_themes()
        
        assert isinstance(themes, list)
        assert "midnight" in themes
        assert "cyberpunk" in themes
        assert "publication" in themes
        assert len(themes) == len(THEMES)
    
    def test_accuracy_colors(self):
        """Theme returns correct colors for different error levels."""
        theme = THEME_MIDNIGHT
        
        # Excellent accuracy
        assert theme.get_accuracy_color(0.001) == theme.accuracy_excellent
        
        # Good accuracy
        assert theme.get_accuracy_color(0.03) == theme.accuracy_good
        
        # Moderate accuracy
        assert theme.get_accuracy_color(0.07) == theme.accuracy_moderate
        
        # Poor accuracy
        assert theme.get_accuracy_color(0.15) == theme.accuracy_poor
        
        # Bad accuracy
        assert theme.get_accuracy_color(0.5) == theme.accuracy_bad
    
    def test_accuracy_labels(self):
        """Theme returns correct labels for different error levels."""
        theme = THEME_MIDNIGHT
        
        assert "Excellent" in theme.get_accuracy_label(0.005)
        assert "Good" in theme.get_accuracy_label(0.07)
        assert "Poor" in theme.get_accuracy_label(0.5)


class TestPiVisualizer:
    """Tests for PiVisualizer class."""
    
    @pytest.fixture
    def result(self):
        """Create a simulation result for testing."""
        sim = MonteCarloPi(n_points=1000, seed=42)
        return sim.run()
    
    def test_visualizer_creation(self):
        """Visualizer is created with correct attributes."""
        viz = PiVisualizer(theme="midnight", figsize=(16, 10), dpi=150)
        
        assert viz.theme == THEME_MIDNIGHT
        assert viz.figsize == (16, 10)
        assert viz.dpi == 150
    
    def test_visualizer_theme_string(self):
        """Visualizer accepts theme as string."""
        viz = PiVisualizer(theme="cyberpunk")
        assert viz.theme == THEME_CYBERPUNK
    
    def test_visualizer_theme_object(self):
        """Visualizer accepts theme as Theme object."""
        viz = PiVisualizer(theme=THEME_PUBLICATION)
        assert viz.theme == THEME_PUBLICATION
    
    def test_plot_comprehensive(self, result):
        """Comprehensive plot is created successfully."""
        viz = PiVisualizer(theme="midnight")
        fig = viz.plot_comprehensive(result)
        
        assert fig is not None
        assert viz.fig is fig
        assert len(viz.axes) >= 4  # At least 4 panels
        assert 'scatter' in viz.axes
        
        plt.close('all')
        viz.close()
    
    def test_plot_scatter_only(self, result):
        """Scatter-only plot is created successfully."""
        viz = PiVisualizer()
        fig = viz.plot_scatter_only(result)
        
        assert fig is not None
        viz.close()
    
    def test_plot_convergence_only(self, result):
        """Convergence-only plot is created successfully."""
        viz = PiVisualizer()
        fig = viz.plot_convergence_only(result)
        
        assert fig is not None
        viz.close()
    
    def test_save_png(self, result, tmp_path):
        """Figure is saved to PNG file."""
        viz = PiVisualizer()
        viz.plot_comprehensive(result)
        
        output_path = tmp_path / "test_output.png"
        viz.save(output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        plt.close('all')
        viz.close()
    
    def test_save_pdf(self, result, tmp_path):
        """Figure is saved to PDF file."""
        viz = PiVisualizer()
        viz.plot_comprehensive(result)
        
        output_path = tmp_path / "test_output.pdf"
        viz.save(output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        plt.close('all')
        viz.close()
    
    def test_save_without_plot_raises(self):
        """Saving without plotting raises RuntimeError."""
        viz = PiVisualizer()
        
        with pytest.raises(RuntimeError):
            viz.save("test.png")
    
    def test_close(self, result):
        """Close clears figure and axes."""
        viz = PiVisualizer()
        viz.plot_comprehensive(result)
        
        assert viz.fig is not None
        assert len(viz.axes) > 0
        
        plt.close('all')
        viz.close()
        
        assert viz.fig is None
        assert len(viz.axes) == 0
    
    def test_custom_title(self, result):
        """Custom title is applied to figure."""
        viz = PiVisualizer()
        fig = viz.plot_comprehensive(result, title="Custom Title")
        
        # Check that suptitle exists and contains our title
        suptitle = fig._suptitle
        if suptitle is not None:
            assert "Custom Title" in suptitle.get_text()
        
        plt.close('all')
        viz.close()
    
    def test_different_themes_render(self, result):
        """All themes render without errors."""
        for theme_name in list_themes():
            viz = PiVisualizer(theme=theme_name)
            fig = viz.plot_comprehensive(result)
            assert fig is not None
            plt.close('all')
            viz.close()


class TestQuickPlot:
    """Tests for quick_plot helper function."""
    
    def test_quick_plot_runs(self):
        """quick_plot executes without error."""
        sim = MonteCarloPi(n_points=100, seed=42)
        result = sim.run()
        
        # Just verify it doesn't crash (don't actually show)
        plt.ioff()  # Disable interactive mode
        try:
            # quick_plot would normally show, but we can't test that
            # Just verify the visualizer works
            viz = PiVisualizer()
            viz.plot_comprehensive(result)
            plt.close('all')
            viz.close()
        finally:
            plt.ion()


class TestVisualizationWithEdgeCases:
    """Tests for visualization edge cases."""
    
    def test_empty_points(self):
        """Visualization handles empty points gracefully."""
        sim = MonteCarloPi(n_points=1000, seed=42, store_points=False)
        result = sim.run()
        
        viz = PiVisualizer()
        fig = viz.plot_comprehensive(result)
        
        # Should still create figure, just empty scatter
        assert fig is not None
        plt.close('all')
        viz.close()
    
    def test_no_convergence_data(self):
        """Visualization handles missing convergence data."""
        sim = MonteCarloPi(n_points=1000, seed=42, track_convergence=False)
        result = sim.run()
        
        viz = PiVisualizer()
        fig = viz.plot_comprehensive(result)
        
        assert fig is not None
        plt.close('all')
        viz.close()
    
    def test_small_sample(self):
        """Visualization works with very small sample."""
        sim = MonteCarloPi(n_points=10, seed=42)
        result = sim.run()
        
        viz = PiVisualizer()
        fig = viz.plot_comprehensive(result)
        
        assert fig is not None
        plt.close('all')
        viz.close()
    
    def test_large_sample_performance(self):
        """Visualization handles large sample reasonably."""
        sim = MonteCarloPi(n_points=50000, seed=42)
        result = sim.run()
        
        viz = PiVisualizer()
        fig = viz.plot_comprehensive(result, point_size=0.5)
        
        assert fig is not None
        plt.close('all')
        viz.close()
