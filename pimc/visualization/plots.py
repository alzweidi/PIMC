"""
Publication-quality visualization for Monte Carlo π estimation.

Features:
- Beautiful static plots with multiple themes
- Real-time animated simulations
- Convergence analysis plots
- Statistical analysis dashboards
- Export to multiple formats (PNG, PDF, SVG)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import PathCollection
import matplotlib.patheffects as path_effects

from pimc.visualization.themes import Theme, get_theme, THEME_MIDNIGHT


class PiVisualizer:
    """
    Publication-quality visualization for Monte Carlo π estimation.
    
    Creates stunning visualizations including:
    - Scatter plots of random points
    - Convergence analysis
    - Error decay plots
    - Real-time animations
    - Statistical dashboards
    
    Example:
        >>> from pimc import MonteCarloPi, PiVisualizer
        >>> sim = MonteCarloPi(n_points=50000)
        >>> result = sim.run()
        >>> viz = PiVisualizer(theme="cyberpunk")
        >>> viz.plot_comprehensive(result)
        >>> viz.save("pi_visualization.png")
    """
    
    def __init__(
        self,
        theme: Union[str, Theme] = "midnight",
        figsize: Tuple[float, float] = (16, 10),
        dpi: int = 150,
    ):
        if isinstance(theme, str):
            theme = get_theme(theme)
        
        self.theme = theme
        self.figsize = figsize
        self.dpi = dpi
        self.fig: Optional[plt.Figure] = None
        self.axes: dict = {}
        
        self._configure_matplotlib()
    
    def _configure_matplotlib(self) -> None:
        """Configure matplotlib for beautiful output."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['SF Pro Display', 'Helvetica Neue', 'Arial', 'sans-serif'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': self.dpi,
            'savefig.dpi': 300,
            'axes.linewidth': 1.0,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.5,
            'axes.spines.top': True,
            'axes.spines.right': True,
        })
    
    def _style_axis(self, ax: plt.Axes, show_grid: bool = False) -> None:
        """Apply theme styling to an axis."""
        ax.set_facecolor(self.theme.panel)
        ax.tick_params(colors=self.theme.text, which='both')
        
        for spine in ax.spines.values():
            spine.set_color(self.theme.grid)
            spine.set_linewidth(1.0)
        
        if show_grid:
            ax.grid(True, alpha=0.3, color=self.theme.grid, linestyle='-', linewidth=0.5)
    
    def plot_comprehensive(
        self,
        result,  # SimulationResult
        title: Optional[str] = None,
        show_legend: bool = True,
        point_size: float = 1.0,
        alpha: float = 0.6,
    ) -> plt.Figure:
        """
        Create comprehensive 4-panel visualization.
        
        Panels:
        1. Scatter plot of points (inside/outside quarter circle)
        2. Results display with accuracy indicator
        3. Convergence plot showing estimate approaching π
        4. Error analysis plot showing 1/√n decay
        """
        self.fig = plt.figure(figsize=self.figsize, facecolor=self.theme.background)
        
        if title is None:
            title = "Monte Carlo Estimation of π"
        
        self.fig.suptitle(
            title,
            fontsize=22,
            fontweight='bold',
            color=self.theme.text,
            y=0.97,
        )
        
        gs = gridspec.GridSpec(
            2, 3,
            height_ratios=[1.2, 1],
            width_ratios=[1.2, 0.6, 1],
            hspace=0.30,
            wspace=0.25,
            left=0.06,
            right=0.97,
            top=0.90,
            bottom=0.08,
        )
        
        # Panel 1: Scatter plot
        ax1 = self.fig.add_subplot(gs[0, 0])
        self._plot_scatter(ax1, result, point_size, alpha, show_legend)
        
        # Panel 2: Results display
        ax2 = self.fig.add_subplot(gs[0, 1])
        self._plot_results(ax2, result)
        
        # Panel 3: Method explanation
        ax3 = self.fig.add_subplot(gs[0, 2])
        self._plot_method(ax3, result)
        
        # Panel 4: Convergence
        ax4 = self.fig.add_subplot(gs[1, :2])
        self._plot_convergence(ax4, result)
        
        # Panel 5: Error decay
        ax5 = self.fig.add_subplot(gs[1, 2])
        self._plot_error_decay(ax5, result)
        
        self.axes = {
            'scatter': ax1,
            'results': ax2,
            'method': ax3,
            'convergence': ax4,
            'error': ax5,
        }
        
        return self.fig
    
    def _plot_scatter(
        self,
        ax: plt.Axes,
        result,
        point_size: float,
        alpha: float,
        show_legend: bool,
    ) -> None:
        """Plot scatter of random points."""
        self._style_axis(ax)
        
        x, y, inside = result.x_coords, result.y_coords, result.inside_mask
        
        if len(x) > 0:
            # Plot outside points first (behind)
            ax.scatter(
                x[~inside], y[~inside],
                c=self.theme.outside,
                s=point_size,
                alpha=alpha,
                rasterized=True,
                label=f'Outside ({np.sum(~inside):,})',
            )
            
            # Plot inside points
            ax.scatter(
                x[inside], y[inside],
                c=self.theme.inside,
                s=point_size,
                alpha=alpha,
                rasterized=True,
                label=f'Inside ({np.sum(inside):,})',
            )
        
        # Draw quarter circle arc
        theta = np.linspace(0, np.pi / 2, 200)
        ax.plot(
            np.cos(theta), np.sin(theta),
            color=self.theme.text,
            linewidth=2.5,
            label='Unit circle',
            zorder=10,
        )
        
        # Draw boundary
        ax.plot(
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            color=self.theme.text_dim,
            linewidth=1.5,
            linestyle='--',
            alpha=0.5,
        )
        
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect('equal')
        ax.set_xlabel('x', color=self.theme.text, fontsize=12)
        ax.set_ylabel('y', color=self.theme.text, fontsize=12)
        ax.set_title(
            f'Random Sampling (n = {result.n_points:,})',
            fontweight='bold',
            color=self.theme.text,
            pad=12,
        )
        
        if show_legend:
            legend = ax.legend(
                loc='upper right',
                facecolor=self.theme.panel,
                edgecolor=self.theme.grid,
                labelcolor=self.theme.text,
                framealpha=0.95,
            )
            legend.get_frame().set_linewidth(1.0)
    
    def _plot_results(self, ax: plt.Axes, result) -> None:
        """Plot results panel with π estimate and accuracy."""
        ax.set_facecolor(self.theme.panel)
        ax.axis('off')
        
        accuracy_color = self.theme.get_accuracy_color(result.error)
        accuracy_label = self.theme.get_accuracy_label(result.error)
        
        # Draw border
        border = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor='none',
            edgecolor=self.theme.grid,
            linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(border)
        
        # Estimated value
        ax.text(
            0.5, 0.90, 'ESTIMATED',
            fontsize=10,
            ha='center', va='center',
            color=self.theme.text_dim,
            transform=ax.transAxes,
            fontweight='bold',
        )
        
        ax.text(
            0.5, 0.78, 'π̂',
            fontsize=28,
            ha='center', va='center',
            color=self.theme.text,
            transform=ax.transAxes,
        )
        
        ax.text(
            0.5, 0.60, f'{result.estimate:.8f}',
            fontsize=20,
            ha='center', va='center',
            color=accuracy_color,
            transform=ax.transAxes,
            fontweight='bold',
            fontfamily='monospace',
        )
        
        # Accuracy badge
        badge = FancyBboxPatch(
            (0.25, 0.44), 0.50, 0.10,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=accuracy_color,
            edgecolor='none',
            alpha=0.2,
            transform=ax.transAxes,
        )
        ax.add_patch(badge)
        
        ax.text(
            0.5, 0.49, accuracy_label.upper(),
            fontsize=11,
            ha='center', va='center',
            color=accuracy_color,
            transform=ax.transAxes,
            fontweight='bold',
        )
        
        # True value
        ax.text(
            0.5, 0.32, 'TRUE VALUE',
            fontsize=9,
            ha='center', va='center',
            color=self.theme.text_dim,
            transform=ax.transAxes,
        )
        
        ax.text(
            0.5, 0.22, f'{np.pi:.8f}',
            fontsize=14,
            ha='center', va='center',
            color=self.theme.text_dim,
            transform=ax.transAxes,
            fontfamily='monospace',
        )
        
        # Error
        ax.text(
            0.5, 0.08,
            f'Error: {result.error:.2e} ({result.error_percent:.4f}%)',
            fontsize=10,
            ha='center', va='center',
            color=accuracy_color,
            transform=ax.transAxes,
        )
    
    def _plot_method(self, ax: plt.Axes, result) -> None:
        """Plot method explanation panel."""
        ax.set_facecolor(self.theme.panel)
        ax.axis('off')
        
        # Draw border
        border = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor='none',
            edgecolor=self.theme.grid,
            linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(border)
        
        items = [
            (0.50, 0.92, 'THE METHOD', 12, 'bold', self.theme.text),
            (0.50, 0.80, 'Area ratio:', 11, 'normal', self.theme.text_dim),
            (0.50, 0.70, 'A_circle / A_square = π/4', 11, 'normal', self.theme.text),
            (0.50, 0.56, 'π = 4 × (N_inside / N_total)', 12, 'bold', self.theme.highlight),
            (0.50, 0.40, 'STATISTICS', 12, 'bold', self.theme.text),
            (0.50, 0.30, f'Total samples: {result.n_points:,}', 10, 'normal', self.theme.text_dim),
            (0.50, 0.22, f'Inside: {result.n_inside:,} ({result.ratio*100:.2f}%)', 10, 'normal', self.theme.inside),
            (0.50, 0.14, f'Outside: {result.n_points - result.n_inside:,} ({(1-result.ratio)*100:.2f}%)', 10, 'normal', self.theme.outside),
            (0.50, 0.05, f'Time: {result.elapsed_time*1000:.1f}ms', 9, 'normal', self.theme.text_dim),
        ]
        
        for x, y, text, size, weight, color in items:
            ax.text(
                x, y, text,
                fontsize=size,
                ha='center', va='center',
                color=color,
                transform=ax.transAxes,
                fontweight=weight,
            )
    
    def _plot_convergence(self, ax: plt.Axes, result) -> None:
        """Plot convergence of π estimate over samples."""
        self._style_axis(ax, show_grid=True)
        
        if result.convergence_data is None:
            ax.text(
                0.5, 0.5, 'Convergence data not available',
                ha='center', va='center',
                color=self.theme.text_dim,
                transform=ax.transAxes,
            )
            return
        
        conv = result.convergence_data
        
        # Plot estimate line with color gradient
        for i in range(len(conv.sample_sizes) - 1):
            color = self.theme.get_accuracy_color(conv.errors[i])
            ax.plot(
                conv.sample_sizes[i:i+2],
                conv.estimates[i:i+2],
                color=color,
                linewidth=1.5,
                alpha=0.9,
            )
        
        # True π line
        ax.axhline(
            y=np.pi,
            color=self.theme.theoretical,
            linestyle='-',
            linewidth=2.5,
            label=f'π = {np.pi:.6f}',
            zorder=5,
        )
        
        # Confidence bands
        ax.fill_between(
            conv.sample_sizes,
            np.pi - 0.01,
            np.pi + 0.01,
            alpha=0.15,
            color=self.theme.confidence_band,
            label='±0.01 band',
        )
        ax.fill_between(
            conv.sample_sizes,
            np.pi - 0.001,
            np.pi + 0.001,
            alpha=0.25,
            color=self.theme.confidence_band,
        )
        
        ax.set_xscale('log')
        ax.set_xlabel('Number of Samples (n)', color=self.theme.text)
        ax.set_ylabel('π̂ Estimate', color=self.theme.text)
        ax.set_title('Convergence Analysis', fontweight='bold', color=self.theme.text, pad=12)
        ax.set_ylim(2.9, 3.4)
        
        legend = ax.legend(
            loc='upper right',
            facecolor=self.theme.panel,
            edgecolor=self.theme.grid,
            labelcolor=self.theme.text,
            framealpha=0.95,
        )
    
    def _plot_error_decay(self, ax: plt.Axes, result) -> None:
        """Plot error decay showing O(1/√n) convergence."""
        self._style_axis(ax, show_grid=True)
        
        if result.convergence_data is None:
            ax.text(
                0.5, 0.5, 'Error data not available',
                ha='center', va='center',
                color=self.theme.text_dim,
                transform=ax.transAxes,
            )
            return
        
        conv = result.convergence_data
        
        # Scatter points with color coding
        colors = [self.theme.get_accuracy_color(e) for e in conv.errors]
        ax.scatter(
            conv.sample_sizes,
            conv.errors,
            c=colors,
            s=8,
            alpha=0.7,
            rasterized=True,
        )
        
        # Theoretical O(1/√n) line
        theoretical = 1.6 / np.sqrt(conv.sample_sizes)
        ax.plot(
            conv.sample_sizes,
            theoretical,
            color=self.theme.theoretical,
            linestyle='--',
            linewidth=2,
            label='O(1/√n) theoretical',
        )
        
        # Threshold lines
        ax.axhline(y=0.01, color=self.theme.accuracy_excellent, linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0.001, color=self.theme.accuracy_excellent, linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Samples (n)', color=self.theme.text)
        ax.set_ylabel('|π̂ - π|', color=self.theme.text)
        ax.set_title('Error Decay', fontweight='bold', color=self.theme.text, pad=12)
        
        legend = ax.legend(
            loc='upper right',
            facecolor=self.theme.panel,
            edgecolor=self.theme.grid,
            labelcolor=self.theme.text,
            framealpha=0.95,
            fontsize=9,
        )
    
    def plot_scatter_only(
        self,
        result,
        figsize: Tuple[float, float] = (10, 10),
        point_size: float = 2.0,
    ) -> plt.Figure:
        """Create a standalone scatter plot."""
        self.fig = plt.figure(figsize=figsize, facecolor=self.theme.background)
        ax = self.fig.add_subplot(111)
        self._plot_scatter(ax, result, point_size, alpha=0.6, show_legend=True)
        plt.tight_layout()
        return self.fig
    
    def plot_convergence_only(
        self,
        result,
        figsize: Tuple[float, float] = (12, 6),
    ) -> plt.Figure:
        """Create a standalone convergence plot."""
        self.fig = plt.figure(figsize=figsize, facecolor=self.theme.background)
        ax = self.fig.add_subplot(111)
        self._plot_convergence(ax, result)
        plt.tight_layout()
        return self.fig
    
    def animate(
        self,
        simulator,  # MonteCarloPi
        batch_size: int = 100,
        interval_ms: int = 30,
        figsize: Tuple[float, float] = (14, 7),
    ) -> FuncAnimation:
        """
        Create real-time animation of the Monte Carlo simulation.
        
        Args:
            simulator: MonteCarloPi instance
            batch_size: Points per frame
            interval_ms: Milliseconds between frames
            
        Returns:
            matplotlib FuncAnimation object
        """
        self.fig = plt.figure(figsize=figsize, facecolor=self.theme.background)
        gs = gridspec.GridSpec(
            1, 2,
            width_ratios=[1, 1.2],
            wspace=0.20,
            left=0.06,
            right=0.97,
            top=0.88,
            bottom=0.10,
        )
        
        self.fig.suptitle(
            'Real-Time Monte Carlo Simulation of π',
            fontsize=18,
            fontweight='bold',
            color=self.theme.text,
            y=0.96,
        )
        
        # Left panel: scatter
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(self.theme.panel)
        ax1.set_xlim(-0.02, 1.02)
        ax1.set_ylim(-0.02, 1.02)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x', color=self.theme.text)
        ax1.set_ylabel('y', color=self.theme.text)
        self._style_axis(ax1)
        
        # Quarter circle
        theta = np.linspace(0, np.pi / 2, 200)
        ax1.plot(np.cos(theta), np.sin(theta), color=self.theme.text, linewidth=2, zorder=10)
        ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=self.theme.text_dim, linewidth=1, linestyle='--', alpha=0.5)
        
        scatter_inside = ax1.scatter([], [], c=self.theme.inside, s=2, alpha=0.6)
        scatter_outside = ax1.scatter([], [], c=self.theme.outside, s=2, alpha=0.6)
        title1 = ax1.set_title('n = 0', fontweight='bold', color=self.theme.text, fontsize=13)
        
        # Right panel: convergence
        n_points = simulator.config.n_points
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(self.theme.panel)
        ax2.set_xlim(1, n_points)
        ax2.set_ylim(2.7, 3.6)
        ax2.set_xscale('log')
        ax2.axhline(y=np.pi, color=self.theme.theoretical, linestyle='-', linewidth=2, zorder=5)
        ax2.fill_between([1, n_points], np.pi - 0.01, np.pi + 0.01, alpha=0.15, color=self.theme.confidence_band)
        ax2.set_xlabel('Number of Samples (n)', color=self.theme.text)
        ax2.set_ylabel('π̂ Estimate', color=self.theme.text)
        self._style_axis(ax2, show_grid=True)
        
        conv_line, = ax2.plot([], [], color=self.theme.accent, linewidth=1.5)
        
        pi_text = ax2.text(
            0.02, 0.95, '',
            transform=ax2.transAxes,
            fontsize=18,
            color=self.theme.text,
            fontweight='bold',
            fontfamily='monospace',
            verticalalignment='top',
        )
        error_text = ax2.text(
            0.02, 0.85, '',
            transform=ax2.transAxes,
            fontsize=12,
            color=self.theme.text_dim,
            verticalalignment='top',
        )
        
        # Animation state
        state = {
            'x_all': [],
            'y_all': [],
            'inside_all': [],
            'estimates': [],
            'counts': [],
        }
        
        stream = simulator.stream(batch_size=batch_size)
        
        def init():
            scatter_inside.set_offsets(np.empty((0, 2)))
            scatter_outside.set_offsets(np.empty((0, 2)))
            conv_line.set_data([], [])
            return scatter_inside, scatter_outside, conv_line, pi_text, error_text
        
        def update(frame):
            try:
                n, estimate, x, y, inside = next(stream)
            except StopIteration:
                return scatter_inside, scatter_outside, conv_line, pi_text, error_text
            
            state['x_all'].extend(x)
            state['y_all'].extend(y)
            state['inside_all'].extend(inside)
            state['estimates'].append(estimate)
            state['counts'].append(n)
            
            x_arr = np.array(state['x_all'])
            y_arr = np.array(state['y_all'])
            inside_arr = np.array(state['inside_all'])
            
            scatter_inside.set_offsets(np.c_[x_arr[inside_arr], y_arr[inside_arr]])
            scatter_outside.set_offsets(np.c_[x_arr[~inside_arr], y_arr[~inside_arr]])
            
            conv_line.set_data(state['counts'], state['estimates'])
            
            error = abs(estimate - np.pi)
            color = self.theme.get_accuracy_color(error)
            conv_line.set_color(color)
            
            title1.set_text(f'n = {n:,}')
            pi_text.set_text(f'π̂ = {estimate:.8f}')
            pi_text.set_color(color)
            error_text.set_text(f'Error: {error:.2e} ({100*error/np.pi:.4f}%)')
            
            return scatter_inside, scatter_outside, conv_line, pi_text, error_text
        
        n_frames = (n_points + batch_size - 1) // batch_size
        anim = FuncAnimation(
            self.fig, update,
            init_func=init,
            frames=n_frames,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
        
        return anim
    
    def save(
        self,
        path: Union[str, Path],
        format: Optional[str] = None,
        dpi: Optional[int] = None,
        transparent: bool = False,
    ) -> None:
        """
        Save the current figure to file.
        
        Args:
            path: Output file path
            format: File format (png, pdf, svg). Auto-detected from extension.
            dpi: Resolution for raster formats
            transparent: Use transparent background
        """
        if self.fig is None:
            raise RuntimeError("No figure to save. Call a plot method first.")
        
        path = Path(path)
        
        if format is None:
            format = path.suffix.lstrip('.').lower() or 'png'
        
        if dpi is None:
            dpi = 300 if format in ('png', 'jpg', 'jpeg') else self.dpi
        
        bg_color = 'none' if transparent else self.theme.background
        
        self.fig.savefig(
            path,
            format=format,
            dpi=dpi,
            facecolor=bg_color,
            edgecolor='none',
            bbox_inches='tight',
            pad_inches=0.1,
        )
        
        print(f"Saved figure to '{path}'")
    
    def show(self) -> None:
        """Display the current figure."""
        if self.fig is not None:
            plt.show()
    
    def close(self) -> None:
        """Close the current figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = {}


def quick_plot(result, theme: str = "midnight", save_path: Optional[str] = None) -> None:
    """Quick one-liner for creating and showing a comprehensive plot."""
    viz = PiVisualizer(theme=theme)
    viz.plot_comprehensive(result)
    if save_path:
        viz.save(save_path)
    viz.show()
