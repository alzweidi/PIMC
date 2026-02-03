"""
Rich terminal CLI for Monte Carlo π estimation.

Features:
- Beautiful progress bars with live statistics
- Multiple output modes (simple, detailed, JSON)
- Theme selection
- Benchmark mode
- Interactive and batch modes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from pimc.core.simulation import MonteCarloPi, SamplingMethod, benchmark
from pimc.core.statistics import StatisticalAnalyzer, required_samples_for_precision


console = Console() if RICH_AVAILABLE else None


BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     ██████╗ ██╗███╗   ███╗ ██████╗                                   ║
║     ██╔══██╗██║████╗ ████║██╔════╝                                   ║
║     ██████╔╝██║██╔████╔██║██║                                        ║
║     ██╔═══╝ ██║██║╚██╔╝██║██║                                        ║
║     ██║     ██║██║ ╚═╝ ██║╚██████╗                                   ║
║     ╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝                                   ║
║                                                                      ║
║     Monte Carlo Estimation of π                                      ║
║     High-Performance • Beautiful • Accurate                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def get_accuracy_color(error: float) -> str:
    """Get rich color based on error magnitude."""
    if error < 0.001:
        return "bright_green"
    elif error < 0.01:
        return "green"
    elif error < 0.05:
        return "yellow"
    elif error < 0.1:
        return "orange3"
    else:
        return "red"


def get_grade(error: float) -> tuple[str, str]:
    """Get letter grade and color."""
    if error < 0.001:
        return "S", "bright_magenta"
    elif error < 0.01:
        return "A", "bright_green"
    elif error < 0.05:
        return "B", "green"
    elif error < 0.1:
        return "C", "yellow"
    elif error < 0.2:
        return "D", "orange3"
    return "F", "red"


def create_results_table(result) -> Table:
    """Create a rich table with simulation results."""
    table = Table(
        title="[bold cyan]Simulation Results[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white",
    )
    
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    error_color = get_accuracy_color(result.error)
    grade, grade_color = get_grade(result.error)
    
    table.add_row("Estimated π", f"[bold {error_color}]{result.estimate:.10f}[/]")
    table.add_row("True π", f"{np.pi:.10f}")
    table.add_row("Absolute Error", f"[{error_color}]{result.error:.2e}[/]")
    table.add_row("Error %", f"[{error_color}]{result.error_percent:.6f}%[/]")
    table.add_row("Grade", f"[bold {grade_color}]{grade}[/]")
    table.add_row("", "")
    table.add_row("Total Points", f"{result.n_points:,}")
    table.add_row("Inside Circle", f"[green]{result.n_inside:,}[/]")
    table.add_row("Outside Circle", f"[red]{result.n_points - result.n_inside:,}[/]")
    table.add_row("Ratio", f"{result.ratio:.6f}")
    table.add_row("", "")
    table.add_row("Elapsed Time", f"{result.elapsed_time*1000:.2f} ms")
    table.add_row("Points/Second", f"{result.points_per_second:,.0f}")
    table.add_row("Sampling Method", result.sampling_method.value)
    
    return table


def create_convergence_table(result) -> Optional[Table]:
    """Create a table showing convergence milestones."""
    if result.convergence_data is None:
        return None
    
    table = Table(
        title="[bold cyan]Convergence Milestones[/bold cyan]",
        box=box.ROUNDED,
    )
    
    table.add_column("Samples", justify="right")
    table.add_column("Estimate", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("Status")
    
    conv = result.convergence_data
    indices = [0, len(conv.sample_sizes)//4, len(conv.sample_sizes)//2, 
               3*len(conv.sample_sizes)//4, -1]
    
    for idx in indices:
        n = int(conv.sample_sizes[idx])
        est = conv.estimates[idx]
        err = conv.errors[idx]
        color = get_accuracy_color(err)
        grade, _ = get_grade(err)
        
        table.add_row(
            f"{n:,}",
            f"{est:.8f}",
            f"[{color}]{err:.2e}[/]",
            f"[{color}]{grade}[/]",
        )
    
    return table


def run_simulation_rich(args) -> None:
    """Run simulation with rich output."""
    console.print(BANNER, style="cyan")
    console.print()
    
    method = SamplingMethod(args.method)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task(
            f"[cyan]Simulating with {args.samples:,} points...",
            total=100,
        )
        
        sim = MonteCarloPi(
            n_points=args.samples,
            seed=args.seed,
            method=method,
            n_workers=args.workers,
            track_convergence=True,
        )
        
        # Simple progress update
        for i in range(20):
            progress.update(task, advance=5)
            time.sleep(0.02)
        
        result = sim.run()
        progress.update(task, completed=100)
    
    console.print()
    
    # Results panel
    results_table = create_results_table(result)
    console.print(results_table)
    console.print()
    
    # Convergence panel
    if args.verbose:
        conv_table = create_convergence_table(result)
        if conv_table:
            console.print(conv_table)
            console.print()
    
    # Summary
    error_color = get_accuracy_color(result.error)
    grade, grade_color = get_grade(result.error)
    
    summary = Panel(
        Text.assemble(
            ("π ≈ ", "white"),
            (f"{result.estimate:.10f}", f"bold {error_color}"),
            (" | Error: ", "dim"),
            (f"{result.error:.2e}", error_color),
            (" | Grade: ", "dim"),
            (grade, f"bold {grade_color}"),
        ),
        title="[bold white]Final Result[/]",
        border_style="cyan",
    )
    console.print(summary)
    
    # Save visualization if requested
    if args.output:
        from pimc.visualization import PiVisualizer
        console.print(f"\n[dim]Generating visualization...[/]")
        viz = PiVisualizer(theme=args.theme)
        viz.plot_comprehensive(result)
        viz.save(args.output)
        console.print(f"[green]✓[/] Saved to [cyan]{args.output}[/]")


def run_simulation_simple(args) -> None:
    """Run simulation with simple output (no rich)."""
    print(BANNER)
    print()
    
    method = SamplingMethod(args.method)
    
    print(f"Running simulation with {args.samples:,} points...")
    
    sim = MonteCarloPi(
        n_points=args.samples,
        seed=args.seed,
        method=method,
        n_workers=args.workers,
    )
    
    result = sim.run()
    
    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Estimated π:    {result.estimate:.10f}")
    print(f"True π:         {np.pi:.10f}")
    print(f"Error:          {result.error:.2e} ({result.error_percent:.6f}%)")
    print(f"Grade:          {result.accuracy_grade}")
    print("-" * 50)
    print(f"Points:         {result.n_points:,}")
    print(f"Inside:         {result.n_inside:,}")
    print(f"Time:           {result.elapsed_time*1000:.2f} ms")
    print("=" * 50)
    
    if args.output:
        from pimc.visualization import PiVisualizer
        print(f"\nGenerating visualization...")
        viz = PiVisualizer(theme=args.theme)
        viz.plot_comprehensive(result)
        viz.save(args.output)
        print(f"Saved to {args.output}")


def run_simulation_json(args) -> None:
    """Run simulation with JSON output."""
    method = SamplingMethod(args.method)
    
    sim = MonteCarloPi(
        n_points=args.samples,
        seed=args.seed,
        method=method,
        n_workers=args.workers,
        store_points=False,
    )
    
    result = sim.run()
    
    output = {
        "estimate": result.estimate,
        "true_pi": np.pi,
        "error": result.error,
        "error_percent": result.error_percent,
        "grade": result.accuracy_grade,
        "n_points": result.n_points,
        "n_inside": result.n_inside,
        "ratio": result.ratio,
        "elapsed_time_ms": result.elapsed_time * 1000,
        "points_per_second": result.points_per_second,
        "method": result.sampling_method.value,
        "seed": result.seed,
    }
    
    print(json.dumps(output, indent=2))


def run_benchmark(args) -> None:
    """Run benchmark across sample sizes and methods."""
    if RICH_AVAILABLE:
        console.print("[bold cyan]Running Benchmark...[/]\n")
    else:
        print("Running Benchmark...\n")
    
    sample_sizes = [1_000, 10_000, 100_000, 1_000_000]
    if args.samples:
        sample_sizes.append(args.samples)
    sample_sizes = sorted(set(sample_sizes))
    
    methods = [SamplingMethod.STANDARD, SamplingMethod.ANTITHETIC, SamplingMethod.STRATIFIED]
    
    results = benchmark(sample_sizes, methods, seed=args.seed or 42)
    
    if RICH_AVAILABLE:
        table = Table(title="[bold]Benchmark Results[/]", box=box.ROUNDED)
        table.add_column("Method", style="cyan")
        table.add_column("Samples", justify="right")
        table.add_column("Estimate", justify="right")
        table.add_column("Error", justify="right")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Points/s", justify="right")
        
        for method_name, method_results in results.items():
            for n, data in method_results.items():
                color = get_accuracy_color(data["error"])
                table.add_row(
                    method_name,
                    f"{n:,}",
                    f"{data['estimate']:.8f}",
                    f"[{color}]{data['error']:.2e}[/]",
                    f"{data['time']*1000:.2f}",
                    f"{data['points_per_sec']:,.0f}",
                )
        
        console.print(table)
    else:
        print(f"{'Method':<12} {'Samples':>12} {'Estimate':>14} {'Error':>12} {'Time (ms)':>10}")
        print("-" * 70)
        for method_name, method_results in results.items():
            for n, data in method_results.items():
                print(f"{method_name:<12} {n:>12,} {data['estimate']:>14.8f} {data['error']:>12.2e} {data['time']*1000:>10.2f}")


def run_analyze(args) -> None:
    """Run statistical analysis with multiple trials."""
    if RICH_AVAILABLE:
        console.print(f"[bold cyan]Running {args.trials:,} trials...[/]\n")
    else:
        print(f"Running {args.trials:,} trials...\n")
    
    analyzer = StatisticalAnalyzer(warmup_samples=min(100, args.trials // 10))
    
    if RICH_AVAILABLE:
        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Analyzing...", total=args.trials)
            
            for _ in range(args.trials):
                sim = MonteCarloPi(n_points=args.samples, store_points=False)
                result = sim.run()
                analyzer.add_estimate(result.estimate)
                progress.advance(task)
    else:
        for i in range(args.trials):
            sim = MonteCarloPi(n_points=args.samples, store_points=False)
            result = sim.run()
            analyzer.add_estimate(result.estimate)
            if (i + 1) % (args.trials // 10) == 0:
                print(f"  {i+1}/{args.trials} trials complete")
    
    summary = analyzer.summary()
    ci = analyzer.confidence_interval()
    diagnostics = analyzer.convergence_diagnostics()
    
    if RICH_AVAILABLE:
        table = Table(title="[bold]Statistical Analysis[/]", box=box.ROUNDED)
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        error = abs(summary["mean"] - np.pi)
        color = get_accuracy_color(error)
        
        table.add_row("Trials", f"{args.trials:,}")
        table.add_row("Points per trial", f"{args.samples:,}")
        table.add_row("", "")
        table.add_row("Mean π estimate", f"[bold {color}]{summary['mean']:.10f}[/]")
        table.add_row("Standard deviation", f"{summary['std']:.6f}")
        table.add_row("Standard error", f"{summary['standard_error']:.2e}")
        table.add_row("", "")
        table.add_row("95% CI Lower", f"{ci.lower:.8f}")
        table.add_row("95% CI Upper", f"{ci.upper:.8f}")
        table.add_row("CI Contains π", "[green]Yes[/]" if ci.contains_pi else "[red]No[/]")
        table.add_row("", "")
        table.add_row("Effective Sample Size", f"{diagnostics.effective_sample_size:.1f}")
        table.add_row("Converged", "[green]Yes[/]" if diagnostics.has_converged else "[yellow]Maybe[/]")
        
        console.print(table)
    else:
        print("=" * 50)
        print("STATISTICAL ANALYSIS")
        print("=" * 50)
        print(f"Mean π estimate: {summary['mean']:.10f}")
        print(f"Standard deviation: {summary['std']:.6f}")
        print(f"95% CI: [{ci.lower:.8f}, {ci.upper:.8f}]")
        print(f"CI contains π: {ci.contains_pi}")
        print("=" * 50)


def run_interactive(args) -> None:
    """Run interactive mode with live animation."""
    from pimc.visualization import PiVisualizer
    
    if RICH_AVAILABLE:
        console.print("[bold cyan]Starting interactive visualization...[/]\n")
    else:
        print("Starting interactive visualization...\n")
    
    sim = MonteCarloPi(n_points=args.samples, seed=args.seed)
    viz = PiVisualizer(theme=args.theme)
    anim = viz.animate(sim, batch_size=args.batch_size or 100, interval_ms=30)
    viz.show()


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="pimc",
        description="πMC - High-performance Monte Carlo estimation of π",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pimc run -n 1000000              # Run with 1M points
  pimc run -n 100000 -o plot.png   # Save visualization
  pimc run -n 100000 --method antithetic  # Use variance reduction
  pimc benchmark                    # Compare methods
  pimc analyze -t 1000 -n 10000    # Statistical analysis
  pimc interactive -n 10000        # Live animation
        """,
    )
    
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 2.0.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run Monte Carlo simulation")
    run_parser.add_argument("-n", "--samples", type=int, default=100_000,
                           help="Number of random points (default: 100000)")
    run_parser.add_argument("-s", "--seed", type=int, help="Random seed for reproducibility")
    run_parser.add_argument("-m", "--method", choices=["standard", "antithetic", "stratified", "quasi_random"],
                           default="standard", help="Sampling method (default: standard)")
    run_parser.add_argument("-w", "--workers", type=int, default=1,
                           help="Number of parallel workers (default: 1)")
    run_parser.add_argument("-o", "--output", help="Output file for visualization")
    run_parser.add_argument("-t", "--theme", default="midnight",
                           help="Visualization theme (default: midnight)")
    run_parser.add_argument("--json", action="store_true", help="Output as JSON")
    run_parser.add_argument("--simple", action="store_true", help="Simple output (no colors)")
    run_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark different methods")
    bench_parser.add_argument("-n", "--samples", type=int, help="Additional sample size to test")
    bench_parser.add_argument("-s", "--seed", type=int, help="Random seed")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Statistical analysis with multiple trials")
    analyze_parser.add_argument("-t", "--trials", type=int, default=1000,
                               help="Number of trials (default: 1000)")
    analyze_parser.add_argument("-n", "--samples", type=int, default=10_000,
                               help="Points per trial (default: 10000)")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive live visualization")
    interactive_parser.add_argument("-n", "--samples", type=int, default=10_000,
                                   help="Total points (default: 10000)")
    interactive_parser.add_argument("-s", "--seed", type=int, help="Random seed")
    interactive_parser.add_argument("-t", "--theme", default="midnight", help="Theme")
    interactive_parser.add_argument("-b", "--batch-size", type=int, help="Points per frame")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    try:
        if args.command == "run":
            if args.json:
                run_simulation_json(args)
            elif args.simple or not RICH_AVAILABLE:
                run_simulation_simple(args)
            else:
                run_simulation_rich(args)
        elif args.command == "benchmark":
            run_benchmark(args)
        elif args.command == "analyze":
            run_analyze(args)
        elif args.command == "interactive":
            run_interactive(args)
        
        return 0
    
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Interrupted by user[/]")
        else:
            print("\nInterrupted by user")
        return 130
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Error: {e}[/]")
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
