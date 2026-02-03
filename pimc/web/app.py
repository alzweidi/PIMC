"""
Interactive web dashboard for Monte Carlo π estimation.

A beautiful, real-time web interface built with Flask and modern JavaScript.
Features WebSocket for live simulation updates and D3.js visualizations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np

from pimc.core.simulation import MonteCarloPi, SamplingMethod
from pimc.core.statistics import StatisticalAnalyzer


app = Flask(__name__, 
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))
CORS(app)


@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run a Monte Carlo simulation and return results."""
    data = request.get_json() or {}
    
    n_points = data.get("n_points", 10_000)
    method = data.get("method", "standard")
    seed = data.get("seed")
    
    n_points = min(max(100, int(n_points)), 10_000_000)
    
    try:
        sampling_method = SamplingMethod(method)
    except ValueError:
        sampling_method = SamplingMethod.STANDARD
    
    sim = MonteCarloPi(
        n_points=n_points,
        seed=seed,
        method=sampling_method,
        store_points=n_points <= 100_000,
    )
    
    result = sim.run()
    
    response = {
        "success": True,
        "estimate": float(result.estimate),
        "true_pi": float(np.pi),
        "error": float(result.error),
        "error_percent": float(result.error_percent),
        "grade": result.accuracy_grade,
        "n_points": result.n_points,
        "n_inside": result.n_inside,
        "ratio": float(result.ratio),
        "elapsed_ms": float(result.elapsed_time * 1000),
        "points_per_second": float(result.points_per_second),
        "method": result.sampling_method.value,
    }
    
    if result.convergence_data is not None:
        response["convergence"] = {
            "sample_sizes": result.convergence_data.sample_sizes.tolist(),
            "estimates": result.convergence_data.estimates.tolist(),
            "errors": result.convergence_data.errors.tolist(),
        }
    
    if len(result.x_coords) > 0 and len(result.x_coords) <= 100_000:
        sample_size = min(20000, len(result.x_coords))
        indices = np.random.choice(len(result.x_coords), sample_size, replace=False)
        response["points"] = {
            "x": result.x_coords[indices].tolist(),
            "y": result.y_coords[indices].tolist(),
            "inside": result.inside_mask[indices].tolist(),
        }
    
    return jsonify(response)


@app.route("/api/stream", methods=["POST"])
def stream_simulation():
    """Stream simulation results for real-time visualization."""
    data = request.get_json() or {}
    
    n_points = min(max(100, int(data.get("n_points", 5000))), 50_000)
    batch_size = min(max(10, int(data.get("batch_size", 100))), 1000)
    
    sim = MonteCarloPi(n_points=n_points)
    
    batches = []
    for current_n, estimate, x, y, inside in sim.stream(batch_size=batch_size):
        batches.append({
            "n": current_n,
            "estimate": float(estimate),
            "error": float(abs(estimate - np.pi)),
            "x": x.tolist(),
            "y": y.tolist(),
            "inside": inside.tolist(),
        })
    
    return jsonify({
        "success": True,
        "batches": batches,
        "final_estimate": float(batches[-1]["estimate"]) if batches else 0,
    })


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    """Run benchmark across different methods."""
    data = request.get_json() or {}
    
    sample_sizes = data.get("sample_sizes", [1000, 10000, 100000])
    sample_sizes = [min(max(100, int(n)), 1_000_000) for n in sample_sizes]
    
    results = {}
    
    for method in ["standard", "antithetic", "stratified"]:
        results[method] = []
        for n in sample_sizes:
            sim = MonteCarloPi(n_points=n, method=method, store_points=False)
            result = sim.run()
            results[method].append({
                "n": n,
                "estimate": float(result.estimate),
                "error": float(result.error),
                "time_ms": float(result.elapsed_time * 1000),
            })
    
    return jsonify({"success": True, "results": results})


@app.route("/api/analyze", methods=["POST"])
def run_analysis():
    """Run statistical analysis with multiple trials."""
    data = request.get_json() or {}
    
    n_trials = min(max(10, int(data.get("n_trials", 100))), 10000)
    n_points = min(max(100, int(data.get("n_points", 10000))), 100_000)
    
    analyzer = StatisticalAnalyzer(warmup_samples=min(50, n_trials // 10))
    
    for _ in range(n_trials):
        sim = MonteCarloPi(n_points=n_points, store_points=False)
        result = sim.run()
        analyzer.add_estimate(result.estimate)
    
    summary = analyzer.summary()
    ci = analyzer.confidence_interval()
    
    return jsonify({
        "success": True,
        "n_trials": n_trials,
        "n_points_per_trial": n_points,
        "mean": float(summary["mean"]),
        "std": float(summary["std"]),
        "standard_error": float(summary["standard_error"]),
        "ci_lower": float(ci.lower),
        "ci_upper": float(ci.upper),
        "ci_contains_pi": bool(ci.contains_pi),
        "error_from_pi": float(summary["error_from_pi"]),
        "has_converged": bool(summary["has_converged"]),
        "running_means": analyzer.running_means.tolist()[-100:],  # Last 100
    })


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Run the Flask development server."""
    print(f"\npiMC Web Dashboard")
    print(f"   Running at http://{host}:{port}")
    print(f"   Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
