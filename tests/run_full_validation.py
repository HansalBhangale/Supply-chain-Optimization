"""
Full Validation Suite Runner.

Runs all benchmarks and validation tests, generates comprehensive report.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_scalability import run_all_benchmarks, BENCHMARK_CONFIGS
from validate_accuracy import run_full_validation, run_consistency_analysis


def generate_report(
    scalability_results: list,
    validation_results: dict,
    output_dir: str = "results"
) -> str:
    """Generate markdown report from test results."""
    
    output_path = Path(output_dir)
    report_lines = []
    
    # Header
    report_lines.append("# Supply Chain Optimization - Validation Report\n")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Executive Summary
    report_lines.append("## Executive Summary\n")
    
    # Scalability summary
    if scalability_results:
        total_configs = len(scalability_results)
        successful = sum(1 for r in scalability_results if r.objective_value < 1e10)
        report_lines.append(f"- **Scalability Tests:** {successful}/{total_configs} passed\n")
    
    # Accuracy summary
    if validation_results and validation_results.get("accuracy_tests"):
        tests = validation_results["accuracy_tests"]
        feasible_count = sum(1 for t in tests if t.get("is_feasible", False))
        total = len(tests)
        
        gaps = [t["optimality_gap_percent"] for t in tests 
                if t.get("optimality_gap_percent") is not None]
        avg_gap = sum(gaps) / len(gaps) if gaps else None
        
        report_lines.append(f"- **Accuracy Tests:** {feasible_count}/{total} feasible\n")
        if avg_gap is not None:
            report_lines.append(f"- **Average Optimality Gap:** {avg_gap:.2f}%\n")
    
    # Consistency
    if validation_results and validation_results.get("consistency_analysis"):
        cv = validation_results["consistency_analysis"]["cv_percent"]
        report_lines.append(f"- **Solution Consistency (CV):** {cv:.1f}%\n")
    
    report_lines.append("\n---\n\n")
    
    # Scalability Results
    report_lines.append("## Scalability Benchmark Results\n\n")
    report_lines.append("| Config | QUBO Vars | Time (s) | Memory (MB) | Objective | Status |\n")
    report_lines.append("|--------|-----------|----------|-------------|-----------|--------|\n")
    
    for r in scalability_results:
        status = "✓" if r.objective_value < 1e10 else "✗"
        obj = f"{r.objective_value:.0f}" if r.objective_value < 1e10 else "FAILED"
        report_lines.append(
            f"| {r.config_name} | {r.qubo_variables} | {r.solve_time_seconds:.2f} | "
            f"{r.memory_peak_mb:.1f} | {obj} | {status} |\n"
        )
    
    report_lines.append("\n---\n\n")
    
    # Accuracy Results
    report_lines.append("## Accuracy Validation Results\n\n")
    
    if validation_results and validation_results.get("accuracy_tests"):
        report_lines.append("| Test | Solver | Gap (%) | Feasible | Violations |\n")
        report_lines.append("|------|--------|---------|----------|------------|\n")
        
        for t in validation_results["accuracy_tests"]:
            gap = f"{t['optimality_gap_percent']:.2f}" if t.get('optimality_gap_percent') is not None else "N/A"
            feasible = "✓" if t.get('is_feasible') else "✗"
            violations = sum(t.get('constraint_violations', {}).values())
            report_lines.append(
                f"| {t['test_name']} | {t['solver_type']} | {gap} | {feasible} | {violations} |\n"
            )
    
    report_lines.append("\n---\n\n")
    
    # Consistency Analysis
    report_lines.append("## Consistency Analysis\n\n")
    
    if validation_results and validation_results.get("consistency_analysis"):
        ca = validation_results["consistency_analysis"]
        report_lines.append(f"- **Number of runs:** {ca['num_runs']}\n")
        report_lines.append(f"- **Mean objective:** {ca['mean_objective']:.0f}\n")
        report_lines.append(f"- **Std deviation:** {ca['std_objective']:.0f}\n")
        report_lines.append(f"- **CV (Coefficient of Variation):** {ca['cv_percent']:.1f}%\n")
        report_lines.append(f"- **Mean solve time:** {ca['mean_solve_time']:.2f}s\n")
    
    report_lines.append("\n---\n\n")
    
    # Quantum Readiness
    report_lines.append("## Quantum Hardware Readiness\n\n")
    report_lines.append("| Problem Size | QUBO Variables | Quantum-Ready? | Notes |\n")
    report_lines.append("|--------------|----------------|----------------|-------|\n")
    report_lines.append("| Tiny (3S, 2W, 5C) | ~16 | ✓ Yes | Suitable for current hardware |\n")
    report_lines.append("| Small (5S, 3W, 10C) | ~45 | ✓ Yes | Within IBM Heron range |\n")
    report_lines.append("| Medium (10S, 5W, 20C) | ~150 | ⚠️ Maybe | Needs 127+ qubit system |\n")
    report_lines.append("| Large (20S, 10W, 50C) | ~700 | ✗ No | Exceeds current hardware |\n")
    
    # Save report
    report_content = "".join(report_lines)
    report_file = output_path / "validation_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"\nReport saved to: {report_file}")
    return report_content


def run_all_tests(
    include_quantum: bool = False,
    output_dir: str = "results"
) -> None:
    """
    Run complete test suite.
    
    Args:
        include_quantum: If True, attempt IBM Quantum tests (requires valid token)
        output_dir: Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*70)
    print("SUPPLY CHAIN OPTIMIZATION - FULL VALIDATION SUITE")
    print("="*70)
    
    start_time = time.time()
    
    # 1. Scalability benchmarks
    print("\n" + "="*70)
    print("PHASE 1: SCALABILITY BENCHMARKS")
    print("="*70)
    
    solver_types = ["classical"]
    if include_quantum:
        solver_types.append("qaoa")
    
    scalability_results = run_all_benchmarks(
        configs=BENCHMARK_CONFIGS,
        solver_types=solver_types,
        output_dir=output_dir
    )
    
    # 2. Accuracy validation
    print("\n" + "="*70)
    print("PHASE 2: ACCURACY VALIDATION")
    print("="*70)
    
    validation_results = run_full_validation(output_dir=output_dir)
    
    # 3. Generate report
    print("\n" + "="*70)
    print("PHASE 3: GENERATING REPORT")
    print("="*70)
    
    generate_report(scalability_results, validation_results, output_dir)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"VALIDATION COMPLETE in {total_time:.1f} seconds")
    print(f"Results saved to: {output_path.absolute()}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full validation suite")
    parser.add_argument("--include-quantum", action="store_true",
                        help="Include QAOA simulator tests")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    run_all_tests(include_quantum=args.include_quantum, output_dir=args.output)
