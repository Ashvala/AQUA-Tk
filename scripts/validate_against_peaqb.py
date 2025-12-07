#!/usr/bin/env python3
"""
Validation script to compare Python PEAQ implementation against peaqb-fast (C).

This script runs both implementations on the same audio files and compares:
1. Final ODG/DI scores
2. All 11 MOV (Model Output Variable) values
3. Frame-by-frame comparisons (optional)
"""

import subprocess
import sys
import os
import re
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aquatk.metrics.PEAQ.peaq import PEAQ


@dataclass
class PEAQBOutput:
    """Parsed output from peaqb-fast C implementation"""
    frame: int
    BandwidthRefb: float
    BandwidthTestb: float
    TotalNMRb: float
    WinModDiff1b: float
    ADBb: float
    EHSb: float
    AvgModDiff1b: float
    AvgModDiff2b: float
    RmsNoiseLoudb: float
    MFPDb: float
    RelDistFramesb: float
    DI: float
    ODG: float


def run_peaqb_fast(ref_file: str, test_file: str, peaqb_path: str) -> List[PEAQBOutput]:
    """
    Run peaqb-fast C implementation and parse output.

    Args:
        ref_file: Path to reference WAV file
        test_file: Path to test WAV file
        peaqb_path: Path to peaqb binary

    Returns:
        List of PEAQBOutput for each frame
    """
    cmd = [peaqb_path, "-r", ref_file, "-t", test_file, "-c"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(peaqb_path)
        )
    except subprocess.TimeoutExpired:
        print("ERROR: peaqb-fast timed out")
        return []
    except FileNotFoundError:
        print(f"ERROR: peaqb binary not found at {peaqb_path}")
        return []

    if result.returncode != 0:
        print(f"ERROR: peaqb-fast failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        return []

    # Parse CSV output
    outputs = []
    lines = result.stdout.strip().split('\n')

    # Find the header line and data lines
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("frame,"):
            header_idx = i
            break

    if header_idx is None:
        print("ERROR: Could not find CSV header in peaqb output")
        print("Output was:")
        print(result.stdout[:1000])
        return []

    # Parse data lines
    for line in lines[header_idx + 1:]:
        if not line.strip() or line.startswith("frame,"):
            continue

        parts = line.split(',')
        if len(parts) < 14:
            continue

        try:
            outputs.append(PEAQBOutput(
                frame=int(parts[0]),
                BandwidthRefb=float(parts[1]),
                BandwidthTestb=float(parts[2]),
                TotalNMRb=float(parts[3]),
                WinModDiff1b=float(parts[4]),
                ADBb=float(parts[5]),
                EHSb=float(parts[6]),
                AvgModDiff1b=float(parts[7]),
                AvgModDiff2b=float(parts[8]),
                RmsNoiseLoudb=float(parts[9]),
                MFPDb=float(parts[10]),
                RelDistFramesb=float(parts[11]),
                DI=float(parts[12]),
                ODG=float(parts[13])
            ))
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse line: {line} ({e})")
            continue

    return outputs


def run_python_peaq(ref_file: str, test_file: str) -> Tuple[float, float, Dict[str, float], float]:
    """
    Run Python PEAQ implementation.

    Returns:
        Tuple of (ODG, DI, MOVs dict, elapsed_time)
    """
    peaq = PEAQ(version="basic")

    start_time = time.time()
    result = peaq.analyze_files(ref_file, test_file, progress_bar=False)
    elapsed = time.time() - start_time

    return result.odg, result.di, result.mov, elapsed


def compare_results(
    c_outputs: List[PEAQBOutput],
    py_odg: float,
    py_di: float,
    py_movs: Dict[str, float]
) -> Dict[str, float]:
    """
    Compare C and Python implementation results.

    Returns:
        Dictionary of differences for each metric
    """
    if not c_outputs:
        return {"error": "No C outputs to compare"}

    # Get final frame from C implementation
    final_c = c_outputs[-1]

    differences = {
        "ODG_diff": abs(final_c.ODG - py_odg),
        "ODG_c": final_c.ODG,
        "ODG_py": py_odg,
        "DI_diff": abs(final_c.DI - py_di),
        "DI_c": final_c.DI,
        "DI_py": py_di,
    }

    # Compare MOVs
    mov_mapping = {
        "BandwidthRefb": "BandwidthRefb",
        "BandwidthTestb": "BandwidthTestb",
        "TotalNMRb": "TotalNMRb",
        "WinModDiff1b": "WinModDiff1b",
        "ADBb": "ADBb",
        "EHSb": "EHSb",
        "AvgModDiff1b": "AvgModDiff1b",
        "AvgModDiff2b": "AvgModDiff2b",
        "RmsNoiseLoudb": "RmsNoiseLoudb",
        "MFPDb": "MFPDb",
        "RelDistFramesb": "RelDistFramesb",
    }

    for c_name, py_name in mov_mapping.items():
        c_val = getattr(final_c, c_name)
        py_val = py_movs.get(py_name, 0)
        differences[f"{c_name}_diff"] = abs(c_val - py_val)
        differences[f"{c_name}_c"] = c_val
        differences[f"{c_name}_py"] = py_val

    return differences


def print_comparison_table(differences: Dict[str, float]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Extract metrics
    metrics = ["ODG", "DI", "BandwidthRefb", "BandwidthTestb", "TotalNMRb",
               "WinModDiff1b", "ADBb", "EHSb", "AvgModDiff1b", "AvgModDiff2b",
               "RmsNoiseLoudb", "MFPDb", "RelDistFramesb"]

    print(f"\n{'Metric':<20} {'C (peaqb)':<15} {'Python':<15} {'Diff':<15} {'Status':<10}")
    print("-" * 75)

    for metric in metrics:
        c_val = differences.get(f"{metric}_c", "N/A")
        py_val = differences.get(f"{metric}_py", "N/A")
        diff = differences.get(f"{metric}_diff", float('inf'))

        # Determine pass/fail based on metric type
        if metric in ["ODG", "DI"]:
            threshold = 0.1  # Stricter threshold for main outputs
        else:
            threshold = 1.0  # More lenient for MOVs

        status = "PASS" if diff < threshold else "FAIL"

        if isinstance(c_val, float):
            c_str = f"{c_val:.6f}"
        else:
            c_str = str(c_val)

        if isinstance(py_val, float):
            py_str = f"{py_val:.6f}"
        else:
            py_str = str(py_val)

        if isinstance(diff, float) and diff != float('inf'):
            diff_str = f"{diff:.6f}"
        else:
            diff_str = str(diff)

        print(f"{metric:<20} {c_str:<15} {py_str:<15} {diff_str:<15} {status:<10}")

    print("-" * 75)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate Python PEAQ against peaqb-fast C implementation"
    )
    parser.add_argument(
        "--ref", "-r",
        required=True,
        help="Path to reference WAV file"
    )
    parser.add_argument(
        "--test", "-t",
        required=True,
        help="Path to test WAV file"
    )
    parser.add_argument(
        "--peaqb",
        default="/Users/ashvala/Projects/gatech/PhD/peaqb-fast/src/peaqb",
        help="Path to peaqb binary"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output including frame-by-frame data"
    )

    args = parser.parse_args()

    # Verify files exist
    if not os.path.exists(args.ref):
        print(f"ERROR: Reference file not found: {args.ref}")
        sys.exit(1)
    if not os.path.exists(args.test):
        print(f"ERROR: Test file not found: {args.test}")
        sys.exit(1)
    if not os.path.exists(args.peaqb):
        print(f"ERROR: peaqb binary not found: {args.peaqb}")
        sys.exit(1)

    print(f"Reference: {args.ref}")
    print(f"Test: {args.test}")
    print(f"peaqb binary: {args.peaqb}")

    # Run C implementation
    print("\nRunning peaqb-fast (C)...")
    c_start = time.time()
    c_outputs = run_peaqb_fast(args.ref, args.test, args.peaqb)
    c_elapsed = time.time() - c_start
    print(f"  Completed in {c_elapsed:.3f}s ({len(c_outputs)} frames)")

    if c_outputs and args.verbose:
        print(f"  Final frame ODG: {c_outputs[-1].ODG:.6f}")
        print(f"  Final frame DI: {c_outputs[-1].DI:.6f}")

    # Run Python implementation
    print("\nRunning Python PEAQ...")
    try:
        py_odg, py_di, py_movs, py_elapsed = run_python_peaq(args.ref, args.test)
        print(f"  Completed in {py_elapsed:.3f}s")
        print(f"  ODG: {py_odg:.6f}")
        print(f"  DI: {py_di:.6f}")
    except Exception as e:
        print(f"ERROR: Python PEAQ failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Compare results
    if c_outputs:
        differences = compare_results(c_outputs, py_odg, py_di, py_movs)
        print_comparison_table(differences)

        # Performance comparison
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        print(f"C implementation:      {c_elapsed:.3f}s")
        print(f"Python implementation: {py_elapsed:.3f}s")
        print(f"Speedup factor:        {py_elapsed/c_elapsed:.1f}x slower")

        # Summary
        odg_diff = differences.get("ODG_diff", float('inf'))
        di_diff = differences.get("DI_diff", float('inf'))

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        if odg_diff < 0.1 and di_diff < 0.1:
            print("PASS: ODG and DI within acceptable tolerance (< 0.1)")
        else:
            print("FAIL: ODG and/or DI exceed acceptable tolerance")
            print(f"  ODG difference: {odg_diff:.6f}")
            print(f"  DI difference: {di_diff:.6f}")
    else:
        print("\nWARNING: Could not compare - no C outputs available")


if __name__ == "__main__":
    main()
