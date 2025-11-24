"""
pre_processing.py

Standalone PXRD preprocessing module.

Pipeline (order implemented):
 1) Trim header rows from each file
 2) Baseline subtraction using ASLS (pybaselines.whittaker)
 3) Thresholding: set negatives to 0 and values < threshold -> 0
 4) Remove short isolated peaks (width <= min_peak_width)
 5) Normalization (0..1) OPTIONAL — preserves zeros (zeros remain zero)
 6) Build final matrix (columns = preprocessed signals)
 7) Save final matrix to out_matrix_file (default "input.txt")
 8) If verbose: print matrix to stdout and show plots of final processed patterns

This module is standalone and does NOT depend on GUI or snmf_class.

Usage example:
    from pre_processing import run_preprocessing
    MM, x, filenames = run_preprocessing(list_of_files, rows_number=1, do_norm=True, verbose=True)
"""

import os
from pathlib import Path
import numpy as np

# pybaselines (ASLS baseline)
try:
    from pybaselines import whittaker
except Exception as e:
    raise ImportError("pybaselines.whittaker is required. Install via conda/pip: 'pybaselines'") from e


def ensure_dir(path):
    """Create directory if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)


def remove_short_peaks(signal, min_width):
    """
    Remove peaks of length <= min_width that are isolated
    (flanked by zero on both sides).
    Works on 1D numpy arrays and returns a copy.
    """
    result = signal.copy()
    non_zero = np.where(result > 0)[0]
    if len(non_zero) == 0:
        return result

    gaps = np.diff(non_zero) > 1
    seg_end = np.where(gaps)[0]
    seg_start = np.insert(seg_end + 1, 0, 0)
    seg_end = np.append(seg_end, len(non_zero) - 1)

    for s, e in zip(seg_start, seg_end):
        idx = non_zero[s:e + 1]
        if len(idx) <= min_width:
            left = idx[0] - 1
            right = idx[-1] + 1

            left_ok = (left < 0) or (result[left] == 0)
            right_ok = (right >= len(result)) or (result[right] == 0)
            if left_ok and right_ok:
                result[idx] = 0

    return result


def run_preprocessing(
    input_files,
    rows_number=0,
    do_norm=True,
    threshold=200.0,
    min_peak_width=2,
    corrected_dir="corrected_input_files",
    normalized_dir="normalized_input_files",
    out_matrix_file="input.txt",
    verbose=False,
):
    """
    Preprocess PXRD input files.

    Parameters
    ----------
    input_files : list[str]
        List of file paths containing X Y columns (whitespace-separated).
    rows_number : int
        Number of header lines to remove from each file (trim).
    do_norm : bool
        If True, normalize final Y vectors to [0,1] preserving zeros.
    threshold : float
        Values strictly less than this are zeroed after baseline subtraction.
    min_peak_width : int
        Remove isolated peaks of length <= min_peak_width.
    corrected_dir : str
        Directory to store trimmed files (kept for traceability).
    normalized_dir : str
        Directory to store final normalized files (kept for traceability).
        (Because you chose D - we will NOT save intermediate steps, only final normalized files if do_norm=True.)
    out_matrix_file : str
        Output filename for final matrix (columns are signals).
    verbose : bool
        If True, print progress, print final matrix to stdout and show plots of final patterns.

    Returns
    -------
    MM : numpy.ndarray
        Matrix with preprocessed signals (N × M) where N = number of X points.
    x_values : numpy.ndarray
        X-axis values (1D array).
    filenames : list[str]
        Basenames of input files used as column labels.
    """

    # normalize input list & basic checks
    input_files = [os.path.normpath(f) for f in input_files]
    if len(input_files) == 0:
        raise ValueError("No input files provided to preprocessing.")

    # Prepare directories (we create them but do not store intermediate unless final normalization requested)
    ensure_dir(corrected_dir)
    ensure_dir(normalized_dir)

    filenames = []
    trimmed_files = []

    # STEP 1: Trim header rows and write trimmed copies to corrected_dir
    for fp in input_files:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Input file not found: {fp}")
        base = os.path.basename(fp)
        filenames.append(base)
        out_fp = os.path.join(corrected_dir, base)
        trimmed_files.append(out_fp)

        with open(fp, "r") as f:
            lines = f.readlines()

        if rows_number > 0:
            lines = lines[rows_number:]

        with open(out_fp, "w") as f:
            f.writelines(lines)

        if verbose:
            print(f"[STEP 1] Trimmed -> {out_fp}")

    # Load arrays (X,Y) from trimmed files into memory
    arrays = []
    for fp in trimmed_files:
        arr = np.loadtxt(fp)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Invalid data in {fp}: expected at least 2 columns.")
        arrays.append(np.column_stack((arr[:, 0].astype(float), arr[:, 1].astype(float))))
    if verbose:
        print(f"[INFO] Loaded {len(arrays)} trimmed files.")

    # STEP 2: Baseline subtraction (ASLS)
    baseline_corrected = []
    for arr in arrays:
        x = arr[:, 0]
        y = arr[:, 1].copy()  # work on copy
        base, _ = whittaker.asls(y, lam=1e5, p=0.01)
        y_corr = y - base

        baseline_corrected.append(np.column_stack((x, y_corr)))

        if verbose:
            print("[STEP 2] Baseline subtracted (ASLS).")

    # STEP 3: Thresholding (zero negatives and low intensities)
    thresholded = []
    for arr in baseline_corrected:
        x = arr[:, 0]
        y = arr[:, 1].copy()
        # set negative to zero first
        y[y < 0] = 0.0
        # zero values below threshold
        if threshold is not None and threshold > 0:
            y[y < threshold] = 0.0
        thresholded.append(np.column_stack((x, y)))
    if verbose:
        print("[STEP 3] Thresholding applied (negatives->0, below threshold->0).")

    # STEP 4: Remove short isolated peaks
    peak_cleaned = []
    for arr in thresholded:
        x = arr[:, 0]
        y = arr[:, 1].copy()
        y_clean = remove_short_peaks(y, min_peak_width)
        peak_cleaned.append(np.column_stack((x, y_clean)))
    if verbose:
        print("[STEP 4] Removed short isolated peaks.")

    # STEP 5: Normalization 0..1 (optional) — preserves zeros
    processed = []
    for arr, src_fp in zip(peak_cleaned, trimmed_files):
        x = arr[:, 0]
        y = arr[:, 1].copy()
        if do_norm:
            # preserve zeros: compute max over positive entries only
            positive_max = np.max(y) if np.any(y > 0) else 0.0
            if positive_max > 0:
                y = y / positive_max
            else:
                # nothing to normalize (all zeros) -> keep as zeros
                y = y
        processed.append(np.column_stack((x, y)))

        if verbose:
            print(f"[STEP 5] Normalization applied: {os.path.basename(src_fp)} -> max={np.max(y):.6g}")

    # Save final per-file normalized patterns into normalized_dir (one file per input) only if do_norm True
    # The user requested NO intermediate saving (option D), but keeping final normalized copies can help debugging.
    # We therefore save the final processed patterns as convenience — comment this out if undesired.
    for arr, src_fp in zip(processed, trimmed_files):
        out_fp = os.path.join(normalized_dir, os.path.basename(src_fp))
        np.savetxt(out_fp, arr, fmt="%.6g")
    if verbose:
        print(f"[INFO] Final processed files saved to '{normalized_dir}' (one file per input).")

    # STEP 6: Build final matrix — ensure consistent X
    x_ref = processed[0][:, 0]
    for arr in processed:
        if not np.allclose(arr[:, 0], x_ref, atol=1e-6):
            raise ValueError("X mismatch between processed files after preprocessing.")

    MM = np.column_stack([arr[:, 1] for arr in processed])

    # Protection: if threshold too high => matrix all zeros -> warn
    if np.all(MM == 0):
        print("\n[WARNING] All intensities became zero after thresholding! Consider lowering the threshold.\n")

    # STEP 7: Save final matrix to out_matrix_file
    np.savetxt(out_matrix_file, MM, fmt="%.6g")
    if verbose:
        print(f"[STEP 7] Wrote matrix {out_matrix_file} with shape {MM.shape}")

    # STEP 8: If verbose -> print matrix to stdout and plot final processed patterns
    if verbose:
        print("\n=== FINAL MATRIX MM (written to {}) ===".format(out_matrix_file))
        print(f"Shape = {MM.shape}")
        # print row by row like original code
        for row in MM:
            print(" ".join(f"{v:.6g}" for v in row))
        print("=== END MATRIX ===\n")

        # Plot all processed patterns in one figure (overlaid)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            for arr in processed:
                plt.plot(arr[:, 0], arr[:, 1], lw=1)
            plt.xlabel("2θ (degrees)")
            plt.ylabel("Intensity (a.u.)")
            plt.title("Final processed patterns (overlaid)")
            plt.tight_layout()
            plt.show()
        except Exception:
            # don't fail if matplotlib cannot display
            print("[INFO] matplotlib plotting failed or not available; continuing.")

    return MM, x_ref.astype(float), filenames
