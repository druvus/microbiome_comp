#!/usr/bin/env python3
"""
microbiome_pipeline.py: Python module for microbial genomics compositional workflows.
Includes:
 - filtering taxa by read counts and zero proportion
 - cmult_repl zero-replacement (Bayesian-multiplicative)
 - pivot log-ratio (PLR) transformation
"""
import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import gmean
from tqdm import tqdm
from numba import njit, prange


def filter_taxa(
    df: pd.DataFrame,
    min_reads: int,
    nonzeros_prop: float
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filter taxa based on minimum reads and zero-inflation.

    Args:
        df: DataFrame with taxa as rows and samples as columns
        min_reads: Minimum number of reads in the maximum sample
        nonzeros_prop: Minimum proportion of non-zero samples

    Returns:
        df_keep: Taxa meeting filtering criteria
        removed_sums: Sum of reads of removed taxa per sample
    """
    # Calculate number of non-zero samples required
    nonzeros_count = int(round(df.shape[1] * nonzeros_prop))

    # Taxa with at least min_reads in their max sample
    df_keep = df.loc[df.max(axis=1) >= min_reads]

    # Taxa with at least nonzeros_count non-zero values
    df_keep = df_keep.loc[(df_keep > 0).sum(axis=1) >= nonzeros_count]

    # Calculate removed taxa
    df_removed = df.loc[~df.index.isin(df_keep.index)]
    removed_sums = df_removed.sum(axis=0)

    logging.info(f"Filtered taxa: kept {len(df_keep)} of {len(df)} taxa")
    return df_keep, removed_sums


def add_removed_reads(
    df_filtered: pd.DataFrame,
    removed_sums: pd.Series,
    label: str = 'Non-species'
) -> pd.DataFrame:
    """
    Add removed reads back into a new taxon labeled `label`.

    Args:
        df_filtered: Filtered DataFrame
        removed_sums: Sum of reads of removed taxa per sample
        label: Label for the new taxon

    Returns:
        DataFrame with an additional row containing the removed reads
    """
    df_out = df_filtered.copy()

    if label in df_out.index:
        logging.warning(f"Label '{label}' already exists in the data. Adding removed reads to it.")
        df_out.loc[label] += removed_sums
    else:
        df_out.loc[label] = removed_sums

    return df_out


@njit(parallel=True)
def _impute_and_adjust(
    X2: np.ndarray,
    repl: np.ndarray,
    col_mins: np.ndarray,
    frac: float,
    adjust: bool,
    mask: np.ndarray
) -> np.ndarray:
    """
    JIT‐compiled helper to do the “close → replace → adjust → renormalize” loop in parallel.
    """
    N, D = X2.shape
    for i in prange(N):
        # 1) fill & optionally adjust
        sum_repl = 0.0
        if adjust:
            for j in range(D):
                if mask[i, j]:
                    val = repl[i, j]
                    if val > col_mins[j]:
                        val = frac * col_mins[j]
                    X2[i, j] = val
                    sum_repl += val
        else:
            for j in range(D):
                if mask[i, j]:
                    val = repl[i, j]
                    X2[i, j] = val
                    sum_repl += val

        # 2) renormalize non-missing parts
        f = 1.0 - sum_repl
        for j in range(D):
            if not mask[i, j]:
                X2[i, j] = f * X2[i, j]

    return X2

def cmult_repl(
    X: np.ndarray,
    label: float = 0.0,
    method: str = "GBM",
    output: str = "prop",
    frac: float = 0.65,
    threshold: float = 0.5,
    adjust: bool = True,
    t: Optional[np.ndarray] = None,
    s: Optional[np.ndarray] = None,
    z_warning: float = 0.8,
    z_delete: bool = True,
) -> np.ndarray:
    """
    Bayesian-multiplicative replacement of count zeros with optimized inner loops.
    """
    # ─── 1) INPUT VALIDATION ─────────────────────────────────
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array (samples × features)")
    if X.shape[0] == 1:
        raise ValueError("Input X must have at least two rows (samples)")
    if X.dtype.kind not in ('i', 'u', 'f'):
        raise ValueError("Input matrix X must be numeric")

    X = X.astype(float)
    if np.any(X < 0):
        raise ValueError("X contains negative values")
    if not np.any(X == label):
        raise ValueError(f"Label {label} was not found in the data set")
    if label != 0.0 and np.any(X == 0):
        raise ValueError("Zero values not labelled as count zeros were found in X")

    # mark zeros as NaN
    X[X == label] = np.nan
    N, D = X.shape
    n = np.nansum(X, axis=1)
    method = method.upper()
    output = output.lower()

    # ─── 2) ZERO‐THRESHOLD HANDLING ─────────────────────────
    # columns
    col_frac = np.isnan(X).sum(axis=0) / N
    drop_cols = np.where(col_frac > z_warning)[0]
    if drop_cols.size:
        if z_delete:
            if drop_cols.size > D - 2:
                raise ValueError(f"Almost all columns contain >{z_warning*100}% zeros.")
            logging.warning(f"Dropping {drop_cols.size} columns with >{z_warning*100}% zeros")
            X = np.delete(X, drop_cols, axis=1)
            D = X.shape[1]
        else:
            raise ValueError(f"Columns {drop_cols.tolist()} exceed zero threshold.")

    # rows
    row_frac = np.isnan(X).sum(axis=1) / D
    drop_rows = np.where(row_frac > z_warning)[0]
    if drop_rows.size:
        if z_delete:
            if drop_rows.size > N - 2:
                raise ValueError(f"Almost all rows contain >{z_warning*100}% zeros.")
            logging.warning(f"Dropping {drop_rows.size} rows with >{z_warning*100}% zeros")
            X = np.delete(X, drop_rows, axis=0)
            N = X.shape[0]
            n = np.nansum(X, axis=1)
        else:
            raise ValueError(f"Rows {drop_rows.tolist()} exceed zero threshold.")

    # ─── 3) VECTORIZED α / t COMPUTATION ────────────────────
    if method != "CZM":
        if method == "USER":
            if t is None or s is None:
                raise ValueError("User method requires t and s hyper-parameters.")
            t_mat = t.copy()
            s_vec = s.copy()
        else:
            # vectorized alpha = col_sums - X
            col_sums = np.nansum(X, axis=0)                  # shape (D,)
            alpha    = col_sums[None, :] - np.nan_to_num(X)  # shape (N, D)
            alpha_sum = np.sum(alpha, axis=1, keepdims=True) # shape (N, 1)
            alpha_sum[alpha_sum == 0] = np.finfo(float).eps
            t_mat    = alpha / alpha_sum                     # shape (N, D)

            # compute s_vec
            if method == "GBM":
                if np.any(t_mat == 0):
                    raise ValueError("GBM method: insufficient data to compute t hyper-parameter.")
                s_vec = 1.0 / np.exp(np.nanmean(np.log(t_mat), axis=1))
            elif method == "SQ":
                s_vec = np.sqrt(n)
            elif method == "BL":
                s_vec = np.full(N, t_mat.shape[1])
            else:
                raise ValueError(f"Unknown method '{method}'.")

        # common: repl = t_mat * (s_vec/(n + s_vec))[:, None]
        s_ratio = (s_vec / (n + s_vec))[:, None]
        repl    = t_mat * s_ratio
    else:
        repl = frac * np.ones((N, D)) * (threshold / n)[:, None]

    # ─── 4) CLOSURE & JIT‐ACCELERATED IMPUTATION ─────────────
    # initial closure
    X2 = X / np.nansum(X, axis=1)[:, None]
    mask = np.isnan(X2)                           # positions to impute
    col_mins = np.nanmin(X2, axis=0)

    # replace, adjust, renormalize in parallel
    X2 = _impute_and_adjust(X2, repl, col_mins, frac, adjust, mask)

    # ─── 5) PREPARE OUTPUT ──────────────────────────────────
    if output == "p-counts":
        res = X.copy()
        for i in range(N):
            row_mask = mask[i, :]
            if not row_mask.any():
                continue
            pos = np.where(~row_mask)[0][0]
            res[i, row_mask] = (X[i, pos] / X2[i, pos]) * X2[i, row_mask]
    else:
        res = X2

    logging.info(f"Zero replacement completed using {method} method")
    return res


def plr(
    mat: np.ndarray,
    use_tqdm: bool = False  # retained for API compatibility, but no longer used
) -> np.ndarray:
    """
    Vectorized pivot log‐ratio (PLR) transform for all taxa (rows) in `mat`.

    Args:
        mat: 2D array, shape (D taxa × N samples)
        use_tqdm: unused (kept for compatibility)

    Returns:
        D×N array of PLR values (rows = pivots, cols = samples)
    """
    # mat must be numeric
    if mat.ndim != 2:
        raise ValueError("Input mat must be 2D")
    D, N = mat.shape

    # 1) mask and log-transform
    mask_pos = mat > 0
    log_mat = np.zeros_like(mat, dtype=float)
    # in-place log only where positive
    np.log(mat, out=log_mat, where=mask_pos)

    # 2) per-sample sums & positive counts
    sum_logs = log_mat.sum(axis=0)             # shape (N,)
    pos_counts = mask_pos.sum(axis=0).astype(float)  # shape (N,)

    # 3) leave-one-out sums and counts (D×N arrays via broadcasting)
    sum_logs_excl   = sum_logs[None, :] - log_mat     # shape (D, N)
    pos_counts_excl = pos_counts[None, :] - mask_pos.astype(float)

    # 4) compute geometric means safely
    eps = np.finfo(float).eps
    # avoid division-by-zero; where count>0 compute exp(sum/count), else eps
    gm = np.exp(
        np.divide(
            sum_logs_excl,
            pos_counts_excl,
            out=np.zeros_like(sum_logs_excl),
            where=pos_counts_excl > 0
        )
    )
    gm = np.where(pos_counts_excl > 0, gm, eps)

    # 5) normalization factor
    norm = np.sqrt((D - 1) / D)

    # 6) compute PLR matrix in one go
    plr_mat = np.zeros_like(mat, dtype=float)
    # only where pivot value > 0
    idx = mask_pos
    plr_mat[idx] = norm * (log_mat[idx] - np.log(gm[idx]))

    logging.info(f"PLR transformation completed for {D} taxa")
    return plr_mat

def read_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read data from TSV file.

    Args:
        file_path: Path to the input file

    Returns:
        DataFrame with taxa as rows and samples as columns
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        logging.info(f"Read {df.shape[0]} taxa and {df.shape[1]} samples from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        raise


def write_data(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Write data to TSV file.

    Args:
        df: DataFrame to write
        file_path: Path to the output file
    """
    file_path = Path(file_path)

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.reset_index().to_csv(file_path, sep='\t', index=False)
        logging.info(f"Wrote {df.shape[0]} taxa and {df.shape[1]} samples to {file_path}")
    except Exception as e:
        logging.error(f"Failed to write output file: {e}")
        raise


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        verbose: Whether to enable debug logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(level=log_level, format=log_format,
                       handlers=[logging.StreamHandler()])


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Microbial genomics compositional pipeline"
    )

    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help='Pipeline command')

    # Filter subcommand
    parser_filter = subparsers.add_parser('filter', help='Filter taxa')
    parser_filter.add_argument('-i', '--input', required=True, help='Input TSV file')
    parser_filter.add_argument('-o', '--output', required=True, help='Output filtered TSV file')
    parser_filter.add_argument('--min-reads', type=int, default=100,
                              help='Minimum number of reads (default: 100)')
    parser_filter.add_argument('--nonzeros-prop', type=float, default=2/3,
                              help='Minimum proportion of non-zero samples (default: 0.6667)')
    parser_filter.add_argument('--non-species-label', default='Non-species',
                              help='Label for removed taxa (default: Non-species)')

    # Replace subcommand
    parser_replace = subparsers.add_parser('replace', help='Zero replacement')
    parser_replace.add_argument('-i', '--input', required=True, help='Input filtered TSV file')
    parser_replace.add_argument('-o', '--output', required=True, help='Output replaced TSV file')
    parser_replace.add_argument('--label', type=float, default=0.0,
                               help='Value marking zeros (default: 0.0)')
    parser_replace.add_argument('--method',
                               choices=['GBM', 'SQ', 'BL', 'CZM', 'USER'],
                               default='GBM',
                               help='Replacement method (default: GBM)')
    parser_replace.add_argument('--output-type',
                               choices=['prop', 'p-counts'],
                               default='prop',
                               help='Output type (default: prop)')
    parser_replace.add_argument('--frac', type=float, default=0.65,
                               help='Fraction for CZM method and adjustment (default: 0.65)')
    parser_replace.add_argument('--threshold', type=float, default=0.5,
                               help='Threshold for CZM method (default: 0.5)')
    parser_replace.add_argument('--no-adjust', dest='adjust', action='store_false',
                               help='Disable replacement adjustment')
    parser_replace.add_argument('--z-warning', type=float, default=0.8,
                               help='Threshold for zero warning (default: 0.8)')
    parser_replace.add_argument('--no-drop', dest='z_delete', action='store_false',
                               help='Don\'t drop rows/columns with too many zeros')
    parser_replace.add_argument('--t-file',
                               help='File containing t matrix (required for USER method)')
    parser_replace.add_argument('--s-file',
                               help='File containing s vector (required for USER method)')

    # PLR subcommand
    parser_plr = subparsers.add_parser('plr', help='Pivot log-ratio transform')
    parser_plr.add_argument('-i', '--input', required=True, help='Input replaced TSV file')
    parser_plr.add_argument('-o', '--output', required=True, help='Output PLR TSV file')
    parser_plr.add_argument('--no-progress', dest='use_tqdm', action='store_false',
                           help='Disable progress bar')

    # Pipeline subcommand (runs the entire pipeline)
    parser_pipeline = subparsers.add_parser('pipeline', help='Run the entire pipeline')
    parser_pipeline.add_argument('-i', '--input', required=True, help='Input TSV file')
    parser_pipeline.add_argument('-o', '--output', required=True, help='Output directory')
    parser_pipeline.add_argument('--min-reads', type=int, default=100,
                                help='Minimum number of reads (default: 100)')
    parser_pipeline.add_argument('--nonzeros-prop', type=float, default=2/3,
                                help='Minimum proportion of non-zero samples (default: 0.6667)')
    parser_pipeline.add_argument('--non-species-label', default='Non-species',
                                help='Label for removed taxa (default: Non-species)')
    parser_pipeline.add_argument('--method',
                                choices=['GBM', 'SQ', 'BL', 'CZM', 'USER'],
                                default='GBM',
                                help='Replacement method (default: GBM)')
    parser_pipeline.add_argument('--output-type',
                                choices=['prop', 'p-counts'],
                                default='prop',
                                help='Zero replacement output type (default: prop)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    return args


def run_filter(args: argparse.Namespace) -> None:
    """
    Run the filter command.

    Args:
        args: Command-line arguments
    """
    df = read_data(args.input)
    df_filt, removed = filter_taxa(df, args.min_reads, args.nonzeros_prop)
    df_out = add_removed_reads(df_filt, removed, label=args.non_species_label)
    write_data(df_out, args.output)


def run_replace(args: argparse.Namespace) -> None:
    """
    Run the replace command.

    Args:
        args: Command-line arguments
    """
    df = read_data(args.input)
    X = df.to_numpy().T

    # Handle USER method parameters
    t = None
    s = None
    if args.method == 'USER':
        if not args.t_file or not args.s_file:
            raise ValueError("USER method requires --t-file and --s-file parameters")

        try:
            # Load t matrix
            t_df = pd.read_csv(args.t_file, sep='\t', header=None)
            t = t_df.to_numpy()

            # Load s vector
            s_df = pd.read_csv(args.s_file, sep='\t', header=None)
            s = s_df.to_numpy().flatten()  # Ensure it's a 1D array

            logging.info(f"Loaded t matrix {t.shape} and s vector {s.shape}")

            # Validate dimensions
            if t.shape[0] != X.shape[0] or t.shape[1] != X.shape[1]:
                raise ValueError(f"t matrix shape {t.shape} does not match data shape {X.shape}")
            if len(s) != X.shape[0]:
                raise ValueError(f"s vector length {len(s)} does not match data rows {X.shape[0]}")

        except Exception as e:
            raise ValueError(f"Error loading USER method parameters: {e}")

    # Track which rows get dropped
    original_indices = np.arange(X.shape[0])
    original_columns = df.columns.copy()

    Y = cmult_repl(
        X,
        label=args.label,
        method=args.method,
        output=args.output_type,
        frac=args.frac,
        threshold=args.threshold,
        adjust=args.adjust,
        z_warning=args.z_warning,
        z_delete=args.z_delete,
        t=t,
        s=s,
    )

    # If rows were dropped, we need to adjust the columns in the output
    if Y.shape[0] != len(original_columns):
        logging.warning(f"Shape changed during zero replacement: {X.shape} -> {Y.shape}")
        # Create DataFrame with the actual columns we have data for
        df_rep = pd.DataFrame(Y.T, index=df.index, columns=df.columns[:Y.shape[0]])
    else:
        df_rep = pd.DataFrame(Y.T, index=df.index, columns=df.columns)

    write_data(df_rep, args.output)


def run_plr(args: argparse.Namespace) -> None:
    """
    Run the PLR command.

    Args:
        args: Command-line arguments
    """
    df = read_data(args.input)
    mat = df.to_numpy()
    arr_plr = plr(mat, use_tqdm=args.use_tqdm)
    df_plr = pd.DataFrame(arr_plr, index=df.index, columns=df.columns)
    write_data(df_plr, args.output)


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Run the complete pipeline.

    Args:
        args: Command-line arguments
    """
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output files
    filtered_file = output_dir / "filtered.tsv"
    replaced_file = output_dir / "replaced.tsv"
    plr_file = output_dir / "plr.tsv"

    # Step 1: Filter
    logging.info("=== STEP 1: Filtering ===")
    df = read_data(args.input)
    df_filt, removed = filter_taxa(df, args.min_reads, args.nonzeros_prop)
    df_out = add_removed_reads(df_filt, removed, label=args.non_species_label)
    write_data(df_out, filtered_file)

    # Step 2: Replace
    logging.info("=== STEP 2: Zero Replacement ===")
    X = df_out.to_numpy().T
    Y = cmult_repl(
        X,
        method=args.method,
        output=args.output_type,
    )
    df_rep = pd.DataFrame(Y.T, index=df_out.index, columns=df_out.columns)
    write_data(df_rep, replaced_file)

    # Step 3: PLR
    logging.info("=== STEP 3: PLR Transformation ===")
    mat = df_rep.to_numpy()
    arr_plr = plr(mat)
    df_plr = pd.DataFrame(arr_plr, index=df_rep.index, columns=df_rep.columns)
    write_data(df_plr, plr_file)

    logging.info(f"Pipeline complete! Results saved to {output_dir}")


def main() -> None:
    """
    Main entry point for the pipeline.
    """
    args = parse_args()
    setup_logging(args.verbose)

    try:
        if args.command == 'filter':
            run_filter(args)
        elif args.command == 'replace':
            run_replace(args)
        elif args.command == 'plr':
            run_plr(args)
        elif args.command == 'pipeline':
            run_pipeline(args)
        else:
            logging.error(f"Unknown command: {args.command}")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            logging.exception("Detailed traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()

# Example commands:
# 1) Run the complete pipeline:
#    python microbiome_pipeline.py pipeline -i ella.txt -o results/
#
# 2) Or run steps individually:
#    # Filter taxa:
#    python microbiome_pipeline.py filter -i ella.txt -o ella_filtered.tsv --min-reads 100 --nonzeros-prop 0.6667
#    # Zero replacement:
#    python microbiome_pipeline.py replace -i ella_filtered.tsv -o ella_replaced.tsv --method GBM --output-type prop
#    # PLR transformation:
#    python microbiome_pipeline.py plr -i ella_replaced.tsv -o ella_plr.tsv