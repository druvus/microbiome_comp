# microbiome_comp

A Python package for microbial genomics compositional workflows. This tool provides optimized implementations for:

- Taxa filtering by read counts and zero proportion
- Bayesian-multiplicative zero replacement (cmult_repl)
- Pivot log-ratio (PLR) transformation

## Installation


```bash
# Install environment
conda create -n microbiome_comp python numpy pandas scipy numba tqdm
conda activate microbiome_comp

```

```bash
# Install from source
git clone https://github.com/druvus/microbiome_comp
cd microbiome_comp
pip install .
```

## Usage

### Command Line Interface

The package provides a command-line interface with several subcommands:

#### Complete Pipeline

Run the entire workflow from filtering to PLR transformation:

```bash
microbiome_comp pipeline -i input.tsv -o results/
```

#### Individual Steps

You can also run each step individually:

1. **Filtering taxa**:
```bash
microbiome_comp filter -i input.tsv -o filtered.tsv --min-reads 100 --nonzeros-prop 0.6667
```

2. **Zero replacement**:
```bash
microbiome_comp replace -i filtered.tsv -o replaced.tsv --method GBM --output-type prop
```

3. **PLR transformation**:
```bash
microbiome_comp plr -i replaced.tsv -o plr.tsv
```

### Command Options

#### Global Options:
- `--verbose`, `-v`: Enable verbose logging

#### Filter Command:
- `-i`, `--input`: Input TSV file
- `-o`, `--output`: Output filtered TSV file
- `--min-reads`: Minimum number of reads (default: 100)
- `--nonzeros-prop`: Minimum proportion of non-zero samples (default: 0.6667)
- `--non-species-label`: Label for aggregated removed taxa (default: "Non-species")

#### Replace Command:
- `-i`, `--input`: Input filtered TSV file
- `-o`, `--output`: Output replaced TSV file
- `--label`: Value marking zeros (default: 0.0)
- `--method`: Replacement method (choices: GBM, SQ, BL, CZM, USER; default: GBM)
- `--output-type`: Output type (choices: prop, p-counts; default: prop)
- `--frac`: Fraction for CZM method and adjustment (default: 0.65)
- `--threshold`: Threshold for CZM method (default: 0.5)
- `--no-adjust`: Disable replacement adjustment
- `--z-warning`: Threshold for zero warning (default: 0.8)
- `--no-drop`: Don't drop rows/columns with too many zeros
- `--t-file`: File containing t matrix (required for USER method)
- `--s-file`: File containing s vector (required for USER method)

#### PLR Command:
- `-i`, `--input`: Input replaced TSV file
- `-o`, `--output`: Output PLR TSV file
- `--no-progress`: Disable progress bar

## Input Format

The input should be a tab-separated value (TSV) file with:
- Taxa as rows
- Samples as columns
- First column contains taxa names/IDs
- First row contains sample names/IDs

## Methods

### Taxa Filtering

Filters taxa based on two criteria:
1. Minimum number of reads in at least one sample
2. Minimum proportion of non-zero samples

### Zero Replacement (cmult_repl)

A Bayesian-multiplicative replacement method for count zeros with several variants:
- **GBM**: Geometric Bayesian multiplicative (default)
- **SQ**: Square-root count method
- **BL**: Base-level method
- **CZM**: Constant zero multiplicative method
- **USER**: User-defined parameters

### PLR Transformation

The pivot log-ratio (PLR) transformation computes, for each taxon and sample:
1. The geometric mean of all other taxa in the sample (excluding the current taxon)
2. The log-ratio of the current taxon to this geometric mean
3. Normalization by √(D-1)/D, where D is the number of taxa

This transformation is particularly useful for compositional data as it addresses the compositional nature of microbiome data.

## Performance

The package uses Numba JIT compilation for improved performance, particularly for the zero replacement step.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Author

- Andreas Sjödin
- Daniel Svensson