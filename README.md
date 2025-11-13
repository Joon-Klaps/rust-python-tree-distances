# rust-python-tree-distances CLI

Compute pairwise tree distances (Robinson–Foulds, weighted RF, Kuhner–Felsenstein) from BEAST/NEXUS `.trees` files and write a labeled distance matrix.

## Install / Build

> [!INFO]
> Install rust toolchain from https://rustup.rs/ if you don't have it yet.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Clone the repository:
```bash
git clone https://github.com/Joon-Klaps/rust-python-tree-distances.git
```

Requirements: Rust toolchain (stable). Then build the binary:

```bash
cd rust-python-tree-distances
cargo build --release
```

The executable will be at `target/release/rust-python-tree-distances`.

## Usage

```bash
# if rust-python-tree-distances is not in your PATH, use the full path, e.g. ./target/release/rust-python-tree-distances
rust-python-tree-distances \
  --input <path/to/file.trees> \
  --output <path/to/output.tsv[.gz]> \
  [--burnin-trees <N>] \
  [--burnin-states <STATE>] \
  [--use-real-taxa] \
  [--metric rf|weighted|kf] \
  [-q|--quiet]
```

>[!NOTE]
> Something might be of with kf distance metric, it didn't pass testing where we compared the parallell implementation vs the single-threaded one.

Flags and options:

- `-i, --input <INPUT>`: Path to BEAST `.trees` (NEXUS) file.
- `-o, --output <OUTPUT>`: Output path for the TSV distance matrix. If the path ends with `.gz` it will be gzip-compressed. Use `-` to write to stdout (uncompressed).
- `-t, --burnin-trees <N>`: Drop the first N trees (default: 0).
- `-s, --burnin-states <STATE>`: Keep only trees with `STATE_ > STATE` (default: 0).
- `--use-real-taxa`: Map numeric taxon IDs to labels using the TRANSLATE block if present.
- `--metric <rf|weighted|kf>`: Choose the distance metric (default: `rf`), weighted referring to weighted RF.
- `-q, --quiet`: Suppress progress messages on stdout. Errors still go to stderr.

The output is a square TSV matrix where both the header row and the first column are tree names, constructed as `<file_basename>_tree_STATE<state>`. When writing to stdout (`-o -`), the matrix is printed to stdout, allowing easy piping.

## Examples

- Compute RF matrix and write to gzipped file:

```bash
rust-python-tree-distances \
  -i tests/data/hiv1.trees \
  -o out/hiv1_rf.tsv.gz \
  --metric rf
```

- Compute weighted RF matrix using real taxon labels and print to stdout:

```bash
rust-python-tree-distances \
  -i tests/data/hiv1.trees \
  -o - \
  --use-real-taxa \
  --metric weighted \
  -q | head
```

- Apply burn-in by tree count and state:

```bash
rust-python-tree-distances \
  -i tests/data/hiv1.trees \
  -o out/hiv1_kf.tsv \
  -t 100 \
  -s 300000 \
  --metric kf
```

## Performance notes

- Trees are parsed once. Bitset snapshots are built once and reused for pairwise comparisons. Parallelism is provided by `rayon`.
- Weighted RF and KF produce floating-point matrices; RF produces integer matrices.

## Troubleshooting

- If no trees are parsed, verify the input is a valid NEXUS `.trees` file and adjust `--burnin-*` settings.
- Use `-q` when writing to stdout and piping to other tools to suppress timing messages.
- For gzipped output, ensure the output filename ends with `.gz`.
