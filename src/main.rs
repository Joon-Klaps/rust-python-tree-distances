use clap::{Parser, ValueEnum};
use rust_python_tree_distances::io::{read_beast_trees, write_matrix_tsv};
use rust_python_tree_distances::snapshot::TreeSnapshot;
use rust_python_tree_distances::distances::{rf_from_snapshots, weighted_rf_from_snapshots, kf_from_snapshots};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

/// Compute pairwise Robinson–Foulds distances from a BEAST/NEXUS tree file
/// and write a labeled distance matrix (TSV) where row/column names are tree names.
#[derive(Parser, Debug)]
#[command(name = "tree-dists", version, about = "Pairwise RF distance matrix for BEAST trees")]
struct Args {
    /// Path to BEAST .trees (NEXUS) file
    #[arg(short = 'i', long = "input")]
    input: PathBuf,

    /// Burn-in by number of trees (drop first N trees)
    #[arg(short = 't', long = "burnin-trees", default_value_t = 0)]
    burnin_trees: usize,

    /// Burn-in by state (keep trees with STATE_ > value)
    #[arg(short = 's', long = "burnin-states", default_value_t = 0)]
    burnin_states: usize,

    /// Output path for TSV distance matrix
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Use TRANSLATE block to map taxon IDs to labels when available
    #[arg(long = "use-real-taxa", default_value_t = false)]
    use_real_taxa: bool,

    /// Distance metric to compute: rf | weighted | kf
    #[arg(long = "metric", value_enum, default_value_t = MetricArg::Rf)]
    metric: MetricArg,

    /// Quiet mode: suppresses progress messages on stdout
    #[arg(short = 'q', long = "quiet", default_value_t = false)]
    quiet: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum MetricArg { Rf, Weighted, Kf }

fn main() {
    let args = Args::parse();

    // Read trees with names
    let t0 = Instant::now();
    let (taxons,named_trees) = read_beast_trees(
        &args.input,
        args.burnin_trees,
        args.burnin_states,
        args.use_real_taxa,
    );
    if named_trees.is_empty() {
        eprintln!("No trees parsed from {:?}.", args.input);
        std::process::exit(2);
    }
    let read_s = t0.elapsed().as_secs_f64();
    log_if(!args.quiet, format!("Reading in beast {read_s:.3}s"));
    log_if(!args.quiet, format!("Read in {} taxons for {} trees", taxons.len(), named_trees.len()));
    let (names, trees): (Vec<String>, Vec<_>) = named_trees.into_iter().unzip();

    // Build bitset snapshots once
    let t1 = Instant::now();
    let snaps: Vec<TreeSnapshot> = trees
        .iter()
        .map(|tree| TreeSnapshot::from_tree(tree))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|e| {
            eprintln!("Failed to build snapshots: {e}");
            std::process::exit(3);
        });
    let snap_s = t1.elapsed().as_secs_f64();
    log_if(!args.quiet, format!("Creating tree bit snapshots {snap_s:.3}s"));

    let t2 = Instant::now();
    let (metric_label, metric_fn): (&str, fn(&TreeSnapshot, &TreeSnapshot) -> f64) = match args.metric {
        // rf is the only one that returns usize, so cast to f64
        MetricArg::Rf => ("RF", |a, b| rf_from_snapshots(a, b) as f64),
        MetricArg::Weighted => ("Weighted", weighted_rf_from_snapshots),
        MetricArg::Kf => ("KF", kf_from_snapshots),
    };

    log_if(!args.quiet, format!("Determining distances using {metric_label} for {} combinations", names.len() * (names.len() - 1) / 2));

    let n = names.len();

    // Compute distances in parallel
    let pairs: Vec<(usize, usize, f64)> = (0..n).into_par_iter()
        .flat_map_iter(|i| {
            (i+1..n).map(move |j| (i, j))
        })
        .map(|(i, j)| {
            let dist = metric_fn(&snaps[i], &snaps[j]);
            (i, j, dist)
        })
        .collect();

    let mut mat = vec![vec![0.0f64; n]; n];
    for (i, j, d) in pairs {
        mat[i][j] = d;
        mat[j][i] = d;
    }

    let comp_s = t2.elapsed().as_secs_f64();
    log_if(!args.quiet, format!("Determining distances using {metric_label} {comp_s:.3}s"));

    let t3 = Instant::now();
    if let Err(e) = write_matrix_tsv(&args.output, &names, &mat) {
        eprintln!("Failed to write output {:?}: {e}", args.output);
        std::process::exit(4);
    }
    let write_s = t3.elapsed().as_secs_f64();
    log_write_done(!args.quiet, &args.output, write_s);
}

fn log_if(show: bool, msg: String) {
    if show { println!("{}", msg); }
}

fn log_write_done(show: bool, output: &PathBuf, secs: f64) {
    if !show { return; }
    let is_stdout = output.as_os_str() == "-";
    if is_stdout {
        println!("Writing to stdout {secs:.3}s");
    } else {
        println!("Writing to output {secs:.3}s");
    }
}
