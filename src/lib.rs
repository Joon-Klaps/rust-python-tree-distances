//! Crate root: lightweight module orchestration and public re-exports.
//!
//! Modules:
//! - `distances`: generic distance trait + RF / cluster affinity implementations.
//! - `io`: reading and parsing BEAST/NEXUS tree files.
//! - `utils`: shared internal helpers for parsing & normalization.
//! - `api`: (optional) Python bindings via `pyo3`.
//!
//! Public API kept stable by re-exporting key items from the new modules.

pub mod distances;
pub mod io;
// Note: Python bindings are omitted by default to avoid proc-macro/toolchain friction.
// Add a gated `api` module when needed.
// Re-export frequently used types & functions
// pub use distances::{
//     DistanceMetric, RobinsonFoulds, ClusterAffinity,
//     compute_pairwise_dist, compute_pairwise_robinson_foulds, compute_pairiwise_cluster_affinity,
// };
// pub use crate::utils::parse_newick_to_phylo;
pub use io::{read_beast_trees, write_matrix_tsv};



