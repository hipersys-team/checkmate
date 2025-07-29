#![feature(slice_ptr_get)]

use pyo3::prelude::*;

pub mod client;
pub mod server;

pub use client::Client;
pub use server::Server;

/// A Python module implemented in Rust.
#[pymodule]
fn restore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Server>()?;
    m.add_class::<Client>()?;
    Ok(())
}
