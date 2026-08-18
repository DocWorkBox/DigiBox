//! Native DigiBox desktop contracts for the AVTR-1 runtime.

pub mod app;
pub mod health;
pub mod navigation;
pub mod runtime;
pub mod supervisor;

#[cfg(windows)]
pub mod windows_job;

pub use app::run;
