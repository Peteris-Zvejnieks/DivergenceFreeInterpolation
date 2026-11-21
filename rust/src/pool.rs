//! The thread pool for parallel work called from Python, safe across `fork`.
//!
//! After a `fork` (e.g. `multiprocessing` workers), the child inherits rayon's global pool
//! state but not its threads, so any parallel call would wait forever. Work from Python runs
//! in a pool owned by the current process instead: a child sees a new process id and builds
//! its own pool.

use std::sync::{Arc, Mutex};

use rayon::ThreadPool;

static POOL: Mutex<Option<(u32, Arc<ThreadPool>)>> = Mutex::new(None);

/// Runs `operation` in this process's thread pool.
pub fn install<R: Send>(operation: impl FnOnce() -> R + Send) -> R {
    let pool = {
        let mut current = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let process = std::process::id();
        match current.as_ref() {
            Some((owner, pool)) if *owner == process => Arc::clone(pool),
            _ => {
                // A pool inherited through fork has no threads in this process: dropping it
                // would wait for them, so leak it
                if let Some(inherited) = current.take() {
                    std::mem::forget(inherited);
                }
                let pool = Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .build()
                        .expect("failed to start the thread pool"),
                );
                *current = Some((process, Arc::clone(&pool)));
                pool
            }
        }
    };
    pool.install(operation)
}
