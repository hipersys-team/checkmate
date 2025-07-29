#![allow(dead_code)]
#![feature(test)]
#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]
#![feature(mpmc_channel)]

pub mod chunk_info;
pub mod grad_bucket;
mod worker;

use crate::grad_bucket::GradBucket;
use crate::worker::start_worker;
use libtcp::ffi::*;
use log::info;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::hint::spin_loop;
use std::sync::mpmc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use thread_priority::ThreadPriority;
use worker::{set_affinity, TCPWorker};

lazy_static::lazy_static! {
    pub static ref NUM_TRAINING: i64 = std::env::var("NUM_TRAINING")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .expect("NUM_TRAINING must be a number");
}

// These are storage related constants and not the training cluster related.
pub const NRANKS: i64 = 2;
pub const NCCL_NCHANNELS: i64 = 4;
pub const NUM_CONNECTIONS: usize = (NRANKS * NCCL_NCHANNELS) as usize;

#[pyclass]
pub struct Server {
    poller: thread::JoinHandle<()>,
    req_sender: Sender<usize>,
    resp_receiver: Receiver<Arc<GradBucket>>,

    nnodes: i64,
    node_rank: i64,
    memcpy_tpool: ThreadPool,
}

fn initialize_grad_buckets(sizes: Vec<i32>, dtype_size: i64) -> Vec<Arc<GradBucket>> {
    log::warn!("Chunking buckets for {} training nodes", *NUM_TRAINING);
    sizes
        .into_iter()
        .enumerate()
        .map(|(id, size)| Arc::new(GradBucket::new(id, size as i64, *NUM_TRAINING, dtype_size)))
        .collect()
}

fn process_bucket(
    current_bucket: &GradBucket,
    workers: &Arc<Vec<Arc<TCPWorker>>>,
    bucket_len: usize,
    bucket_base: *mut u8,
) {
    let mut requested_size = 0;
    for worker in workers.iter() {
        let channel_id = worker.channel_id;
        current_bucket.chunk_info[worker.rank][channel_id]
            .iter()
            .filter(|info| !info.chunk_offset.is_negative())
            .for_each(|info| {
                assert!(info.chunk_offset <= bucket_len as i32);
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        bucket_base.add(info.chunk_offset as usize),
                        info.chunk_size as usize,
                    )
                };
                worker.req_sender.send((slice, info.clone())).unwrap();
                requested_size += info.chunk_size;
            });
    }
    assert!(bucket_len == requested_size as usize);
}

fn wait_for_workers(workers: &Arc<Vec<Arc<TCPWorker>>>, bucket_len: usize) {
    let mut filled_size = 0;
    'data: loop {
        for worker in workers.iter() {
            if let Ok(size) = worker.resp_receiver.try_recv() {
                filled_size += size;
                if filled_size == bucket_len {
                    break 'data;
                }
            }
        }
    }
    assert!(bucket_len == filled_size);
}

#[pymethods]
impl Server {
    #[new]
    pub fn new(
        start_port: u16,
        sizes: Vec<i32>,
        element_size: i64,
        nnodes: i64,
        node_rank: i64,
    ) -> Self {
        let _ = env_logger::try_init();
        unsafe {
            if tpa_init(NUM_CONNECTIONS as i32) < 0 {
                panic!("tpa_init");
            }
            let barrier = Arc::new(std::sync::Barrier::new(NUM_CONNECTIONS + 1));

            // Start workers in a given order
            let mut workers = Arc::new(Vec::with_capacity(NUM_CONNECTIONS));
            (0..NRANKS).for_each(|rank| {
                (0..NCCL_NCHANNELS).for_each(|id| {
                    let worker = Arc::new(start_worker(
                        rank as usize,
                        id as usize,
                        start_port,
                        barrier.clone(),
                    ));
                    Arc::get_mut(&mut workers).unwrap().push(worker);
                    thread::sleep(std::time::Duration::from_millis(100));
                });
            });
            barrier.wait();
            println!(":: all {NUM_CONNECTIONS} workers waiting for connections");
            let (req_sender, req_receiver) = channel::<usize>();
            let (resp_sender, resp_receiver) = channel();

            let poller = thread::spawn(move || {
                set_affinity(NUM_CONNECTIONS + 1, ThreadPriority::Max);
                let mut grad_buckets = initialize_grad_buckets(sizes, element_size);
                log::info!(
                    ":: initialized {} grad buckets on node {}",
                    grad_buckets.len(),
                    node_rank
                );

                let mut bucket_id = node_rank as usize;
                loop {
                    // Wait if the current bucket is not used by pytorch
                    while !grad_buckets[bucket_id].is_ready_to_fill() {
                        spin_loop();
                    }

                    let current_bucket = &mut grad_buckets[bucket_id];
                    // Scoped locked on the current bucket
                    {
                        // Give each worker a chunk of the bucket to fill
                        let mut bucket_wr_guard = current_bucket.write_guard();
                        let bucket_slice = bucket_wr_guard.as_mut_slice();
                        let bucket_base = bucket_slice.as_mut_ptr();
                        let bucket_len = bucket_slice.len();
                        process_bucket(current_bucket, &workers, bucket_len, bucket_base);

                        // Check if all workers have filled their chunks
                        wait_for_workers(&workers, bucket_len);
                        current_bucket.set_ready_to_drain();
                    }

                    // Server -> PyTorch
                    if let Ok(bucketid) = req_receiver.recv() {
                        let bucket = grad_buckets[bucketid].clone();
                        resp_sender.send(bucket).unwrap();
                    }
                    info!("\n");
                    bucket_id += nnodes as usize;
                    if bucket_id < grad_buckets.len() {
                        bucket_id %= grad_buckets.len();
                    } else {
                        bucket_id = node_rank as usize;
                    }
                    assert_eq!(bucket_id as i64 % nnodes, node_rank);
                }
            });

            // This is fine because these cores are idle until the server
            // returns all the buckets to the PyTorch side.
            let available_cores: Vec<usize> =
                ((NUM_CONNECTIONS + 2)..num_cpus::get_physical()).collect();
            let memcpy_tpool = ThreadPoolBuilder::new()
                .num_threads(available_cores.len())
                .start_handler(move |thread_index| {
                    if let Some(core_id) = available_cores.get(thread_index) {
                        set_affinity(*core_id, ThreadPriority::Max);
                    }
                })
                .build()
                .expect("Failed to build thread pool");
            Server {
                poller,
                req_sender,
                resp_receiver,

                nnodes,
                node_rank,
                memcpy_tpool,
            }
        }
    }

    pub fn update_grad_bucket(&self, id: usize, buf_base_addr: u64, size: i32) {
        Python::with_gil(|gil| {
            gil.allow_threads(|| {
                assert_eq!(id as i64 % self.nnodes, self.node_rank);
                self.req_sender.send(id).unwrap();
                loop {
                    if let Ok(grad_bucket) = self.resp_receiver.try_recv() {
                        let buffer = unsafe {
                            std::slice::from_raw_parts_mut(buf_base_addr as *mut u8, size as usize)
                        };
                        grad_bucket.drain_buffer(buffer, &self.memcpy_tpool);
                        break;
                    }
                }
            });
        });
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn network(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Server>()?;
    Ok(())
}
