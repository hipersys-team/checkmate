extern crate test;

use crate::chunk_info::ChunkOffsetInfo;
use crate::worker::ChunkInfo;
use crate::NCCL_NCHANNELS;
use std::hint::spin_loop;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::{atomic::AtomicBool, RwLock};

use crossbeam_utils::CachePadded;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_stream_si128};
const EMPTY: bool = false;
const FILLED: bool = true;

pub struct GradBucket {
    pub id: usize,
    pub size: i64,
    pub state: CachePadded<AtomicBool>,

    buffer: RwLock<Vec<u8>>,

    // Two ranks, each with NCCL_NCHANNELS, each with a vector of ChunkInfo.
    pub chunk_info: Vec<Vec<Vec<Arc<ChunkInfo>>>>,
}

impl GradBucket {
    pub fn new(id: usize, size: i64, nranks: i64, dtype_size: i64) -> Self {
        let buffer = RwLock::new(vec![0u8; size as usize]);
        let mut chunk_info = Vec::with_capacity(nranks as usize);
        let mut rank_0_chunk_info = Vec::with_capacity(NCCL_NCHANNELS as usize);
        let mut rank_n_chunk_info = Vec::with_capacity(NCCL_NCHANNELS as usize);

        // For Rank 0
        let nelem = size / dtype_size;
        for channel_id in 0..NCCL_NCHANNELS {
            let chunk_offset_info =
                ChunkOffsetInfo::new(0, nranks, channel_id, nelem, NCCL_NCHANNELS, dtype_size);
            rank_0_chunk_info.push(
                chunk_offset_info
                    .get_iter()
                    .map(|info| {
                        Arc::new(ChunkInfo {
                            chunk_offset: *info.0 as i32,
                            chunk_size: *info.1 as i32 * dtype_size as i32,
                        })
                    })
                    .collect::<Vec<Arc<ChunkInfo>>>(),
            );
        }

        // For Rank (N-1)
        for channel_id in 0..NCCL_NCHANNELS {
            let chunk_offset_info = ChunkOffsetInfo::new(
                nranks - 1,
                nranks,
                channel_id,
                nelem,
                NCCL_NCHANNELS,
                dtype_size,
            );
            rank_n_chunk_info.push(
                chunk_offset_info
                    .get_iter()
                    .map(|info| {
                        Arc::new(ChunkInfo {
                            chunk_offset: *info.0 as i32,
                            chunk_size: *info.1 as i32 * dtype_size as i32,
                        })
                    })
                    .collect::<Vec<Arc<ChunkInfo>>>(),
            );
        }
        chunk_info.push(rank_0_chunk_info);
        chunk_info.push(rank_n_chunk_info);
        Self {
            id,
            size,
            state: CachePadded::new(AtomicBool::new(false)),
            buffer,
            chunk_info,
        }
    }

    pub fn is_ready_to_fill(&self) -> bool {
        self.state.load(Ordering::Acquire) == EMPTY
    }
    pub fn set_ready_to_fill(&self) {
        assert!(self
            .state
            .compare_exchange(FILLED, EMPTY, Ordering::AcqRel, Ordering::Acquire,)
            .unwrap());
    }

    pub fn is_ready_to_drain(&self) -> bool {
        self.state.load(Ordering::Acquire) == FILLED
    }

    pub fn set_ready_to_drain(&self) {
        assert!(!self
            .state
            .compare_exchange(EMPTY, FILLED, Ordering::AcqRel, Ordering::Acquire,)
            .unwrap());
    }

    pub fn write_guard(&self) -> std::sync::RwLockWriteGuard<'_, Vec<u8>> {
        self.buffer.write().unwrap()
    }

    pub fn drain_buffer(&self, slice: &mut [u8], pool: &ThreadPool) {
        while !self.is_ready_to_drain() {
            spin_loop();
        }
        let buffer = self.buffer.read().unwrap();
        assert!(slice.len() == buffer.len());

        #[cfg(feature = "memcpy")]
        pool.install(|| {
            parallel_memcpy(buffer.as_slice(), slice);
        });
        #[cfg(not(feature = "memcpy"))]
        slice.copy_from_slice(buffer.as_slice());
        self.set_ready_to_fill();
    }
}

#[inline]
pub fn parallel_memcpy(src: &[u8], dst: &mut [u8]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "Source and destination must have the same length"
    );

    assert!(
        (dst.as_ptr() as usize).is_multiple_of(16),
        "Destination must be 16-byte aligned"
    );
    let chunk_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    src.par_chunks_exact(chunk_size)
        .zip(dst.par_chunks_exact_mut(chunk_size))
        .for_each(|(src_chunk, dst_chunk)| {
            (0..chunk_size).step_by(16).for_each(|offset| unsafe {
                opt_mov16(
                    dst_chunk.as_mut_ptr().add(offset),
                    src_chunk.as_ptr().add(offset),
                );
            });
        });

    // Process the remaining data with copy_from_slice
    let remaining = src.len() % chunk_size;
    if remaining > 0 {
        let offset = src.len() - remaining;
        dst[offset..].copy_from_slice(&src[offset..]);
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn opt_mov16(dst: *mut u8, src: *const u8) {
    let zmm0 = _mm_loadu_si128(src as *const __m128i);
    _mm_stream_si128(dst as *mut __m128i, zmm0);
}
