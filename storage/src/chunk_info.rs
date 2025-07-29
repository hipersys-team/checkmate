use std::cmp;

type ChunkOffset = i64;
type ChunkElems = i64;

const NCCL_STEPS: i64 = 8;

// Generic function to compute the ceiling of integer division
fn div_up(x: i64, y: i64) -> i64 {
    (x + y - 1) / y
}

// Function to align size to the nearest higher multiple of alignment
fn align_up(size: i64, alignment: i64) -> i64 {
    (size + alignment - 1) & !(alignment - 1)
}

// Define the CBD structure
#[derive(Default, Debug)]
struct Cbd {
    count_lo: i64,
    count_mid: i64,
    count_hi: i64,
    chunk_grains_lo: i64,
    chunk_grains_mid: i64,
    chunk_grains_hi: i64,
}

// Define the ncclDevWorkColl structure
#[derive(Default, Debug)]
struct NcclDevWorkColl {
    channel_lo: i64,
    channel_hi: i64,
    cbd: Cbd,
}

// Function to compute offsets and counts based on channel ID
fn nccl_coll_cbd_part(
    work: &NcclDevWorkColl,
    channel_id: i64,
    elt_size: i64,
    part_offset: &mut i64,
    part_count: &mut i64,
    chunk_count: &mut i64,
) {
    let elt_per_grain = 512 / elt_size;
    let n_mid_channels = work.channel_hi - work.channel_lo - 1;
    // Assuming nMidChannels < 0 implies countMid == 0
    if channel_id == work.channel_lo {
        *part_offset = 0.into(); // Assuming Int can be zero-initialized
        *part_count = work.cbd.count_lo;
        *chunk_count = work.cbd.chunk_grains_lo * elt_per_grain;
    } else if channel_id == work.channel_hi {
        *part_offset = work.cbd.count_lo + n_mid_channels * work.cbd.count_mid;
        *part_count = work.cbd.count_hi;
        *chunk_count = work.cbd.chunk_grains_hi * elt_per_grain;
    } else {
        let mid = channel_id - work.channel_lo - 1;
        *part_offset = work.cbd.count_lo + mid * work.cbd.count_mid;
        *part_count = work.cbd.count_mid;
        *chunk_count = work.cbd.chunk_grains_mid * elt_per_grain;
    }
}

// Rust function equivalent to scheduleCollTasksToPlan
fn schedule_coll_tasks_to_plan(
    n_channels: i64,
    datatype_size: i64,
    count: i64,
    dev_work: &mut NcclDevWorkColl,
    nccl_buff_size: i64,
) {
    const MIN_TRAFFIC_PER_CHANNEL: i64 = 512;
    const GRAIN_SIZE: i64 = 512;
    let element_size = datatype_size;
    let traffic_bytes = count * element_size * 2;

    let traffic_per_channel = std::cmp::max(MIN_TRAFFIC_PER_CHANNEL, traffic_bytes / n_channels);
    let mut channel_id = 0;
    let current_traffic = 0;
    const CELL_SIZE: i64 = 16;
    let elements_per_cell = CELL_SIZE / element_size;
    let cells = div_up(count * element_size, CELL_SIZE);
    let traffic_per_byte = 2;
    let _traffic_per_element = element_size * traffic_per_byte;
    let traffic_per_cell = CELL_SIZE * traffic_per_byte;
    let cells_per_channel = std::cmp::min(cells, div_up(traffic_per_channel, traffic_per_cell));
    let mut cells_lo = if channel_id + 1 == n_channels {
        // On last channel, everything goes to "lo"
        cells
    } else {
        std::cmp::min(
            cells,
            (traffic_per_channel - current_traffic) / traffic_per_cell,
        )
    };
    let mut n_mid_channels = (cells - cells_lo) / cells_per_channel;
    let mut cells_hi = (cells - cells_lo) % cells_per_channel;
    if cells_hi == 0 && n_mid_channels != 0 {
        cells_hi = cells_per_channel;
        n_mid_channels -= 1;
    }
    if cells_lo == 0 {
        // Least channel skipped. Make the next channel the new least.
        channel_id += 1;
        if n_mid_channels == 0 {
            cells_lo = cells_hi;
            cells_hi = 0;
        } else {
            cells_lo = cells_per_channel;
            n_mid_channels -= 1;
        }
    }
    let count_mid = if n_mid_channels != 0 {
        cells_per_channel * elements_per_cell
    } else {
        0
    };
    let count_lo = cells_lo * elements_per_cell;
    let mut count_hi = cells_hi * elements_per_cell;
    count_hi = if count_hi != 0 { count_hi } else { count_lo };
    count_hi -= cells * elements_per_cell - count;

    let n_channels = ((count_lo != 0) as i64) + n_mid_channels + ((cells_hi != 0) as i64);
    let chunk_size = (nccl_buff_size / NCCL_STEPS) * (NCCL_STEPS / 2) / GRAIN_SIZE * GRAIN_SIZE;

    dev_work.channel_lo = channel_id;
    dev_work.channel_hi = (channel_id + n_channels - 1) as i64;
    dev_work.cbd.count_lo = count_lo as i64;
    dev_work.cbd.count_mid = count_mid as i64;
    dev_work.cbd.count_hi = count_hi as i64;

    if count_lo != 0 {
        dev_work.cbd.chunk_grains_lo = chunk_size / GRAIN_SIZE;
    }
    if count_hi != 0 {
        dev_work.cbd.chunk_grains_hi = chunk_size / GRAIN_SIZE;
    }
    if n_mid_channels != 0 {
        dev_work.cbd.chunk_grains_mid = chunk_size / GRAIN_SIZE;
    }

    log::info!("Collective AllReduce(elemsize: {}, Ring, Simple) count={} channel{{Lo..Hi}}={{{}..{}}} count{{Lo,Mid,Hi}}={{{},{},{}}}, chunkBytes{{Lo,Mid,Hi}}={{{},{},{}}}",
        datatype_size,
        count, dev_work.channel_lo, dev_work.channel_hi,
        dev_work.cbd.count_lo, dev_work.cbd.count_mid, dev_work.cbd.count_hi,
        dev_work.cbd.chunk_grains_lo * GRAIN_SIZE,
        dev_work.cbd.chunk_grains_mid * GRAIN_SIZE,
        dev_work.cbd.chunk_grains_hi * GRAIN_SIZE);
}

// Function to perform modulo operation on ranks
fn mod_ranks(r: i64, nranks: i64) -> i64 {
    if r >= nranks {
        r - nranks
    } else {
        r
    }
}

// Implement fill_chunk_info function
fn fill_chunk_info(
    chunk_info: &mut ChunkOffsetInfo,
    ring_ix: i64,
    nranks: i64,
    channel_id: i64,
    count: i64,
    n_channel: i64,
    dtype_size: i64,
) {
    let mut grid_offset: i64 = 0;
    let mut channel_count: i64 = 0;
    let mut chunk_count: i64 = 0;

    let mut work = NcclDevWorkColl::default();
    schedule_coll_tasks_to_plan(
        n_channel,
        dtype_size,
        count,
        &mut work,
        chunk_info.nccl_buff_size,
    );
    nccl_coll_cbd_part(
        &work,
        channel_id,
        dtype_size,
        &mut grid_offset,
        &mut channel_count,
        &mut chunk_count,
    );

    chunk_info.ring_ix = ring_ix;
    chunk_info.nranks = nranks;
    chunk_info.channel_id = channel_id;
    chunk_info.count = count;
    chunk_info.n_channel = n_channel;
    chunk_info.dtype_size = dtype_size;

    log::info!("chunkCount {}", chunk_count);
    let loop_count = nranks * chunk_count;
    let mut offset;
    let mut nelem;
    let mut chunk;
    let mut tag_chunk;
    let mut tag_offset;
    let step_size = chunk_info.nccl_buff_size / NCCL_STEPS / dtype_size;

    for elem_offset in (0..channel_count).step_by(loop_count as usize) {
        let rem_count = channel_count - elem_offset;

        if rem_count < loop_count {
            chunk_count = align_up(div_up(rem_count, nranks), 16 / dtype_size);
        }

        chunk = ring_ix;
        let mut chunk_offset = chunk * chunk_count;
        offset = grid_offset + elem_offset + chunk_offset;
        nelem = cmp::min(chunk_count, rem_count - chunk_offset);
        tag_chunk = if ring_ix == 0 || ring_ix == nranks - 1 {
            offset
        } else {
            -1
        };

        let mut slice = 0;
        let mut offset_generic_op = 0;

        let mut slice_size = step_size * 2;
        slice_size = cmp::max(div_up(nelem, 16 * 2) * 16, slice_size / 32);
        while slice < 2 && offset_generic_op < nelem {
            slice_size = cmp::min(slice_size, nelem - offset_generic_op);
            tag_offset = if tag_chunk != -1 {
                (tag_chunk + offset_generic_op) * dtype_size
            } else {
                i64::MAX
            };
            offset_generic_op += slice_size;
            slice += 1;
            chunk_info.tag_offsets.push(tag_offset);
            chunk_info.tag_chunks.push(tag_chunk);
            chunk_info.slice_sizes.push(slice_size);
            log::info!(
                "tagChunk: {}, tagOffset = {}, nelem={}, size={}, ringIx {}",
                tag_chunk,
                tag_offset,
                nelem,
                slice_size * dtype_size,
                ring_ix
            );
        }

        // k-2 steps: copy to next GPU
        for j in 1..nranks - 1 {
            chunk = mod_ranks(ring_ix + nranks - j, nranks);
            chunk_offset = chunk * chunk_count;
            offset = grid_offset + elem_offset + chunk_offset;
            nelem = cmp::min(chunk_count, rem_count - chunk_offset);
            tag_chunk = if ring_ix == nranks - 1 { offset } else { -1 };
            slice = 0;
            let mut slice_size = step_size * 2;
            slice_size = cmp::max(div_up(nelem, 16 * 2) * 16, slice_size / 32);
            offset_generic_op = 0;
            while slice < 2 && offset_generic_op < nelem {
                slice_size = cmp::min(slice_size, nelem - offset_generic_op);
                tag_offset = if tag_chunk != -1 {
                    (tag_chunk + offset_generic_op) * dtype_size
                } else {
                    i64::MAX
                };
                offset_generic_op += slice_size;
                slice += 1;
                chunk_info.tag_offsets.push(tag_offset);
                chunk_info.tag_chunks.push(tag_chunk);
                chunk_info.slice_sizes.push(slice_size);
                log::info!(
                    "tagChunk: {}, tagOffset = {}, nelem={}, size={}, ringIx {}",
                    tag_chunk,
                    tag_offset,
                    nelem,
                    slice_size * dtype_size,
                    ring_ix
                );
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct ChunkOffsetInfo {
    ring_ix: i64,
    nranks: i64,
    channel_id: i64,
    count: i64,
    n_channel: i64,
    dtype_size: i64,
    tag_chunks: Vec<i64>,
    tag_offsets: Vec<i64>,
    slice_sizes: Vec<i64>,
    nccl_buff_size: i64,
}

impl ChunkOffsetInfo {
    pub fn new(
        ring_ix: i64,
        nranks: i64,
        channel_id: i64,
        count: i64,
        n_channel: i64,
        dtype_size: i64,
    ) -> ChunkOffsetInfo {
        let mut chunk_info = ChunkOffsetInfo {
            nccl_buff_size: match std::env::var("NCCL_BUFFSIZE") {
                Ok(v) => v.parse::<i64>().unwrap(),
                Err(_) => 1 << 22,
            },
            ..Default::default()
        };

        fill_chunk_info(
            &mut chunk_info,
            ring_ix,
            nranks,
            channel_id,
            count,
            n_channel,
            dtype_size,
        );
        chunk_info
    }

    pub fn get_iter(&self) -> impl Iterator<Item = (&ChunkOffset, &ChunkElems)> {
        self.tag_offsets.iter().zip(self.slice_sizes.iter())
    }
}
