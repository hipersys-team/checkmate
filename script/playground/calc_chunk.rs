use std::ops::Add;
use std::env;
use std::cmp;

// // Generic function to compute the ceiling of integer division
// fn div_up<X, Y, Z>(x: X, y: Y) -> Z
// where
//     X: Add<Y, Output = Z> + Copy,
//     Y: Copy,
//     Z: From<X> + std::ops::Div<Y, Output = Z> + std::ops::Sub<Z, Output = Z> + std::ops::Add<Y>
// {
//     // (Z::from(x) + y - Z::from(1.into())) / y
//     (x+y-1)/y 
// }

// Generic function to compute the ceiling of integer division
fn div_up(x: i64, y: i64) -> i64
{
    (x+y-1)/y 
}


// Function to align size to the nearest higher multiple of alignment
fn align_up(size: i64, alignment: i64) -> i64 {
    (size + alignment - 1) & !(alignment - 1)
}

// Define the CBD structure
#[derive(Default, Debug)]
struct CBD {
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
    cbd: CBD,
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
        *part_count = work.cbd.count_lo.into();
        *chunk_count = (work.cbd.chunk_grains_lo * elt_per_grain).into();
    } else if channel_id == work.channel_hi {
        *part_offset = (work.cbd.count_lo + n_mid_channels * work.cbd.count_mid).into();
        *part_count = work.cbd.count_hi.into();
        *chunk_count = (work.cbd.chunk_grains_hi * elt_per_grain).into();
    } else {
        let mid = channel_id - work.channel_lo - 1;
        *part_offset = (work.cbd.count_lo + mid * work.cbd.count_mid).into();
        *part_count = work.cbd.count_mid.into();
        *chunk_count = (work.cbd.chunk_grains_mid * elt_per_grain).into();
    }
}

// Rust function equivalent to scheduleCollTasksToPlan
fn schedule_coll_tasks_to_plan(n_channels: i64, datatype_size: i64, count: i64, dev_work: &mut NcclDevWorkColl) {
    const MIN_TRAFFIC_PER_CHANNEL: i64 = 512;
    let element_size = datatype_size;
    let traffic_bytes = count * element_size * 2;

    let mut traffic_per_channel = std::cmp::max(MIN_TRAFFIC_PER_CHANNEL, traffic_bytes / n_channels as i64);
    let mut channel_id = 0;
    let mut current_traffic = 0;
    const CELL_SIZE: i64 = 16;
    let elements_per_cell = CELL_SIZE / element_size;
    let cells = div_up(count * element_size, CELL_SIZE);
    let traffic_per_byte = 2;
    let traffic_per_element = element_size * traffic_per_byte;
    let traffic_per_cell = CELL_SIZE * traffic_per_byte;
    let mut cells_per_channel = std::cmp::min(cells, div_up(traffic_per_channel, traffic_per_cell));
    let mut cells_lo = if channel_id + 1 == n_channels {
        // On last channel, everything goes to "lo"
        cells
    } else {
        std::cmp::min(cells, (traffic_per_channel - current_traffic) / traffic_per_cell)
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
    let count_mid = if n_mid_channels != 0 { cells_per_channel * elements_per_cell } else { 0 };
    let count_lo = cells_lo * elements_per_cell;
    let mut count_hi = cells_hi * elements_per_cell;
    count_hi = if count_hi != 0 { count_hi } else { count_lo };
    count_hi -= cells * elements_per_cell - count;

    let n_channels = ((count_lo != 0) as i64) + n_mid_channels + ((cells_hi != 0) as i64);

    dev_work.channel_lo = channel_id as i64;
    dev_work.channel_hi = (channel_id + n_channels - 1) as i64;
    dev_work.cbd.count_lo = count_lo as i64;
    dev_work.cbd.count_mid = count_mid as i64;
    dev_work.cbd.count_hi = count_hi as i64;

    let grain_size = 512;
    if count_lo != 0 {
        dev_work.cbd.chunk_grains_lo = 2097152 / grain_size;
    }
    if count_hi != 0 {
        dev_work.cbd.chunk_grains_hi = 2097152 / grain_size;
    }
    if n_mid_channels != 0 {
        dev_work.cbd.chunk_grains_mid = 2097152 / grain_size;
    }

    eprintln!("Collective AllReduce(elemsize: {}, Ring, Simple) count={} channel{{Lo..Hi}}={{{}..{}}} count{{Lo,Mid,Hi}}={{{},{},{}}}, chunkBytes{{Lo,Mid,Hi}}={{{},{},{}}}",
        datatype_size, 
        count, dev_work.channel_lo, dev_work.channel_hi,
        dev_work.cbd.count_lo, dev_work.cbd.count_mid, dev_work.cbd.count_hi,
        dev_work.cbd.chunk_grains_lo * 512,
        dev_work.cbd.chunk_grains_mid * 512,
        dev_work.cbd.chunk_grains_hi * 512);
}


struct ChunkInfo {
    ring_ix: i64,
    nranks: i64,
    channel_id: i64,
    count: i64,
    n_channel: i64,
    dtype_size: i64,
    tag_chunks: Vec<i64>,
    tag_offsets: Vec<i64>,
    slice_sizes: Vec<i64>,
}

// Function to perform modulo operation on ranks
fn mod_ranks(r: i64, nranks: i64) -> i64 {
    if r >= nranks { r - nranks } else { r }
}

// Implement fill_chunk_info function
fn fill_chunk_info(chunk_info: &mut ChunkInfo, ring_ix: i64, nranks: i64, channel_id: i64, count: i64, n_channel: i64, dtype_size: i64) {
    let mut grid_offset: i64 = 0;
    let mut channel_count: i64 = 0;
    let mut chunk_count: i64 = 0;

    let mut work = NcclDevWorkColl::default();
    schedule_coll_tasks_to_plan(n_channel, dtype_size, count, &mut work);
    nccl_coll_cbd_part(&work, channel_id as i64, dtype_size as i64, &mut grid_offset, &mut channel_count, &mut chunk_count);

    chunk_info.ring_ix = ring_ix;
    chunk_info.nranks = nranks;
    chunk_info.channel_id = channel_id;
    chunk_info.count = count;
    chunk_info.n_channel = n_channel;
    chunk_info.dtype_size = dtype_size;

    eprintln!("chunkCount {}", chunk_count);
    let loop_count = nranks as i64 * chunk_count;
    let mut offset;
    let mut nelem;
    let mut chunk;
    let mut tag_chunk;
    let mut tag_offset;
    let mut offset_generic_op = 0;
    let step_size = (1 << 22) / 8 / dtype_size;
    
    let mut slice = 0;

    for elem_offset in (0..channel_count).step_by(loop_count as usize) {
        let rem_count = channel_count - elem_offset;
        let mut chunk_offset = 0;

        if rem_count < loop_count {
            chunk_count = align_up(div_up(rem_count, nranks as i64), 16 / dtype_size);
        }

        chunk = ring_ix + 0;
        chunk_offset = chunk * chunk_count as i64;
        offset = grid_offset + elem_offset + chunk_offset as i64;
        nelem = cmp::min(chunk_count as i64, (rem_count - chunk_offset as i64) as i64);
        tag_chunk = if ring_ix == 0 || ring_ix == nranks - 1 { offset as i64 } else { -1 };
        slice = 0;
        let mut slice_size = step_size * 2;
        slice_size = cmp::max(div_up(nelem, 16 * 2) * 16, slice_size / 32);
        offset_generic_op = 0;
        while slice < 2 && offset_generic_op < nelem {
            slice_size = cmp::min(slice_size, nelem - offset_generic_op);
            tag_offset = if tag_chunk != -1 { ((tag_chunk + offset_generic_op) * dtype_size) as i64 } else { i64::MAX };
            offset_generic_op += slice_size;
            slice += 1;
            chunk_info.tag_offsets.push(tag_offset);
            chunk_info.tag_chunks.push(tag_chunk);
            chunk_info.slice_sizes.push(slice_size);
            eprintln!("tagChunk: {}, tagOffset = {}, nelem={}, size={}, ringIx {}", tag_chunk, tag_offset, nelem, slice_size * dtype_size, ring_ix);
        }

        // k-2 steps: copy to next GPU
        for j in 1..nranks - 1 {
            chunk = mod_ranks(ring_ix + nranks - j, nranks);
            chunk_offset = chunk * chunk_count as i64;
            offset = grid_offset + elem_offset + chunk_offset as i64;
            nelem = cmp::min(chunk_count as i64, (rem_count - chunk_offset as i64) as i64);
            tag_chunk = if ring_ix == nranks - 1 { offset as i64 } else { -1 };
            slice = 0;
            let mut slice_size = step_size * 2;
            slice_size = cmp::max(div_up(nelem, 16 * 2) * 16, slice_size / 32);
            offset_generic_op = 0;
            while slice < 2 && offset_generic_op < nelem {
                slice_size = cmp::min(slice_size, nelem - offset_generic_op);
                tag_offset = if tag_chunk != -1 { ((tag_chunk + offset_generic_op) * dtype_size) as i64 } else { i64::MAX };
                offset_generic_op += slice_size;
                slice += 1;
                chunk_info.tag_offsets.push(tag_offset);
                chunk_info.tag_chunks.push(tag_chunk);
                chunk_info.slice_sizes.push(slice_size);
                eprintln!("tagChunk: {}, tagOffset = {}, nelem={}, size={}, ringIx {}", tag_chunk, tag_offset, nelem, slice_size * dtype_size, ring_ix);
            }
        }
    }
}


fn main() {
    let args: Vec<String> = env::args().collect();

    // Check if all required arguments are passed
    if args.len() != 7 {
        eprintln!("Usage: {} <ring_ix> <nranks> <channel_id> <count> <n_channel> <dtype_size>", args[0]);
        std::process::exit(1);
    }

    let ring_ix = args[1].parse::<i64>().expect("Error parsing ring_ix");
    let nranks = args[2].parse::<i64>().expect("Error parsing nranks");
    let channel_id = args[3].parse::<i64>().expect("Error parsing channel_id");
    let count = args[4].parse::<i64>().expect("Error parsing count");
    let n_channel = args[5].parse::<i64>().expect("Error parsing n_channel");
    let dtype_size = args[6].parse::<i64>().expect("Error parsing dtype_size");

    let mut chunk_info = ChunkInfo {
        ring_ix: 0,
        nranks: 0,
        channel_id: 0,
        count: 0,
        n_channel: 0,
        dtype_size: 0,
        tag_chunks: Vec::new(),
        tag_offsets: Vec::new(),
        slice_sizes: Vec::new(),
    };

    fill_chunk_info(&mut chunk_info, ring_ix, nranks, channel_id, count, n_channel, dtype_size);
}


