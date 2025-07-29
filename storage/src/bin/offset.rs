use env_logger::Builder;
use libc::exit;
use network::{chunk_info::ChunkOffsetInfo, NCCL_NCHANNELS};

fn calc_chunkinfo(
    from_rank: i64,
    nranks: i64,
    dtype_size: i64,
    channel_id: i64,
    nelem: i64,
) -> ChunkOffsetInfo {
    ChunkOffsetInfo::new(
        from_rank,
        nranks,
        channel_id,
        nelem,
        NCCL_NCHANNELS,
        dtype_size,
    )
}

fn display_data(rank: i64, chunk_info: &ChunkOffsetInfo, dtype_size: i64) {
    for (c, (offset, size)) in chunk_info.get_iter().enumerate() {
        let offset = *offset as i32;
        if offset >= 0 {
            print!(
                "| R:{}, C:{}, O:{}, S:{} \t",
                rank,
                c,
                offset,
                size * dtype_size
            );
        }
    }
}

fn usage(args0: String) {
    log::error!("Usage: {} <nelem>, <ele_size> <nranks>", args0);
    unsafe { exit(1) };
}

fn main() {
    let _ = Builder::from_default_env()
        .format_timestamp(None)
        .format_module_path(false)
        .try_init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        usage(args[0].clone());
    }
    let nelem = args[1].parse::<i64>().unwrap();
    let dtype_size = args[2].parse::<i64>().unwrap();
    let nranks = args[3].parse::<i64>().unwrap();

    println!(
        "Offset information for GradBucket size {}\n",
        nelem * dtype_size
    );
    for channel_id in 0..NCCL_NCHANNELS {
        let chunk_info_0 = calc_chunkinfo(0, nranks, dtype_size, channel_id, nelem);
        let chunk_info_n = calc_chunkinfo(nranks - 1, nranks, dtype_size, channel_id, nelem);
        print!("Ch {}\t", channel_id);
        display_data(0, &chunk_info_0, dtype_size);
        println!();
        display_data(nranks - 1, &chunk_info_n, dtype_size);
        println!("|\n");
    }
}
