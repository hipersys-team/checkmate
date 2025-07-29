use clap::{value_t, App, Arg};
use env_logger::Builder;
use network::Server;

use tcmalloc2::TcMalloc;

#[global_allocator]
static GLOBAL: TcMalloc = TcMalloc;

pub const DTYPE_SIZE: i64 = 4;

#[allow(clippy::infinite_iter)]
fn handle_models(args: Args, sizes: Vec<i32>) {
    log::warn!("Handling {} buffers\n", args.name);
    let mut buffers = sizes
        .iter()
        .map(|&s| vec![0u8; s as usize])
        .collect::<Vec<Vec<u8>>>();
    let buffer_sizes = buffers.iter().map(|b| b.len() as i32).collect::<Vec<i32>>();
    let server = Server::new(41000, buffer_sizes, DTYPE_SIZE, args.nnodes, args.node_rank);

    (1..).for_each(|iteration_count| {
        let start = std::time::Instant::now();
        for (id, buffer) in buffers
            .iter_mut()
            .enumerate()
            .filter(|(id, _)| *id as i64 % args.nnodes == args.node_rank)
        {
            let buffer_len = buffer.len() as i32;
            let buffer_base_addr = buffer.as_mut_ptr().addr() as u64;
            server.update_grad_bucket(id, buffer_base_addr, buffer_len);
        }
        println!(
            "Iteration {} finished in {} ms",
            iteration_count,
            start.elapsed().as_millis()
        );
    });
}

fn nccl(args: Args) {
    log::warn!("Handles NCCL AllReduce with 1MB buffers");
    let mut buffers = vec![vec![0u8; 33554432]; 32];
    let buffer_sizes = buffers.iter().map(|b| b.len() as i32).collect::<Vec<i32>>();
    let server = Server::new(41000, buffer_sizes, DTYPE_SIZE, args.nnodes, args.node_rank);

    loop {
        for (id, buffer) in buffers
            .iter_mut()
            .enumerate()
            .filter(|(id, _)| *id as i64 % args.nnodes == args.node_rank)
        {
            let buffer_len = buffer.len() as i32;
            let buffer_base_addr = buffer.as_mut_ptr().addr() as u64;
            server.update_grad_bucket(id, buffer_base_addr, buffer_len);
        }
    }
}

#[derive(Debug)]
struct Args {
    name: String,
    nnodes: i64,
    node_rank: i64,
    _batch_size: i64,
}

fn parse_args() -> Args {
    let args = std::env::args();
    let matches = App::new("Benchmark")
        .about("Benchmark for Network Stack")
        .arg(
            Arg::with_name("bench")
                .short('b')
                .long("bench")
                .possible_values([
                    "resnet50",
                    "vgg11",
                    "vit_h_14",
                    "gpt3_medium",
                    "gpt3_large",
                    "gpt3_xl",
                    "nccl",
                ])
                .default_value("resnet50")
                .help("Name of the benchmark")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("nnodes")
                .short('n')
                .long("nnodes")
                .possible_values(["1", "2", "3", "4"])
                .default_value("1")
                .help("Number of ranks")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("node-rank")
                .short('r')
                .long("node-rank")
                .possible_values(["0", "1", "2", "3"])
                .default_value("0")
                .help("Node rank")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("batch-size")
                .short('s')
                .long("batch-size")
                .default_value("128")
                .help("Batch size")
                .takes_value(true),
        )
        .get_matches_from(args);

    let name = value_t!(matches, "bench", String).unwrap();
    let nnodes = value_t!(matches, "nnodes", i64).unwrap();
    let node_rank = value_t!(matches, "node-rank", i64).unwrap();
    let _batch_size = value_t!(matches, "batch-size", i64).unwrap();
    println!("\n");
    let args = Args {
        name,
        nnodes,
        node_rank,
        _batch_size,
    };
    log::warn!("{:?}", args);
    args
}

fn main() {
    let _ = Builder::from_default_env()
        .format_timestamp(None)
        .format_module_path(false)
        .try_init();
    let args = parse_args();

    let name = args.name.as_str();
    match name {
        "resnet50" => {
            let sizes = [8196000, 31502336, 26255360, 26550272, 9724160];
            handle_models(args, sizes.to_vec());
        }
        "vgg11" => {
            let sizes = [16388000, 67125248, 411058176, 28315648, 8566272];
            handle_models(args, sizes.to_vec());
        }
        "vit_h_14" => {
            let sizes = [
                5124000, 26229760, 26234880, 26245120, 26229760, 26234880, 26245120, 26229760,
                26234880, 26245120, 26229760, 26234880, 26245120, 26229760, 26234880, 26245120,
                26229760, 26234880, 26245120, 26229760, 26234880, 26245120, 26229760, 26234880,
                26245120, 26229760, 26234880, 26245120, 26229760, 26234880, 26245120, 26229760,
                26234880, 26245120, 26229760, 26234880, 26245120, 26229760, 26234880, 26245120,
                26229760, 26234880, 26245120, 26229760, 26234880, 26245120, 26229760, 26234880,
                26245120, 26229760, 26234880, 26245120, 26229760, 26234880, 26245120, 26229760,
                26234880, 26245120, 26229760, 26234880, 26245120, 26229760, 26234880, 26245120,
                26229760, 26234880, 26245120, 26229760, 26234880, 26245120, 26229760, 26234880,
                26245120, 26229760, 26234880, 26245120, 26229760, 26234880, 26245120, 26229760,
                26234880, 26245120, 26229760, 26234880, 26245120, 26229760, 26234880, 26245120,
                26229760, 26234880, 26245120, 26229760, 26234880, 26245120, 26229760, 26234880,
                26245120, 4346880,
            ];
            handle_models(args, sizes.to_vec());
        }
        "gpt3_medium" => {
            let sizes = [
                205852672, 33583104, 37789696, 37793792, 33583104, 37789696, 37793792, 33583104,
                37789696, 37793792, 33583104, 37789696, 37793792, 33583104, 37789696, 37793792,
                33583104, 37789696, 37793792, 33583104, 37789696, 37793792, 33583104, 37789696,
                37793792, 33583104, 37789696, 37793792, 33583104, 37789696, 37793792, 33583104,
                37789696, 37793792, 33583104, 37789696, 37793792, 214249472,
            ];
            handle_models(args, sizes.to_vec());
        }
        "gpt3_large" => {
            let sizes = [
                308779008, 37767168, 37773312, 47228928, 37767168, 37773312, 47228928, 37767168,
                37773312, 47228928, 37767168, 37773312, 47228928, 37767168, 37773312, 47228928,
                37767168, 37773312, 47228928, 37767168, 37773312, 47228928, 37767168, 37773312,
                47228928, 37767168, 37773312, 47228928, 37767168, 37773312, 47228928, 37767168,
                37773312, 47228928, 37767168, 37773312, 47228928, 37767168, 37773312, 47228928,
                37767168, 37773312, 47228928, 37767168, 37773312, 47228928, 37767168, 37773312,
                47228928, 37767168, 37773312, 47228928, 37767168, 37773312, 47228928, 37767168,
                37773312, 47228928, 37767168, 37773312, 47228928, 37767168, 37773312, 47228928,
                37767168, 37773312, 47228928, 37767168, 37773312, 47228928, 37767168, 37773312,
                47228928, 321374208,
            ];
            handle_models(args, sizes.to_vec());
        }
        "gpt3_xl" => {
            let sizes = [
                411705344, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 67133440, 67141632, 33587200, 50356224, 67133440, 67141632, 33587200,
                50356224, 428498944,
            ];
            handle_models(args, sizes.to_vec());
        }
        "nccl" => nccl(args),
        _ => log::error!("Unknown benchmark"),
    }
}
