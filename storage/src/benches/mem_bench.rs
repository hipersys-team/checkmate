// Execute: cargo bench --bench mem_bench

use criterion::black_box;
use network::grad_bucket::parallel_memcpy;
use nix::sched::sched_setaffinity;
use nix::sched::CpuSet;
use nix::unistd::Pid;
use rayon::ThreadPoolBuilder;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use thread_priority::set_current_thread_priority;
use thread_priority::ThreadPriority;

pub fn set_affinity(coreid: usize, priority: ThreadPriority) {
    let mut cpu_set = CpuSet::new();
    cpu_set.set(coreid).unwrap();
    sched_setaffinity(Pid::from_raw(0), &cpu_set).unwrap();
    set_current_thread_priority(priority).unwrap();
}

fn memcpy_comparison(c: &mut Criterion) {
    let available_cores: Vec<usize> = (0..12).collect();
    let memcpy_tpool = ThreadPoolBuilder::new()
        .num_threads(available_cores.len())
        .start_handler(move |thread_index| {
            if let Some(core_id) = available_cores.get(thread_index) {
                set_affinity(*core_id, ThreadPriority::Max);
            }
        })
        .build()
        .expect("Failed to build thread pool");

    (20..31).for_each(|i| {
        let size = 1 << i;
        let src = vec![0u8; size];
        let mut dst = vec![0u8; size];

        // Memcpy latency benchmark
        let mut group = c.benchmark_group(format!("memcpy/{}", size));
        group.bench_with_input("copy_from_slice", &size, |b, _| {
            b.iter(|| {
                black_box(dst.copy_from_slice(src.as_slice()));
            });
        });

        group.bench_with_input("parallel_memcpy", &size, |b, _| {
            b.iter(|| {
                memcpy_tpool.install(|| {
                    black_box(parallel_memcpy(src.as_slice(), dst.as_mut_slice()));
                });
            });
        });
        group.finish();
    });
}

criterion_group!(benches, memcpy_comparison);
criterion_main!(benches);
