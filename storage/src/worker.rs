#![allow(dead_code)]

use crate::NCCL_NCHANNELS;
#[cfg(feature = "switch")]
use libtcp::ffi::tpa_connect_to;
use libtcp::ffi::{
    tpa_accept_burst, tpa_close, tpa_event, tpa_event_poll, tpa_iovec, tpa_ip, tpa_sock_info,
    tpa_sock_info_get, tpa_thread_register, tpa_worker, tpa_worker_init, tpa_worker_run,
    tpa_zreadv, TPA_EVENT_IN,
};
use nix::sched::{sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;
use std::ops::Add;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use thread_priority::*;

pub const BATCH_SIZE: usize = 128;

#[derive(Serialize, Deserialize, Debug)]
pub struct ChunkInfo {
    pub chunk_offset: i32,
    pub chunk_size: i32,
}

pub struct TCPWorker {
    pub req_sender: Sender<(&'static mut [u8], Arc<ChunkInfo>)>,
    pub resp_receiver: Receiver<usize>,
    thread: Arc<thread::JoinHandle<()>>,

    pub rank: usize,
    pub channel_id: usize,
}

unsafe impl Send for TCPWorker {}
unsafe impl Sync for TCPWorker {}

#[cfg(feature = "switch")]
unsafe fn connect(switch_ip: &str, switch_port: u16) -> i32 {
    println!("Connecting to {} on remote port {}", switch_ip, switch_port);
    tpa_connect_to(
        switch_ip.as_bytes().as_ptr() as *const i8,
        switch_port,
        std::ptr::null(),
    )
}

pub unsafe fn start_worker(
    rank: usize,
    channel_id: usize,
    base_port: u16,
    barrier: Arc<std::sync::Barrier>,
) -> TCPWorker {
    let (req_sender, req_receiver) = channel::<(&'static mut [u8], Arc<ChunkInfo>)>();
    let (resp_sender, resp_receiver) = channel();

    // Rank 0 connects to 0004.. Rank 1 connects to 0000..
    let port = base_port + (((rank + 1) % 2) as u16 * NCCL_NCHANNELS as u16) + channel_id as u16;

    let b = barrier.clone();
    let thread = Arc::new(thread::spawn(move || {
        set_affinity(
            (NCCL_NCHANNELS as usize * rank) + channel_id + 1,
            ThreadPriority::Max,
        );
        let mut worker = match tpa_worker_init() {
            ptr if ptr.is_null() => panic!("tpa_worker_init failed"),
            ptr => Box::from_raw(ptr),
        };
        assert!(libtcp::ffi::tpa_listen_on(std::ptr::null(), port, std::ptr::null()) >= 0);

        #[cfg(feature = "switch")]
        let switch_fd = connect(&String::from("192.168.10.245"), port);

        let mut _ctrl_reader = TCPReader::new();
        #[cfg(not(feature = "switch"))]
        {
            while tpa_accept_burst(worker.as_mut(), &mut _ctrl_reader.sid, 1) == 0 {
                tpa_worker_run(worker.as_mut());
            }
            register_connection(_ctrl_reader.sid).unwrap();
        }

        let mut tcp_reader = TCPReader::new();
        while tpa_accept_burst(worker.as_mut(), &mut tcp_reader.sid, 1) == 0 {
            tpa_worker_run(worker.as_mut());
        }
        register_connection(tcp_reader.sid).unwrap();
        b.wait();
        println!(":: connection established");

        #[cfg(feature = "switch")]
        tpa_close(switch_fd);

        #[cfg(not(feature = "switch"))]
        let mut chunk_info = ChunkInfo {
            chunk_offset: 0,
            chunk_size: 0,
        };

        loop {
            #[cfg(not(feature = "switch"))]
            {
                let mut chunk_bytes = bincode::serialize(&chunk_info).unwrap();
                _ctrl_reader.process_request(&mut chunk_bytes, worker.as_mut());
                chunk_info = bincode::deserialize::<ChunkInfo>(&chunk_bytes).unwrap();
                if chunk_info.chunk_size == 0 {
                    continue;
                }
            }

            loop {
                tpa_worker_run(worker.as_mut());
                if let Ok((buf, req_chunk_info)) = req_receiver.try_recv() {
                    #[cfg(not(feature = "switch"))]
                    {
                        log::info!("Chunk: {:?}, Requested {:?}", chunk_info, req_chunk_info);
                        assert!(chunk_info.chunk_size == req_chunk_info.chunk_size);
                        assert!(chunk_info.chunk_offset == req_chunk_info.chunk_offset);
                    }

                    let read_bytes = tcp_reader.process_request(buf, worker.as_mut());
                    assert!(read_bytes == req_chunk_info.chunk_size as usize);
                    resp_sender.send(read_bytes).unwrap();
                    break;
                }
            }
        }
    }));
    TCPWorker {
        req_sender,
        resp_receiver,
        thread,
        rank,
        channel_id,
    }
}

pub fn set_affinity(coreid: usize, priority: ThreadPriority) {
    let mut cpu_set = CpuSet::new();
    cpu_set.set(coreid).unwrap();
    sched_setaffinity(Pid::from_raw(0), &cpu_set).unwrap();
    set_current_thread_priority(priority).unwrap();
    unsafe { assert!(tpa_thread_register() == 0) };
}

pub fn tpa_ip_to_str(ip: tpa_ip) -> String {
    let ip_bytes = unsafe { ip.__bindgen_anon_1.u32_[3].to_be_bytes() };
    let ip_addr = std::net::Ipv4Addr::new(ip_bytes[3], ip_bytes[2], ip_bytes[1], ip_bytes[0]);
    ip_addr.to_string()
}

pub fn tcp_socket_info_get(sid: i32) -> Result<tpa_sock_info, std::io::Error> {
    let mut uninit = MaybeUninit::<tpa_sock_info>::uninit();
    let info = uninit.as_mut_ptr();
    match unsafe { tpa_sock_info_get(sid, info) } {
        0 => Ok(unsafe { uninit.assume_init() }),
        _ => Err(std::io::Error::other("tpa_sock_info_get failed")),
    }
}

fn register_connection(sid: i32) -> Result<(), std::io::Error> {
    let mut uninit = MaybeUninit::<tpa_event>::uninit();
    let event = uninit.as_mut_ptr();
    let mut event = unsafe {
        (*event).events = TPA_EVENT_IN;
        (*event).data = sid as *mut std::ffi::c_void;
        uninit.assume_init()
    };
    let _info = tcp_socket_info_get(sid).expect("tcp_socket_info_get failed");

    println!(
        "Local IP {}, remote IP {}",
        tpa_ip_to_str(_info.local_ip),
        tpa_ip_to_str(_info.remote_ip),
    );

    match unsafe {
        libtcp::ffi::tpa_event_ctrl(sid, libtcp::ffi::TPA_EVENT_CTRL_ADD as i32, &mut event)
    } {
        0 => Ok(()),
        _ => Err(std::io::Error::other("tpa_event_ctrl failed")),
    }
}

pub struct TCPReader {
    sid: i32,

    events: Vec<tpa_event>,

    iovs: Vec<tpa_iovec>,
    iov_idx: usize,
    iov_read_bytes: isize,
    current_iov_used: usize,
}

impl TCPReader {
    pub fn new() -> Self {
        let mut uninit = [MaybeUninit::<tpa_event>::uninit(); BATCH_SIZE];
        let events = uninit
            .iter_mut()
            .map(|x| unsafe { x.assume_init() })
            .collect::<Vec<tpa_event>>();

        let mut uninit = [MaybeUninit::<tpa_iovec>::uninit(); BATCH_SIZE];
        let iovs = uninit
            .iter_mut()
            .map(|x| unsafe { x.assume_init() })
            .collect::<Vec<tpa_iovec>>();

        TCPReader {
            sid: 0,
            events,
            iovs,
            iov_idx: 0,
            iov_read_bytes: 0,
            current_iov_used: 0,
        }
    }

    #[inline]
    fn poll_connection(&mut self, worker: &mut tpa_worker) {
        let events = self.events.as_mut_slice().as_mut_ptr();

        unsafe {
            tpa_worker_run(worker);
            tpa_event_poll(worker, events, BATCH_SIZE as i32);
        }
    }

    #[inline]
    fn poll_and_read(&mut self, worker: &mut tpa_worker) {
        self.poll_connection(worker);
        let bytes_read = unsafe { tpa_zreadv(self.sid, self.iovs.as_mut_ptr(), BATCH_SIZE as i32) };
        if bytes_read >= 0 {
            self.iov_read_bytes = bytes_read;
            self.iov_idx = 0;
        }
    }

    // TODO: Check indexing and bounds
    fn copy_iov_data(&mut self, buf: &mut [u8]) -> usize {
        unsafe {
            let iov = self.iovs[self.iov_idx];
            let iov_remaining = iov.iov_len as usize - self.current_iov_used;

            let copied = std::cmp::min(iov_remaining, buf.len());
            let base = (iov.iov_base as u64).add(self.current_iov_used as u64);
            std::ptr::copy_nonoverlapping(base as *const u8, buf.as_mut_ptr(), copied);

            self.current_iov_used += copied;
            self.iov_read_bytes -= copied as isize;
            if self.current_iov_used == iov.iov_len as usize {
                self.current_iov_used = 0;
                self.iov_idx += 1;
                if let Some(iov_read_done) = iov.__bindgen_anon_1.iov_read_done {
                    iov_read_done(iov.iov_base, iov.iov_param);
                }
            }
            copied
        }
    }

    #[inline]
    pub fn process_request(&mut self, buf: &mut [u8], worker: &mut tpa_worker) -> usize {
        let size = buf.len();
        let mut buf = buf;
        while !buf.is_empty() {
            if self.iov_read_bytes == 0 {
                self.poll_and_read(worker);
            } else {
                let copied = self.copy_iov_data(buf);
                buf = &mut buf[copied..];
            }
        }
        size
    }
}

impl Drop for TCPReader {
    fn drop(&mut self) {
        unsafe { tpa_close(self.sid) };
    }
}
