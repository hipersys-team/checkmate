use libtcp::ffi::*;
use pyo3::prelude::*;
use std::{
    mem::MaybeUninit,
    ops::Add,
    ptr::slice_from_raw_parts_mut,
    sync::{mpsc::Sender, Arc},
};

const BATCH_SIZE: usize = 128;

// Event related functions
pub fn tcp_event_register(sid: i32, events: u32) -> Result<(), std::io::Error> {
    let mut uninit = MaybeUninit::<tpa_event>::uninit();
    let event = uninit.as_mut_ptr();
    let mut event = unsafe {
        (*event).events = events;
        (*event).data = sid as *mut std::ffi::c_void;
        uninit.assume_init()
    };

    match unsafe {
        libtcp::ffi::tpa_event_ctrl(sid, libtcp::ffi::TPA_EVENT_CTRL_ADD as i32, &mut event)
    } {
        0 => Ok(()),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "tpa_event_ctrl failed",
        )),
    }
}

pub fn tcp_event_poll(
    worker: &mut Box<tpa_worker>,
    events: &mut [tpa_event],
    maxevents: i32,
) -> i32 {
    unsafe { libtcp::ffi::tpa_worker_run(worker.as_mut()) }
    unsafe { libtcp::ffi::tpa_event_poll(worker.as_mut(), events.as_mut_ptr(), maxevents) }
}

/// # Safety
/// This function is unsafe because it dereferences a raw pointer
pub unsafe fn tcp_connect(
    worker: &mut Box<tpa_worker>,
    server_ip: &str,
    port: u16,
) -> Result<i32, std::io::Error> {
    let fd = tpa_connect_to(
        server_ip.as_bytes().as_ptr() as *const i8,
        port,
        std::ptr::null(),
    );
    tcp_event_register(fd, TPA_EVENT_OUT | TPA_EVENT_ERR | TPA_EVENT_HUP)
        .expect("tcp_event_register failed");
    let mut uninit = [MaybeUninit::<tpa_event>::uninit(); 1];
    let mut events = uninit
        .iter_mut()
        .map(|x| unsafe { x.assume_init() })
        .collect::<Vec<tpa_event>>();
    loop {
        match tcp_event_poll(worker, &mut events, 1) {
            0 => {}
            _ => {
                if events[0].events & TPA_EVENT_OUT != 0 {
                    break;
                }
            }
        }
    }
    Ok(fd)
}

#[pyclass]
pub struct Client {
    nchannels: u16,
    _threads: Vec<Arc<std::thread::JoinHandle<()>>>,
    req_sender: Vec<Sender<(u64, usize)>>,
    exit_barrier: std::sync::Arc<std::sync::Barrier>,
}

#[pymethods]
impl Client {
    #[new]
    pub fn new(base_port: u16, nchannels: u16) -> Self {
        let mut _threads = Vec::new();
        let mut req_sender = Vec::new();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(nchannels as usize + 1));
        let exit_barrier = std::sync::Arc::new(std::sync::Barrier::new(nchannels as usize + 1));

        unsafe {
            if tpa_init(nchannels as i32) < 0 {
                panic!("tpa_init");
            }
            println!("nchannels: {}", nchannels);

            for id in 0..nchannels {
                let b = barrier.clone();
                let exit_b = exit_barrier.clone();

                let (tx, rx) = std::sync::mpsc::channel();
                let thread = std::thread::spawn(move || {
                    let mut worker = match tpa_worker_init() {
                        ptr if ptr.is_null() => panic!("tpa_worker_init failed"),
                        ptr => Box::from_raw(ptr),
                    };
                    let mut tcp_reader = TCPReader::default();
                    let server_ip = String::from("192.168.10.24");
                    tcp_reader.sid = tcp_connect(&mut worker, &server_ip, base_port + id).unwrap();
                    assert!(tcp_reader.sid >= 0);
                    b.wait();
                    println!("Thread {} connected to {}", id, base_port + id);

                    loop {
                        tpa_worker_run(worker.as_mut());
                        if let Ok((base, size)) = rx.try_recv() {
                            let buffer = slice_from_raw_parts_mut(base as *mut u8, size)
                                .as_mut()
                                .unwrap();
                            tcp_reader.process_request(buffer, worker.as_mut());
                            exit_b.wait();
                            break;
                        }
                    }
                });
                _threads.push(Arc::new(thread));
                req_sender.push(tx);
            }
        }
        barrier.wait();
        Client {
            nchannels,
            _threads,
            req_sender,
            exit_barrier: exit_barrier.clone(),
        }
    }

    pub fn receive(&self, base: u64, size: usize) {
        let chunk_size = size / self.nchannels as usize;
        for i in 0..self.nchannels as usize {
            let start = i * chunk_size;
            let end = if i == self.nchannels as usize - 1 {
                size
            } else {
                (i + 1) * chunk_size
            };
            let chunk_base = base.add(start as u64);
            let buflen = end - start;
            self.req_sender[i].send((chunk_base, buflen)).unwrap();
        }
        self.exit_barrier.wait();
        println!(":: received data");
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

impl Default for TCPReader {
    fn default() -> Self {
        Self::new()
    }
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
                //println!("{} remaining {} bytes", self.sid, buf.len());
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
