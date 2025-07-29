use libtcp::{ffi::*, tcp_event_register};
use pyo3::prelude::*;
use std::{
    mem::MaybeUninit,
    ptr::{self, slice_from_raw_parts_mut},
    sync::mpsc::Sender,
};

const BATCH_SIZE: usize = 32;

#[pyclass]
pub struct Server {
    nchannels: u16,
    _threads: Vec<std::thread::JoinHandle<()>>,
    req_barrier: std::sync::Arc<std::sync::Barrier>,
    req_sender: Vec<Sender<(u64, usize)>>,
}

#[pymethods]
impl Server {
    #[new]
    pub fn new(base_port: u16, nchannels: u16) -> Self {
        let mut _threads = Vec::new();
        let mut req_sender = Vec::new();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(nchannels as usize + 1));
        unsafe {
            if tpa_init(nchannels as i32) < 0 {
                panic!("tpa_init");
            }
            for id in 0..nchannels {
                let b = barrier.clone();
                let (tx, rx) = std::sync::mpsc::channel();
                let thread = std::thread::spawn(move || {
                    let mut worker = match tpa_worker_init() {
                        ptr if ptr.is_null() => panic!("tpa_worker_init failed"),
                        ptr => Box::from_raw(ptr),
                    };
                    assert!(
                        libtcp::ffi::tpa_listen_on(
                            std::ptr::null(),
                            base_port + id,
                            std::ptr::null()
                        ) >= 0
                    );
                    println!("listening on {}", base_port + id);

                    let mut tcp_writer = TCPWriter::default();
                    while tpa_accept_burst(worker.as_mut(), &mut tcp_writer.fd, 1) == 0 {
                        tpa_worker_run(worker.as_mut());
                    }
                    tcp_event_register(
                        tcp_writer.fd,
                        TPA_EVENT_IN | TPA_EVENT_OUT | TPA_EVENT_ERR | TPA_EVENT_HUP,
                    )
                    .expect("tcp_event_register failed");
                    b.wait();

                    loop {
                        tpa_worker_run(worker.as_mut());
                        if let Ok((base, size)) = rx.try_recv() {
                            let buffer = slice_from_raw_parts_mut(base as u64 as *mut u8, size);
                            let mut offset = 0u64;
                            while offset < size as u64 {
                                let iov_base = buffer.as_mut_ptr().add(offset as usize)
                                    as *mut std::ffi::c_void;
                                let iov_len = std::cmp::min(1048576, size - offset as usize) as u32;
                                tcp_writer
                                    .tcp_write(&mut worker, iov_base, iov_len)
                                    .unwrap();
                                offset += iov_len as u64;
                            }
                            println!("Finished sending {}, {} on {}", base, size, id);
                        }
                    }
                });
                _threads.push(thread);
                req_sender.push(tx);
            }
        }
        Server {
            nchannels,
            _threads,
            req_barrier: barrier.clone(),
            req_sender,
        }
    }

    pub fn send(&self, bucket: &[u8]) -> PyResult<()> {
        self.req_barrier.wait();
        // divide bucket into nchannels and send to each channel
        let chunk_size = bucket.len() / self.nchannels as usize;
        for i in 0..self.nchannels as usize {
            let start = i * chunk_size;
            let end = if i == self.nchannels as usize - 1 {
                bucket.len()
            } else {
                (i + 1) * chunk_size
            };
            let base = bucket[start..end].as_ptr() as u64;
            let size = end - start;
            self.req_sender[i].send((base, size)).unwrap();
        }
        loop {}
    }
}

pub struct TCPWriter {
    fd: i32,
    events: Vec<tpa_event>,
    iov: tpa_iovec,
}

impl Default for TCPWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TCPWriter {
    pub fn new() -> Self {
        let fd = 0;
        let mut uninit = [MaybeUninit::<tpa_event>::uninit(); BATCH_SIZE];
        let events = uninit
            .iter_mut()
            .map(|x| unsafe { x.assume_init() })
            .collect::<Vec<tpa_event>>();

        let uninit = MaybeUninit::<tpa_iovec>::uninit();
        let iov = unsafe { uninit.assume_init() };
        TCPWriter { events, iov, fd }
    }

    pub fn tcp_write(
        &mut self,
        worker: &mut Box<tpa_worker>,
        buf_base: *mut std::ffi::c_void,
        buf_len: u32,
    ) -> Result<isize, std::io::Error> {
        self.iov.iov_base = buf_base;
        self.iov.iov_len = buf_len;
        self.iov.iov_phys = 0;
        self.iov.__bindgen_anon_1.iov_write_done = None;
        self.iov.iov_param = ptr::null_mut();

        loop {
            unsafe { tpa_worker_run(worker.as_mut()) };
            let ret = unsafe { libtcp::ffi::tpa_zwritev(self.fd, &self.iov, 1) };
            if ret < 0 {
                continue;
            }

            unsafe { tpa_event_poll(worker.as_mut(), self.events.as_mut_ptr(), BATCH_SIZE as i32) };
            return Ok(ret);
        }
    }
}

impl Drop for TCPWriter {
    fn drop(&mut self) {
        unsafe { libtcp::ffi::tpa_close(self.fd) };
    }
}
