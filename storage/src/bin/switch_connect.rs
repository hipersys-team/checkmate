use std::mem::MaybeUninit;

use libtcp::ffi::{
    tpa_accept_burst, tpa_close, tpa_connect_to, tpa_event, tpa_init, tpa_listen_on, tpa_worker,
    tpa_worker_init, tpa_worker_run, TPA_EVENT_ERR, TPA_EVENT_HUP, TPA_EVENT_IN, TPA_EVENT_OUT,
};
use network::NUM_CONNECTIONS;

fn register_connection(sid: i32) -> Result<(), std::io::Error> {
    let mut uninit = MaybeUninit::<tpa_event>::uninit();
    let event = uninit.as_mut_ptr();
    let mut event = unsafe {
        (*event).events = TPA_EVENT_IN | TPA_EVENT_OUT | TPA_EVENT_ERR | TPA_EVENT_HUP;
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

unsafe fn connect(switch_ip: &str, switch_port: u16) {
    println!("Connecting to {} on port {}", switch_ip, switch_port);
    let fd = tpa_connect_to(
        switch_ip.as_bytes().as_ptr() as *const i8,
        switch_port,
        std::ptr::null(),
    );
    register_connection(fd).unwrap();
}

unsafe fn start_worker(worker: &mut Box<tpa_worker>, switch_ip: String, switch_port: u16) {
    let mut sid = 0;
    assert!(tpa_listen_on(std::ptr::null(), switch_port, std::ptr::null()) >= 0);
    connect(&switch_ip, switch_port);

    while tpa_accept_burst(worker.as_mut(), &mut sid, 1) == 0 {
        tpa_worker_run(worker.as_mut());
    }
    println!("Connection established!!!");
    register_connection(sid).unwrap();
    tpa_close(sid);
}

fn main() {
    let switch_ip = String::from("192.168.10.245");
    let switch_port = 41000;

    if std::fs::metadata("tpa.cfg").is_err() {
        panic!("tpa.cfg not found");
    }

    unsafe {
        if tpa_init(NUM_CONNECTIONS as i32) < 0 {
            panic!("tpa_init");
        }
        let mut worker = match tpa_worker_init() {
            ptr if ptr.is_null() => panic!("tpa_worker_init failed"),
            ptr => Box::from_raw(ptr),
        };
        for i in 0..NUM_CONNECTIONS {
            println!();
            start_worker(&mut worker, switch_ip.clone(), switch_port + i as u16);
        }
    }
}
