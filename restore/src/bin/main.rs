use restore::Client;

fn main() {
    let buffer = vec![0u8; 2528387810];
    let client = Client::new(5678, 4);
    client.receive(buffer.as_ptr() as u64, 2528387810);
}
