# maturin develop --release

import network
import threading
import torch
import signal


def test():
    num_buckets = 4
    gradient_size = [104857600] * num_buckets  # 4 buckets of 100MB
    gradients = [
        torch.zeros([int(size / 4)], dtype=torch.float) for size in gradient_size
    ]
    server = network.Server(41000, gradient_size, gradients[0].element_size(), 1, 0)
    while True:
        for index in range(num_buckets):
            server.update_grad_bucket(
                index, gradients[index].data_ptr(), gradient_size[index]
            )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    t = threading.Thread(target=test, args=[])
    t.daemon = True
    t.start()
    while t.is_alive():  # wait for the thread to exit
        t.join(0.1)
