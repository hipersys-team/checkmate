#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>

#define CHECK_CUDA(cmd) do {                                 \
  cudaError_t e = cmd;                                       \
  if( e != cudaSuccess ) {                                   \
    printf("Failed: Cuda error %s:%d '%s'\n",                \
        __FILE__,__LINE__,cudaGetErrorString(e));            \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
} while(0)

#define CHECK_NCCL(cmd) do {                                 \
  ncclResult_t r = cmd;                                      \
  if (r!= ncclSuccess) {                                     \
    printf("Failed, NCCL error %s:%d '%s'\n",                \
        __FILE__,__LINE__,ncclGetErrorString(r));            \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
} while(0)

// Function to perform memory copy between two GPUs
void memcopy_between_gpus(int src_device, int dst_device, float* src_data, float* dst_data, size_t size) {
    CHECK_CUDA(cudaSetDevice(src_device));
    CHECK_CUDA(cudaMemcpyPeerAsync(dst_data, dst_device, src_data, src_device, size));
}

// Function to send data to another GPU on a second machine using NCCL
void send_data_to_another_machine(float* send_buffer, size_t size, ncclComm_t comm, cudaStream_t stream) {
    CHECK_NCCL(ncclSend(send_buffer, size, ncclFloat, 1, comm, stream)); // assuming rank 1 is the receiver
}

void recv_data_from_another_machine(float* recv_buffer, size_t size, ncclComm_t comm, cudaStream_t stream) {
  CHECK_NCCL(ncclRecv(recv_buffer, size, ncclFloat, 0, comm, stream)); // assuming rank 1 is the receiver
}

int main(int argc, char* argv[]) {
    const unsigned long size = 5UL * 1024 * 1024 * 1024; // Size of the data 

    int local_rank = 0; 
    int world_size = 3;
    int src_device = 0; // Source GPU on the local machine
    int dst_device = 1; // Destination GPU on the local machine

    // Initialize NCCL
    ncclComm_t comm;
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, local_rank)); // Initialize the NCCL communicator

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Allocate memory on the source GPU
    // float* src_data;
    // CHECK_CUDA(cudaSetDevice(src_device));
    // CHECK_CUDA(cudaMalloc(&src_data, size * sizeof(float)));

    // // Initialize source data
    // CHECK_CUDA(cudaMemset(src_data, 1, size * sizeof(float)));

    // Allocate memory on the destination GPU
    float* dst_data;
    // CHECK_CUDA(cudaSetDevice(dst_device));
    CHECK_CUDA(cudaMalloc(&dst_data, size * sizeof(float)));

    // Perform memory copy between GPUs
    // for (int i = 0; i < 10000; i++) {
    //   memcopy_between_gpus(src_device, dst_device, src_data, dst_data, size * sizeof(float));
    //   CHECK_CUDA(cudaDeviceSynchronize());
    // }

    // Send data to another machine
    recv_data_from_another_machine(dst_data, size, comm, stream);

    // Clean up
    // CHECK_CUDA(cudaFree(src_data));
    // CHECK_CUDA(cudaFree(dst_data));
    CHECK_CUDA(cudaStreamDestroy(stream));
    // CHECK_NCCL(ncclCommDestroy(comm));

    std::cout << "Program completed successfully." << std::endl;

    return 0;
}

