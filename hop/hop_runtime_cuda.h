#ifndef __HOP_RUNTIME_CUDA_H__
#define __HOP_RUNTIME_CUDA_H__

#include <cuda_runtime.h>

#define gpuMemcpyKind             cudaMemcpyKind
#define gpuMemcpyDeviceToHost     cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice     cudaMemcpyHostToDevice
#define gpuSuccess                cudaSuccess
#define gpuEventDefault           cudaEventDefault
#define gpuEventBlockingSync      cudaEventBlockingSync
#define gpuEventDisableTiming     cudaEventDisableTiming

#define gpuStream_t               cudaStream_t
#define gpuEvent_t                cudaEvent_t
#define gpuError_t                cudaError_t
#define gpuDeviceProp_t           cudaDeviceProp

#define gpuDoubleComplex          cuDoubleComplex
#define make_gpuDoubleComplex     make_cuDoubleComplex
#define gpuCreal                  cuCreal
#define gpuCimag                  cuCimag
#define gpuCadd                   cuCadd
#define gpuCmul                   cuCmul
#define gpuConj                   cuConj

#define gpuGetLastError           cudaGetLastError
#define gpuGetErrorString         cudaGetErrorString

#define gpuSetDevice              cudaSetDevice
#define gpuGetDevice              cudaGetDevice
#define gpuGetDeviceProperties    cudaGetDeviceProperties
#define gpuDeviceSynchronize      cudaDeviceSynchronize

#define gpuFree                   cudaFree
#define gpuFreeHost               cudaFreeHost
#define gpuMalloc                 cudaMalloc
#define gpuHostMalloc             cudaHostAlloc
#define gpuHostMallocPortable     cudaHostAllocPortable
#define gpuMemcpy                 cudaMemcpy
#define gpuMemcpyAsync            cudaMemcpyAsync

#define gpuStreamCreate           cudaStreamCreate
#define gpuStreamDestroy          cudaStreamDestroy
#define gpuStreamWaitEvent        cudaStreamWaitEvent
#define gpuStreamSynchronize      cudaStreamSynchronize

#define gpuEventCreate            cudaEventCreate
#define gpuEventCreateWithFlags   cudaEventCreateWithFlags
#define gpuEventDestroy           cudaEventDestroy
#define gpuEventQuery             cudaEventQuery
#define gpuEventRecord            cudaEventRecord
#define gpuEventSynchronize       cudaEventSynchronize
#define gpuEventElapsedTime       cudaEventElapsedTime

#define gpuLaunchKernel(kernel, dimGrid, dimBlock, shared, stream, ...) \
        kernel<<<dimGrid, dimBlock, shared, stream>>>(__VA_ARGS__)

#endif
