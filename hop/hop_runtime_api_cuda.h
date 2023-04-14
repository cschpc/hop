/* HOP: Header Only Porting

Copyright (c) 2023 Martti Louhivuori
                   CSC - IT Center for Science Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef __HOP_RUNTIME_API_CUDA_H__
#define __HOP_RUNTIME_API_CUDA_H__

#include <cuda_runtime_api.h>

#define gpuGetLastError           cudaGetLastError
#define gpuGetErrorString         cudaGetErrorString

#define gpuSetDevice              cudaSetDevice
#define gpuGetDevice              cudaGetDevice
#define gpuGetDeviceCount         cudaGetDeviceCount
#define gpuGetDeviceProperties    cudaGetDeviceProperties
#define gpuDeviceSynchronize      cudaDeviceSynchronize
#define gpuDeviceReset            cudaDeviceReset

#define gpuFree                   cudaFree
#define gpuFreeHost               cudaFreeHost
#define gpuFreeAsync              cudaFreeAsync
#define gpuMalloc                 cudaMalloc
#define gpuMallocAsync            cudaMallocAsync
#define gpuHostMalloc             cudaHostAlloc
#define gpuHostMallocPortable     cudaHostAllocPortable
#define gpuMemcpy                 cudaMemcpy
#define gpuMemcpyAsync            cudaMemcpyAsync
#define gpuMemset                 cudaMemset
#define gpuMemsetAsync            cudaMemsetAsync

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

/* driver_types */
#define gpuError_t                cudaError_t
#define gpuSuccess                cudaSuccess

#define gpuStream_t               cudaStream_t
#define gpuDeviceProp_t           cudaDeviceProp

#define gpuEvent_t                cudaEvent_t
#define gpuEventDefault           cudaEventDefault
#define gpuEventBlockingSync      cudaEventBlockingSync
#define gpuEventDisableTiming     cudaEventDisableTiming

#define gpuMemcpyKind             cudaMemcpyKind
#define gpuMemcpyDeviceToHost     cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice     cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice   cudaMemcpyDeviceToDevice

#endif
