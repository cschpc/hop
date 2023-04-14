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

#ifndef __HOP_SOURCE_CUDA_RUNTIME_API_H__
#define __HOP_SOURCE_CUDA_RUNTIME_API_H__

#define cudaGetLastError          gpuGetLastError
#define cudaGetErrorString        gpuGetErrorString

#define cudaSetDevice             gpuSetDevice
#define cudaGetDevice             gpuGetDevice
#define cudaGetDeviceCount        gpuGetDeviceCount
#define cudaGetDeviceProperties   gpuGetDeviceProperties
#define cudaDeviceSynchronize     gpuDeviceSynchronize
#define cudaDeviceReset           gpuDeviceReset

#define cudaFree                  gpuFree
#define cudaFreeHost              gpuFreeHost
#define cudaFreeAsync             gpuFreeAsync
#define cudaMalloc                gpuMalloc
#define cudaMallocAsync           gpuMallocAsync
#define cudaHostAlloc             gpuHostMalloc
#define cudaHostAllocPortable     gpuHostMallocPortable
#define cudaMemcpy                gpuMemcpy
#define cudaMemcpyAsync           gpuMemcpyAsync
#define cudaMemset                gpuMemset
#define cudaMemsetAsync           gpuMemsetAsync

#define cudaStreamCreate          gpuStreamCreate
#define cudaStreamDestroy         gpuStreamDestroy
#define cudaStreamWaitEvent       gpuStreamWaitEvent
#define cudaStreamSynchronize     gpuStreamSynchronize

#define cudaEventCreate           gpuEventCreate
#define cudaEventCreateWithFlags  gpuEventCreateWithFlags
#define cudaEventDestroy          gpuEventDestroy
#define cudaEventQuery            gpuEventQuery
#define cudaEventRecord           gpuEventRecord
#define cudaEventSynchronize      gpuEventSynchronize
#define cudaEventElapsedTime      gpuEventElapsedTime

/* driver_types */
#define cudaError_t               gpuError_t
#define cudaSuccess               gpuSuccess

#define cudaStream_t              gpuStream_t
#define cudaDeviceProp            gpuDeviceProp_t

#define cudaEvent_t               gpuEvent_t
#define cudaEventDefault          gpuEventDefault
#define cudaEventBlockingSync     gpuEventBlockingSync
#define cudaEventDisableTiming    gpuEventDisableTiming

#define cudaMemcpyKind            gpuMemcpyKind
#define cudaMemcpyDeviceToHost    gpuMemcpyDeviceToHost
#define cudaMemcpyHostToDevice    gpuMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice  gpuMemcpyDeviceToDevice

#include <hop/hop_runtime_api.h>

#endif
