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

#ifndef __HOP_RUNTIME_API_HIP_H__
#define __HOP_RUNTIME_API_HIP_H__

#include <hip/hip_runtime_api.h>

#define gpuGetLastError           hipGetLastError
#define gpuGetErrorString         hipGetErrorString

#define gpuSetDevice              hipSetDevice
#define gpuGetDevice              hipGetDevice
#define gpuGetDeviceCount         hipGetDeviceCount
#define gpuGetDeviceProperties    hipGetDeviceProperties
#define gpuDeviceSynchronize      hipDeviceSynchronize
#define gpuDeviceReset            hipDeviceReset

#define gpuFree                   hipFree
#define gpuFreeHost               hipHostFree
#define gpuFreeAsync              hipFreeAsync
#define gpuMalloc                 hipMalloc
#define gpuMallocAsync            hipMallocAsync
#define gpuHostMalloc             hipHostMalloc
#define gpuHostMallocPortable     hipHostMallocPortable
#define gpuMemcpy                 hipMemcpy
#define gpuMemcpyAsync            hipMemcpyAsync
#define gpuMemset                 hipMemset
#define gpuMemsetAsync            hipMemsetAsync

#define gpuStreamCreate           hipStreamCreate
#define gpuStreamDestroy          hipStreamDestroy
#define gpuStreamWaitEvent        hipStreamWaitEvent
#define gpuStreamSynchronize      hipStreamSynchronize

#define gpuEventCreate            hipEventCreate
#define gpuEventCreateWithFlags   hipEventCreateWithFlags
#define gpuEventDestroy           hipEventDestroy
#define gpuEventQuery             hipEventQuery
#define gpuEventRecord            hipEventRecord
#define gpuEventSynchronize       hipEventSynchronize
#define gpuEventElapsedTime       hipEventElapsedTime

#define gpuError_t                hipError_t
#define gpuSuccess                hipSuccess

#define gpuStream_t               hipStream_t
#define gpuDeviceProp_t           hipDeviceProp_t

#define gpuEvent_t                hipEvent_t
#define gpuEventDefault           hipEventDefault
#define gpuEventBlockingSync      hipEventBlockingSync
#define gpuEventDisableTiming     hipEventDisableTiming

#define gpuInit                   hipInit
#define gpuDeviceGet              hipDeviceGet
#define gpuDeviceGetName          hipDeviceGetName
#define gpuDeviceGetCount         hipDeviceGetCount
#define gpuDeviceTotalMem         hipDeviceTotalMem
#define gpuDevice_t               hipDevice_t
#define gpuCtx_t                  hipCtx_t
#define gpuCtxCreate              hipCtxCreate
#define gpuCtxSetCurrent          hipCtxSetCurrent
#define gpuMemGetInfo             hipMemGetInfo

/* driver_types */
#define gpuMemcpyKind             hipMemcpyKind
#define gpuMemcpyDeviceToHost     hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice     hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice   hipMemcpyDeviceToDevice

#endif
