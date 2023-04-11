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

#ifndef __HOP_RUNTIME_HIP_H__
#define __HOP_RUNTIME_HIP_H__

#include <hip/hip_runtime.h>

#define gpuMemcpyKind             hipMemcpyKind
#define gpuMemcpyDeviceToHost     hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice     hipMemcpyHostToDevice
#define gpuSuccess                hipSuccess
#define gpuEventDefault           hipEventDefault
#define gpuEventBlockingSync      hipEventBlockingSync
#define gpuEventDisableTiming     hipEventDisableTiming

#define gpuStream_t               hipStream_t
#define gpuEvent_t                hipEvent_t
#define gpuError_t                hipError_t
#define gpuDeviceProp_t           hipDeviceProp_t

#define gpuDoubleComplex          hipDoubleComplex
#define make_gpuDoubleComplex     make_hipDoubleComplex
#define gpuCreal                  hipCreal
#define gpuCimag                  hipCimag
#define gpuCadd                   hipCadd
#define gpuCmul                   hipCmul
#define gpuConj                   hipConj

#define gpuGetLastError           hipGetLastError
#define gpuGetErrorString         hipGetErrorString

#define gpuSetDevice              hipSetDevice
#define gpuGetDevice              hipGetDevice
#define gpuGetDeviceProperties    hipGetDeviceProperties
#define gpuDeviceSynchronize      hipDeviceSynchronize

#define gpuFree                   hipFree
#define gpuFreeHost               hipHostFree
#define gpuMalloc                 hipMalloc
#define gpuHostMalloc             hipHostMalloc
#define gpuHostMallocPortable     hipHostMallocPortable
#define gpuMemcpy                 hipMemcpy
#define gpuMemcpyAsync            hipMemcpyAsync

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

#define gpuLaunchKernel           hipLaunchKernelGGL

#endif
