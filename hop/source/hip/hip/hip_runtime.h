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

#ifndef __HOP_SOURCE_HIP_RUNTIME_H__
#define __HOP_SOURCE_HIP_RUNTIME_H__

#define hipMemcpyKind             gpuMemcpyKind
#define hipMemcpyDeviceToHost     gpuMemcpyDeviceToHost
#define hipMemcpyHostToDevice     gpuMemcpyHostToDevice
#define hipSuccess                gpuSuccess
#define hipEventDefault           gpuEventDefault
#define hipEventBlockingSync      gpuEventBlockingSync
#define hipEventDisableTiming     gpuEventDisableTiming

#define hipStream_t               gpuStream_t
#define hipEvent_t                gpuEvent_t
#define hipError_t                gpuError_t
#define hipDeviceProp_t           gpuDeviceProp_t

#define hipDoubleComplex          gpuDoubleComplex
#define make_hipDoubleComplex     make_gpuDoubleComplex
#define hipCreal                  gpuCreal
#define hipCimag                  gpuCimag
#define hipCadd                   gpuCadd
#define hipCmul                   gpuCmul
#define hipConj                   gpuConj

#define hipGetLastError           gpuGetLastError
#define hipGetErrorString         gpuGetErrorString

#define hipSetDevice              gpuSetDevice
#define hipGetDevice              gpuGetDevice
#define hipGetDeviceProperties    gpuGetDeviceProperties
#define hipDeviceSynchronize      gpuDeviceSynchronize

#define hipFree                   gpuFree
#define hipHostFree               gpuFreeHost
#define hipMalloc                 gpuMalloc
#define hipHostMalloc             gpuHostMalloc
#define hipHostMallocPortable     gpuHostMallocPortable
#define hipMemcpy                 gpuMemcpy
#define hipMemcpyAsync            gpuMemcpyAsync

#define hipStreamCreate           gpuStreamCreate
#define hipStreamDestroy          gpuStreamDestroy
#define hipStreamWaitEvent        gpuStreamWaitEvent
#define hipStreamSynchronize      gpuStreamSynchronize

#define hipEventCreate            gpuEventCreate
#define hipEventCreateWithFlags   gpuEventCreateWithFlags
#define hipEventDestroy           gpuEventDestroy
#define hipEventQuery             gpuEventQuery
#define hipEventRecord            gpuEventRecord
#define hipEventSynchronize       gpuEventSynchronize
#define hipEventElapsedTime       gpuEventElapsedTime

#define hipLaunchKernelGGL        gpuLaunchKernel

#include <hop/hop_runtime.h>

#endif
