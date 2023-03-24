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
