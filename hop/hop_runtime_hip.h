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
