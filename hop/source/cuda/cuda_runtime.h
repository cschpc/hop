#ifndef __HOP_SOURCE_CUDA_RUNTIME_H__
#define __HOP_SOURCE_CUDA_RUNTIME_H__

#define cudaMemcpyKind            gpuMemcpyKind
#define cudaMemcpyDeviceToHost    gpuMemcpyDeviceToHost
#define cudaMemcpyHostToDevice    gpuMemcpyHostToDevice
#define cudaSuccess               gpuSuccess
#define cudaEventDefault          gpuEventDefault
#define cudaEventBlockingSync     gpuEventBlockingSync
#define cudaEventDisableTiming    gpuEventDisableTiming

#define cudaStream_t              gpuStream_t
#define cudaEvent_t               gpuEvent_t
#define cudaError_t               gpuError_t
#define cudaDeviceProp            gpuDeviceProp_t

#define cuDoubleComplex           gpuDoubleComplex
#define make_cuDoubleComplex      make_gpuDoubleComplex
#define cuCreal                   gpuCreal
#define cuCimag                   gpuCimag
#define cuCadd                    gpuCadd
#define cuCmul                    gpuCmul
#define cuConj                    gpuConj

#define cudaGetLastError          gpuGetLastError
#define cudaGetErrorString        gpuGetErrorString

#define cudaSetDevice             gpuSetDevice
#define cudaGetDevice             gpuGetDevice
#define cudaGetDeviceProperties   gpuGetDeviceProperties
#define cudaDeviceSynchronize     gpuDeviceSynchronize

#define cudaFree                  gpuFree
#define cudaFreeHost              gpuFreeHost
#define cudaMalloc                gpuMalloc
#define cudaHostAlloc             gpuHostMalloc
#define cudaHostAllocPortable     gpuHostMallocPortable
#define cudaMemcpy                gpuMemcpy
#define cudaMemcpyAsync           gpuMemcpyAsync

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

#include <hop/hop_runtime.h>

#endif
