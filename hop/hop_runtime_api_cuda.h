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

#ifndef __HOP_HOP_RUNTIME_API_CUDA_H__
#define __HOP_HOP_RUNTIME_API_CUDA_H__

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cudaGL.h>
#include <driver_functions.h>
#include <driver_types.h>

#define GPU_IPC_HANDLE_SIZE              CU_IPC_HANDLE_SIZE
#define GPU_LAUNCH_PARAM_BUFFER_POINTER  CU_LAUNCH_PARAM_BUFFER_POINTER
#define GPU_LAUNCH_PARAM_BUFFER_SIZE     CU_LAUNCH_PARAM_BUFFER_SIZE
#define GPU_LAUNCH_PARAM_END             CU_LAUNCH_PARAM_END
#define gpuAccessPolicyWindow            CUaccessPolicyWindow
#define gpuAccessProperty                CUaccessProperty
#define gpuAccessPropertyNormal          CU_ACCESS_PROPERTY_NORMAL
#define gpuAccessPropertyPersisting      CU_ACCESS_PROPERTY_PERSISTING
#define gpuAccessPropertyStreaming       CU_ACCESS_PROPERTY_STREAMING
#define gpuArray3DCreate                 cuArray3DCreate_v2
#define gpuArray3DGetDescriptor          cuArray3DGetDescriptor_v2
#define gpuArrayCreate                   cuArrayCreate_v2
#define gpuArrayCubemap                  CUDA_ARRAY3D_CUBEMAP
#define gpuArrayDefault                  cudaArrayDefault
#define gpuArrayDestroy                  cuArrayDestroy
#define gpuArrayGetDescriptor            cuArrayGetDescriptor_v2
#define gpuArrayGetInfo                  cudaArrayGetInfo
#define gpuArrayLayered                  CUDA_ARRAY3D_LAYERED
#define gpuArrayMapInfo                  CUarrayMapInfo
#define gpuArraySparseSubresourceType    CUarraySparseSubresourceType
#define gpuArraySparseSubresourceTypeMiptail  \
        CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL
#define gpuArraySparseSubresourceTypeSparseLevel  \
        CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL
#define gpuArraySurfaceLoadStore         CUDA_ARRAY3D_SURFACE_LDST
#define gpuArrayTextureGather            CUDA_ARRAY3D_TEXTURE_GATHER
#define gpuBindTexture                   cudaBindTexture
#define gpuBindTexture2D                 cudaBindTexture2D
#define gpuBindTextureToArray            cudaBindTextureToArray
#define gpuBindTextureToMipmappedArray   cudaBindTextureToMipmappedArray
#define gpuChooseDevice                  cudaChooseDevice
#define gpuComputeMode                   CUcomputemode
#define gpuComputeModeDefault            CU_COMPUTEMODE_DEFAULT
#define gpuComputeModeExclusive          cudaComputeModeExclusive
#define gpuComputeModeExclusiveProcess   CU_COMPUTEMODE_EXCLUSIVE_PROCESS
#define gpuComputeModeProhibited         CU_COMPUTEMODE_PROHIBITED
#define gpuCooperativeLaunchMultiDeviceNoPostSync  \
        CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC
#define gpuCooperativeLaunchMultiDeviceNoPreSync  \
        CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC
#define gpuCpuDeviceId                   CU_DEVICE_CPU
#define gpuCreateSurfaceObject           cudaCreateSurfaceObject
#define gpuCreateTextureObject           cudaCreateTextureObject
#define gpuCtxCreate                     cuCtxCreate_v2
#define gpuCtxDestroy                    cuCtxDestroy_v2
#define gpuCtxDisablePeerAccess          cuCtxDisablePeerAccess
#define gpuCtxEnablePeerAccess           cuCtxEnablePeerAccess
#define gpuCtxGetApiVersion              cuCtxGetApiVersion
#define gpuCtxGetCacheConfig             cuCtxGetCacheConfig
#define gpuCtxGetCurrent                 cuCtxGetCurrent
#define gpuCtxGetDevice                  cuCtxGetDevice
#define gpuCtxGetFlags                   cuCtxGetFlags
#define gpuCtxGetSharedMemConfig         cuCtxGetSharedMemConfig
#define gpuCtxPopCurrent                 cuCtxPopCurrent_v2
#define gpuCtxPushCurrent                cuCtxPushCurrent_v2
#define gpuCtxSetCacheConfig             cuCtxSetCacheConfig
#define gpuCtxSetCurrent                 cuCtxSetCurrent
#define gpuCtxSetSharedMemConfig         cuCtxSetSharedMemConfig
#define gpuCtxSynchronize                cuCtxSynchronize
#define gpuCtx_t                         CUcontext
#define gpuDestroyExternalMemory         cuDestroyExternalMemory
#define gpuDestroyExternalSemaphore      cuDestroyExternalSemaphore
#define gpuDestroySurfaceObject          cudaDestroySurfaceObject
#define gpuDestroyTextureObject          cudaDestroyTextureObject
#define gpuDevP2PAttrAccessSupported     CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED
#define gpuDevP2PAttrHipArrayAccessSupported  \
        cudaDevP2PAttrCudaArrayAccessSupported
#define gpuDevP2PAttrNativeAtomicSupported  \
        CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED
#define gpuDevP2PAttrPerformanceRank     CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK
#define gpuDeviceAttributeAsyncEngineCount  \
        cudaDevAttrAsyncEngineCount
#define gpuDeviceAttributeCanMapHostMemory  \
        cudaDevAttrCanMapHostMemory
#define gpuDeviceAttributeCanUseHostPointerForRegisteredMem  \
        cudaDevAttrCanUseHostPointerForRegisteredMem
#define gpuDeviceAttributeCanUseStreamWaitValue  \
        cudaDevAttrReserved94
#define gpuDeviceAttributeClockRate      cudaDevAttrClockRate
#define gpuDeviceAttributeComputeCapabilityMajor  \
        cudaDevAttrComputeCapabilityMajor
#define gpuDeviceAttributeComputeCapabilityMinor  \
        cudaDevAttrComputeCapabilityMinor
#define gpuDeviceAttributeComputeMode    cudaDevAttrComputeMode
#define gpuDeviceAttributeComputePreemptionSupported  \
        cudaDevAttrComputePreemptionSupported
#define gpuDeviceAttributeConcurrentKernels  \
        cudaDevAttrConcurrentKernels
#define gpuDeviceAttributeConcurrentManagedAccess  \
        cudaDevAttrConcurrentManagedAccess
#define gpuDeviceAttributeCooperativeLaunch  \
        cudaDevAttrCooperativeLaunch
#define gpuDeviceAttributeCooperativeMultiDeviceLaunch  \
        cudaDevAttrCooperativeMultiDeviceLaunch
#define gpuDeviceAttributeDirectManagedMemAccessFromHost  \
        cudaDevAttrDirectManagedMemAccessFromHost
#define gpuDeviceAttributeEccEnabled     cudaDevAttrEccEnabled
#define gpuDeviceAttributeGlobalL1CacheSupported  \
        cudaDevAttrGlobalL1CacheSupported
#define gpuDeviceAttributeHostNativeAtomicSupported  \
        cudaDevAttrHostNativeAtomicSupported
#define gpuDeviceAttributeIntegrated     cudaDevAttrIntegrated
#define gpuDeviceAttributeIsMultiGpuBoard  \
        cudaDevAttrIsMultiGpuBoard
#define gpuDeviceAttributeKernelExecTimeout  \
        cudaDevAttrKernelExecTimeout
#define gpuDeviceAttributeL2CacheSize    cudaDevAttrL2CacheSize
#define gpuDeviceAttributeLocalL1CacheSupported  \
        cudaDevAttrLocalL1CacheSupported
#define gpuDeviceAttributeManagedMemory  cudaDevAttrManagedMemory
#define gpuDeviceAttributeMaxBlockDimX   cudaDevAttrMaxBlockDimX
#define gpuDeviceAttributeMaxBlockDimY   cudaDevAttrMaxBlockDimY
#define gpuDeviceAttributeMaxBlockDimZ   cudaDevAttrMaxBlockDimZ
#define gpuDeviceAttributeMaxBlocksPerMultiProcessor  \
        cudaDevAttrMaxBlocksPerMultiprocessor
#define gpuDeviceAttributeMaxGridDimX    cudaDevAttrMaxGridDimX
#define gpuDeviceAttributeMaxGridDimY    cudaDevAttrMaxGridDimY
#define gpuDeviceAttributeMaxGridDimZ    cudaDevAttrMaxGridDimZ
#define gpuDeviceAttributeMaxPitch       cudaDevAttrMaxPitch
#define gpuDeviceAttributeMaxRegistersPerBlock  \
        cudaDevAttrMaxRegistersPerBlock
#define gpuDeviceAttributeMaxRegistersPerMultiprocessor  \
        cudaDevAttrMaxRegistersPerMultiprocessor
#define gpuDeviceAttributeMaxSharedMemoryPerBlock  \
        cudaDevAttrMaxSharedMemoryPerBlock
#define gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor  \
        cudaDevAttrMaxSharedMemoryPerMultiprocessor
#define gpuDeviceAttributeMaxSurface1D   cudaDevAttrMaxSurface1DWidth
#define gpuDeviceAttributeMaxSurface1DLayered  \
        cudaDevAttrMaxSurface1DLayeredWidth
#define gpuDeviceAttributeMaxSurface2D   cudaDevAttrMaxSurface2DHeight
#define gpuDeviceAttributeMaxSurface2DLayered  \
        cudaDevAttrMaxSurface2DLayeredHeight
#define gpuDeviceAttributeMaxSurface3D   cudaDevAttrMaxSurface3DDepth
#define gpuDeviceAttributeMaxSurfaceCubemap  \
        cudaDevAttrMaxSurfaceCubemapWidth
#define gpuDeviceAttributeMaxSurfaceCubemapLayered  \
        cudaDevAttrMaxSurfaceCubemapLayeredWidth
#define gpuDeviceAttributeMaxTexture1DLayered  \
        cudaDevAttrMaxTexture1DLayeredWidth
#define gpuDeviceAttributeMaxTexture1DLinear  \
        cudaDevAttrMaxTexture1DLinearWidth
#define gpuDeviceAttributeMaxTexture1DMipmap  \
        cudaDevAttrMaxTexture1DMipmappedWidth
#define gpuDeviceAttributeMaxTexture1DWidth  \
        cudaDevAttrMaxTexture1DWidth
#define gpuDeviceAttributeMaxTexture2DGather  \
        cudaDevAttrMaxTexture2DGatherHeight
#define gpuDeviceAttributeMaxTexture2DHeight  \
        cudaDevAttrMaxTexture2DHeight
#define gpuDeviceAttributeMaxTexture2DLayered  \
        cudaDevAttrMaxTexture2DLayeredHeight
#define gpuDeviceAttributeMaxTexture2DLinear  \
        cudaDevAttrMaxTexture2DLinearHeight
#define gpuDeviceAttributeMaxTexture2DMipmap  \
        cudaDevAttrMaxTexture2DMipmappedHeight
#define gpuDeviceAttributeMaxTexture2DWidth  \
        cudaDevAttrMaxTexture2DWidth
#define gpuDeviceAttributeMaxTexture3DAlt  \
        cudaDevAttrMaxTexture3DDepthAlt
#define gpuDeviceAttributeMaxTexture3DDepth  \
        cudaDevAttrMaxTexture3DDepth
#define gpuDeviceAttributeMaxTexture3DHeight  \
        cudaDevAttrMaxTexture3DHeight
#define gpuDeviceAttributeMaxTexture3DWidth  \
        cudaDevAttrMaxTexture3DWidth
#define gpuDeviceAttributeMaxTextureCubemap  \
        cudaDevAttrMaxTextureCubemapWidth
#define gpuDeviceAttributeMaxTextureCubemapLayered  \
        cudaDevAttrMaxTextureCubemapLayeredWidth
#define gpuDeviceAttributeMaxThreadsPerBlock  \
        cudaDevAttrMaxThreadsPerBlock
#define gpuDeviceAttributeMaxThreadsPerMultiProcessor  \
        cudaDevAttrMaxThreadsPerMultiProcessor
#define gpuDeviceAttributeMemoryBusWidth cudaDevAttrGlobalMemoryBusWidth
#define gpuDeviceAttributeMemoryClockRate  \
        cudaDevAttrMemoryClockRate
#define gpuDeviceAttributeMemoryPoolsSupported  \
        cudaDevAttrMemoryPoolsSupported
#define gpuDeviceAttributeMultiGpuBoardGroupID  \
        cudaDevAttrMultiGpuBoardGroupID
#define gpuDeviceAttributeMultiprocessorCount  \
        cudaDevAttrMultiProcessorCount
#define gpuDeviceAttributePageableMemoryAccess  \
        cudaDevAttrPageableMemoryAccess
#define gpuDeviceAttributePageableMemoryAccessUsesHostPageTables  \
        cudaDevAttrPageableMemoryAccessUsesHostPageTables
#define gpuDeviceAttributePciBusId       cudaDevAttrPciBusId
#define gpuDeviceAttributePciDeviceId    cudaDevAttrPciDeviceId
#define gpuDeviceAttributePciDomainID    cudaDevAttrPciDomainId
#define gpuDeviceAttributeSharedMemPerBlockOptin  \
        cudaDevAttrMaxSharedMemoryPerBlockOptin
#define gpuDeviceAttributeSingleToDoublePrecisionPerfRatio  \
        cudaDevAttrSingleToDoublePrecisionPerfRatio
#define gpuDeviceAttributeStreamPrioritiesSupported  \
        cudaDevAttrStreamPrioritiesSupported
#define gpuDeviceAttributeSurfaceAlignment  \
        cudaDevAttrSurfaceAlignment
#define gpuDeviceAttributeTccDriver      cudaDevAttrTccDriver
#define gpuDeviceAttributeTextureAlignment  \
        cudaDevAttrTextureAlignment
#define gpuDeviceAttributeTexturePitchAlignment  \
        cudaDevAttrTexturePitchAlignment
#define gpuDeviceAttributeTotalConstantMemory  \
        cudaDevAttrTotalConstantMemory
#define gpuDeviceAttributeUnifiedAddressing  \
        cudaDevAttrUnifiedAddressing
#define gpuDeviceAttributeVirtualMemoryManagementSupported  \
        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
#define gpuDeviceAttributeWarpSize       cudaDevAttrWarpSize
#define gpuDeviceAttribute_t             cudaDeviceAttr
#define gpuDeviceCanAccessPeer           cuDeviceCanAccessPeer
#define gpuDeviceComputeCapability       cuDeviceComputeCapability
#define gpuDeviceDisablePeerAccess       cudaDeviceDisablePeerAccess
#define gpuDeviceEnablePeerAccess        cudaDeviceEnablePeerAccess
#define gpuDeviceGet                     cuDeviceGet
#define gpuDeviceGetAttribute            cuDeviceGetAttribute
#define gpuDeviceGetByPCIBusId           cuDeviceGetByPCIBusId
#define gpuDeviceGetCacheConfig          cudaThreadGetCacheConfig
#define gpuDeviceGetDefaultMemPool       cuDeviceGetDefaultMemPool
#define gpuDeviceGetGraphMemAttribute    cuDeviceGetGraphMemAttribute
#define gpuDeviceGetLimit                cuCtxGetLimit
#define gpuDeviceGetMemPool              cuDeviceGetMemPool
#define gpuDeviceGetName                 cuDeviceGetName
#define gpuDeviceGetP2PAttribute         cuDeviceGetP2PAttribute
#define gpuDeviceGetPCIBusId             cuDeviceGetPCIBusId
#define gpuDeviceGetSharedMemConfig      cudaDeviceGetSharedMemConfig
#define gpuDeviceGetStreamPriorityRange  cuCtxGetStreamPriorityRange
#define gpuDeviceGetUuid                 cuDeviceGetUuid
#define gpuDeviceGraphMemTrim            cuDeviceGraphMemTrim
#define gpuDeviceLmemResizeToMax         CU_CTX_LMEM_RESIZE_TO_MAX
#define gpuDeviceMapHost                 CU_CTX_MAP_HOST
#define gpuDeviceP2PAttr                 CUdevice_P2PAttribute
#define gpuDevicePrimaryCtxGetState      cuDevicePrimaryCtxGetState
#define gpuDevicePrimaryCtxRelease       cuDevicePrimaryCtxRelease_v2
#define gpuDevicePrimaryCtxReset         cuDevicePrimaryCtxReset_v2
#define gpuDevicePrimaryCtxRetain        cuDevicePrimaryCtxRetain
#define gpuDevicePrimaryCtxSetFlags      cuDevicePrimaryCtxSetFlags_v2
#define gpuDeviceProp_t                  cudaDeviceProp
#define gpuDeviceReset                   cudaThreadExit
#define gpuDeviceScheduleAuto            CU_CTX_SCHED_AUTO
#define gpuDeviceScheduleBlockingSync    cudaDeviceBlockingSync
#define gpuDeviceScheduleMask            CU_CTX_SCHED_MASK
#define gpuDeviceScheduleSpin            CU_CTX_SCHED_SPIN
#define gpuDeviceScheduleYield           CU_CTX_SCHED_YIELD
#define gpuDeviceSetCacheConfig          cudaThreadSetCacheConfig
#define gpuDeviceSetGraphMemAttribute    cuDeviceSetGraphMemAttribute
#define gpuDeviceSetLimit                cuCtxSetLimit
#define gpuDeviceSetMemPool              cuDeviceSetMemPool
#define gpuDeviceSetSharedMemConfig      cudaDeviceSetSharedMemConfig
#define gpuDeviceSynchronize             cudaThreadSynchronize
#define gpuDeviceTotalMem                cuDeviceTotalMem_v2
#define gpuDevice_t                      CUdevice
#define gpuDriverGetVersion              cuDriverGetVersion
#define gpuDrvGetErrorName               cuGetErrorName
#define gpuDrvGetErrorString             cuGetErrorString
#define gpuDrvMemcpy2DUnaligned          cuMemcpy2DUnaligned_v2
#define gpuDrvMemcpy3D                   cuMemcpy3D_v2
#define gpuDrvMemcpy3DAsync              cuMemcpy3DAsync_v2
#define gpuDrvPointerGetAttributes       cuPointerGetAttributes
#define gpuErrorAlreadyAcquired          CUDA_ERROR_ALREADY_ACQUIRED
#define gpuErrorAlreadyMapped            CUDA_ERROR_ALREADY_MAPPED
#define gpuErrorArrayIsMapped            CUDA_ERROR_ARRAY_IS_MAPPED
#define gpuErrorAssert                   CUDA_ERROR_ASSERT
#define gpuErrorCapturedEvent            CUDA_ERROR_CAPTURED_EVENT
#define gpuErrorContextAlreadyCurrent    CUDA_ERROR_CONTEXT_ALREADY_CURRENT
#define gpuErrorContextAlreadyInUse      cudaErrorDeviceAlreadyInUse
#define gpuErrorContextIsDestroyed       CUDA_ERROR_CONTEXT_IS_DESTROYED
#define gpuErrorCooperativeLaunchTooLarge  \
        CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
#define gpuErrorDeinitialized            cudaErrorCudartUnloading
#define gpuErrorECCNotCorrectable        cudaErrorECCUncorrectable
#define gpuErrorFileNotFound             CUDA_ERROR_FILE_NOT_FOUND
#define gpuErrorGraphExecUpdateFailure   CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE
#define gpuErrorHostMemoryAlreadyRegistered  \
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
#define gpuErrorHostMemoryNotRegistered  CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED
#define gpuErrorIllegalAddress           CUDA_ERROR_ILLEGAL_ADDRESS
#define gpuErrorIllegalState             CUDA_ERROR_ILLEGAL_STATE
#define gpuErrorInsufficientDriver       cudaErrorInsufficientDriver
#define gpuErrorInvalidConfiguration     cudaErrorInvalidConfiguration
#define gpuErrorInvalidContext           cudaErrorDeviceUninitialized
#define gpuErrorInvalidDevice            CUDA_ERROR_INVALID_DEVICE
#define gpuErrorInvalidDeviceFunction    cudaErrorInvalidDeviceFunction
#define gpuErrorInvalidDevicePointer     cudaErrorInvalidDevicePointer
#define gpuErrorInvalidGraphicsContext   CUDA_ERROR_INVALID_GRAPHICS_CONTEXT
#define gpuErrorInvalidHandle            cudaErrorInvalidResourceHandle
#define gpuErrorInvalidImage             cudaErrorInvalidKernelImage
#define gpuErrorInvalidKernelFile        cudaErrorInvalidPtx
#define gpuErrorInvalidMemcpyDirection   cudaErrorInvalidMemcpyDirection
#define gpuErrorInvalidPitchValue        cudaErrorInvalidPitchValue
#define gpuErrorInvalidSource            CUDA_ERROR_INVALID_SOURCE
#define gpuErrorInvalidSymbol            cudaErrorInvalidSymbol
#define gpuErrorInvalidValue             CUDA_ERROR_INVALID_VALUE
#define gpuErrorLaunchFailure            CUDA_ERROR_LAUNCH_FAILED
#define gpuErrorLaunchOutOfResources     CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
#define gpuErrorLaunchTimeOut            cudaErrorLaunchTimeout
#define gpuErrorMapFailed                cudaErrorMapBufferObjectFailed
#define gpuErrorMissingConfiguration     cudaErrorMissingConfiguration
#define gpuErrorNoBinaryForGpu           cudaErrorNoKernelImageForDevice
#define gpuErrorNoDevice                 CUDA_ERROR_NO_DEVICE
#define gpuErrorNotFound                 cudaErrorSymbolNotFound
#define gpuErrorNotInitialized           cudaErrorInitializationError
#define gpuErrorNotMapped                CUDA_ERROR_NOT_MAPPED
#define gpuErrorNotMappedAsArray         CUDA_ERROR_NOT_MAPPED_AS_ARRAY
#define gpuErrorNotMappedAsPointer       CUDA_ERROR_NOT_MAPPED_AS_POINTER
#define gpuErrorNotReady                 CUDA_ERROR_NOT_READY
#define gpuErrorNotSupported             CUDA_ERROR_NOT_SUPPORTED
#define gpuErrorOperatingSystem          CUDA_ERROR_OPERATING_SYSTEM
#define gpuErrorOutOfMemory              cudaErrorMemoryAllocation
#define gpuErrorPeerAccessAlreadyEnabled CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
#define gpuErrorPeerAccessNotEnabled     CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
#define gpuErrorPeerAccessUnsupported    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED
#define gpuErrorPriorLaunchFailure       cudaErrorPriorLaunchFailure
#define gpuErrorProfilerAlreadyStarted   CUDA_ERROR_PROFILER_ALREADY_STARTED
#define gpuErrorProfilerAlreadyStopped   CUDA_ERROR_PROFILER_ALREADY_STOPPED
#define gpuErrorProfilerDisabled         CUDA_ERROR_PROFILER_DISABLED
#define gpuErrorProfilerNotInitialized   CUDA_ERROR_PROFILER_NOT_INITIALIZED
#define gpuErrorSetOnActiveProcess       CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
#define gpuErrorSharedObjectInitFailed   CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
#define gpuErrorSharedObjectSymbolNotFound  \
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND
#define gpuErrorStreamCaptureImplicit    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
#define gpuErrorStreamCaptureInvalidated CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
#define gpuErrorStreamCaptureIsolation   CUDA_ERROR_STREAM_CAPTURE_ISOLATION
#define gpuErrorStreamCaptureMerge       CUDA_ERROR_STREAM_CAPTURE_MERGE
#define gpuErrorStreamCaptureUnjoined    CUDA_ERROR_STREAM_CAPTURE_UNJOINED
#define gpuErrorStreamCaptureUnmatched   CUDA_ERROR_STREAM_CAPTURE_UNMATCHED
#define gpuErrorStreamCaptureUnsupported CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
#define gpuErrorStreamCaptureWrongThread CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD
#define gpuErrorUnknown                  CUDA_ERROR_UNKNOWN
#define gpuErrorUnmapFailed              cudaErrorUnmapBufferObjectFailed
#define gpuErrorUnsupportedLimit         CUDA_ERROR_UNSUPPORTED_LIMIT
#define gpuError_t                       cudaError
#define gpuEventBlockingSync             CU_EVENT_BLOCKING_SYNC
#define gpuEventCreate                   cudaEventCreate
#define gpuEventCreateWithFlags          cuEventCreate
#define gpuEventDefault                  CU_EVENT_DEFAULT
#define gpuEventDestroy                  cuEventDestroy_v2
#define gpuEventDisableTiming            CU_EVENT_DISABLE_TIMING
#define gpuEventElapsedTime              cuEventElapsedTime
#define gpuEventInterprocess             CU_EVENT_INTERPROCESS
#define gpuEventQuery                    cuEventQuery
#define gpuEventRecord                   cuEventRecord
#define gpuEventSynchronize              cuEventSynchronize
#define gpuEvent_t                       CUevent
#define gpuExternalMemoryBufferDesc      CUDA_EXTERNAL_MEMORY_BUFFER_DESC
#define gpuExternalMemoryBufferDesc_st   CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
#define gpuExternalMemoryDedicated       CUDA_EXTERNAL_MEMORY_DEDICATED
#define gpuExternalMemoryGetMappedBuffer cuExternalMemoryGetMappedBuffer
#define gpuExternalMemoryHandleDesc      CUDA_EXTERNAL_MEMORY_HANDLE_DESC
#define gpuExternalMemoryHandleDesc_st   CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
#define gpuExternalMemoryHandleType      CUexternalMemoryHandleType
#define gpuExternalMemoryHandleTypeD3D11Resource  \
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
#define gpuExternalMemoryHandleTypeD3D11ResourceKmt  \
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
#define gpuExternalMemoryHandleTypeD3D12Heap  \
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
#define gpuExternalMemoryHandleTypeD3D12Resource  \
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
#define gpuExternalMemoryHandleTypeOpaqueFd  \
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
#define gpuExternalMemoryHandleTypeOpaqueWin32  \
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
#define gpuExternalMemoryHandleTypeOpaqueWin32Kmt  \
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
#define gpuExternalMemoryHandleType_enum CUexternalMemoryHandleType_enum
#define gpuExternalMemory_t              CUexternalMemory
#define gpuExternalSemaphoreHandleDesc   CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
#define gpuExternalSemaphoreHandleDesc_st  \
        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
#define gpuExternalSemaphoreHandleType   CUexternalSemaphoreHandleType
#define gpuExternalSemaphoreHandleTypeD3D12Fence  \
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
#define gpuExternalSemaphoreHandleTypeOpaqueFd  \
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
#define gpuExternalSemaphoreHandleTypeOpaqueWin32  \
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
#define gpuExternalSemaphoreHandleTypeOpaqueWin32Kmt  \
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
#define gpuExternalSemaphoreHandleType_enum  \
        CUexternalSemaphoreHandleType_enum
#define gpuExternalSemaphoreSignalParams cudaExternalSemaphoreSignalParams_v1
#define gpuExternalSemaphoreSignalParams_st  \
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
#define gpuExternalSemaphoreWaitParams   cudaExternalSemaphoreWaitParams_v1
#define gpuExternalSemaphoreWaitParams_st  \
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
#define gpuExternalSemaphore_t           CUexternalSemaphore
#define gpuFree                          cuMemFree_v2
#define gpuFreeArray                     cudaFreeArray
#define gpuFreeAsync                     cuMemFreeAsync
#define gpuFreeMipmappedArray            cudaFreeMipmappedArray
#define gpuFuncAttribute                 cudaFuncAttribute
#define gpuFuncAttributeMax              cudaFuncAttributeMax
#define gpuFuncAttributeMaxDynamicSharedMemorySize  \
        cudaFuncAttributeMaxDynamicSharedMemorySize
#define gpuFuncAttributePreferredSharedMemoryCarveout  \
        cudaFuncAttributePreferredSharedMemoryCarveout
#define gpuFuncAttributes                cudaFuncAttributes
#define gpuFuncCachePreferEqual          CU_FUNC_CACHE_PREFER_EQUAL
#define gpuFuncCachePreferL1             CU_FUNC_CACHE_PREFER_L1
#define gpuFuncCachePreferNone           CU_FUNC_CACHE_PREFER_NONE
#define gpuFuncCachePreferShared         CU_FUNC_CACHE_PREFER_SHARED
#define gpuFuncCache_t                   cudaFuncCache
#define gpuFuncGetAttribute              cuFuncGetAttribute
#define gpuFuncGetAttributes             cudaFuncGetAttributes
#define gpuFuncSetAttribute              cudaFuncSetAttribute
#define gpuFuncSetCacheConfig            cudaFuncSetCacheConfig
#define gpuFuncSetSharedMemConfig        cudaFuncSetSharedMemConfig
#define gpuFunctionLaunchParams          CUDA_LAUNCH_PARAMS
#define gpuFunctionLaunchParams_t        CUDA_LAUNCH_PARAMS_st
#define gpuFunction_t                    CUfunction
#define gpuGLDeviceList                  CUGLDeviceList
#define gpuGLDeviceListAll               CU_GL_DEVICE_LIST_ALL
#define gpuGLDeviceListCurrentFrame      CU_GL_DEVICE_LIST_CURRENT_FRAME
#define gpuGLDeviceListNextFrame         CU_GL_DEVICE_LIST_NEXT_FRAME
#define gpuGLGetDevices                  cuGLGetDevices
#define gpuGetChannelDesc                cudaGetChannelDesc
#define gpuGetDevice                     cudaGetDevice
#define gpuGetDeviceCount                cuDeviceGetCount
#define gpuGetDeviceFlags                cudaGetDeviceFlags
#define gpuGetDeviceProperties           cudaGetDeviceProperties
#define gpuGetErrorName                  cudaGetErrorName
#define gpuGetErrorString                cudaGetErrorString
#define gpuGetLastError                  cudaGetLastError
#define gpuGetMipmappedArrayLevel        cudaGetMipmappedArrayLevel
#define gpuGetSymbolAddress              cudaGetSymbolAddress
#define gpuGetSymbolSize                 cudaGetSymbolSize
#define gpuGetTextureAlignmentOffset     cudaGetTextureAlignmentOffset
#define gpuGetTextureObjectResourceDesc  cudaGetTextureObjectResourceDesc
#define gpuGetTextureObjectResourceViewDesc  \
        cudaGetTextureObjectResourceViewDesc
#define gpuGetTextureObjectTextureDesc   cudaGetTextureObjectTextureDesc
#define gpuGetTextureReference           cudaGetTextureReference
#define gpuGraphAddChildGraphNode        cuGraphAddChildGraphNode
#define gpuGraphAddDependencies          cuGraphAddDependencies
#define gpuGraphAddEmptyNode             cuGraphAddEmptyNode
#define gpuGraphAddEventRecordNode       cuGraphAddEventRecordNode
#define gpuGraphAddEventWaitNode         cuGraphAddEventWaitNode
#define gpuGraphAddHostNode              cuGraphAddHostNode
#define gpuGraphAddKernelNode            cuGraphAddKernelNode
#define gpuGraphAddMemAllocNode          cuGraphAddMemAllocNode
#define gpuGraphAddMemFreeNode           cuGraphAddMemFreeNode
#define gpuGraphAddMemcpyNode            cudaGraphAddMemcpyNode
#define gpuGraphAddMemcpyNode1D          cudaGraphAddMemcpyNode1D
#define gpuGraphAddMemcpyNodeFromSymbol  cudaGraphAddMemcpyNodeFromSymbol
#define gpuGraphAddMemcpyNodeToSymbol    cudaGraphAddMemcpyNodeToSymbol
#define gpuGraphAddMemsetNode            cudaGraphAddMemsetNode
#define gpuGraphChildGraphNodeGetGraph   cuGraphChildGraphNodeGetGraph
#define gpuGraphClone                    cuGraphClone
#define gpuGraphCreate                   cuGraphCreate
#define gpuGraphDebugDotFlags            CUgraphDebugDot_flags
#define gpuGraphDebugDotFlagsEventNodeParams  \
        CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS
#define gpuGraphDebugDotFlagsExtSemasSignalNodeParams  \
        CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS
#define gpuGraphDebugDotFlagsExtSemasWaitNodeParams  \
        CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS
#define gpuGraphDebugDotFlagsHandles     CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES
#define gpuGraphDebugDotFlagsHostNodeParams  \
        CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS
#define gpuGraphDebugDotFlagsKernelNodeAttributes  \
        CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES
#define gpuGraphDebugDotFlagsKernelNodeParams  \
        CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS
#define gpuGraphDebugDotFlagsMemcpyNodeParams  \
        CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS
#define gpuGraphDebugDotFlagsMemsetNodeParams  \
        CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS
#define gpuGraphDebugDotFlagsVerbose     CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE
#define gpuGraphDebugDotPrint            cuGraphDebugDotPrint
#define gpuGraphDestroy                  cuGraphDestroy
#define gpuGraphDestroyNode              cuGraphDestroyNode
#define gpuGraphEventRecordNodeGetEvent  cuGraphEventRecordNodeGetEvent
#define gpuGraphEventRecordNodeSetEvent  cuGraphEventRecordNodeSetEvent
#define gpuGraphEventWaitNodeGetEvent    cuGraphEventWaitNodeGetEvent
#define gpuGraphEventWaitNodeSetEvent    cuGraphEventWaitNodeSetEvent
#define gpuGraphExecChildGraphNodeSetParams  \
        cuGraphExecChildGraphNodeSetParams
#define gpuGraphExecDestroy              cuGraphExecDestroy
#define gpuGraphExecEventRecordNodeSetEvent  \
        cuGraphExecEventRecordNodeSetEvent
#define gpuGraphExecEventWaitNodeSetEvent  \
        cuGraphExecEventWaitNodeSetEvent
#define gpuGraphExecHostNodeSetParams    cuGraphExecHostNodeSetParams
#define gpuGraphExecKernelNodeSetParams  cuGraphExecKernelNodeSetParams
#define gpuGraphExecMemcpyNodeSetParams  cudaGraphExecMemcpyNodeSetParams
#define gpuGraphExecMemcpyNodeSetParams1D  \
        cudaGraphExecMemcpyNodeSetParams1D
#define gpuGraphExecMemcpyNodeSetParamsFromSymbol  \
        cudaGraphExecMemcpyNodeSetParamsFromSymbol
#define gpuGraphExecMemcpyNodeSetParamsToSymbol  \
        cudaGraphExecMemcpyNodeSetParamsToSymbol
#define gpuGraphExecMemsetNodeSetParams  cudaGraphExecMemsetNodeSetParams
#define gpuGraphExecUpdate               cuGraphExecUpdate
#define gpuGraphExecUpdateError          CU_GRAPH_EXEC_UPDATE_ERROR
#define gpuGraphExecUpdateErrorFunctionChanged  \
        CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED
#define gpuGraphExecUpdateErrorNodeTypeChanged  \
        CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED
#define gpuGraphExecUpdateErrorNotSupported  \
        CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED
#define gpuGraphExecUpdateErrorParametersChanged  \
        CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED
#define gpuGraphExecUpdateErrorTopologyChanged  \
        CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED
#define gpuGraphExecUpdateErrorUnsupportedFunctionChange  \
        CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE
#define gpuGraphExecUpdateResult         CUgraphExecUpdateResult
#define gpuGraphExecUpdateSuccess        CU_GRAPH_EXEC_UPDATE_SUCCESS
#define gpuGraphExec_t                   CUgraphExec
#define gpuGraphGetEdges                 cuGraphGetEdges
#define gpuGraphGetNodes                 cuGraphGetNodes
#define gpuGraphGetRootNodes             cuGraphGetRootNodes
#define gpuGraphHostNodeGetParams        cuGraphHostNodeGetParams
#define gpuGraphHostNodeSetParams        cuGraphHostNodeSetParams
#define gpuGraphInstantiate              cuGraphInstantiate_v2
#define gpuGraphInstantiateFlagAutoFreeOnLaunch  \
        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
#define gpuGraphInstantiateFlagDeviceLaunch  \
        CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH
#define gpuGraphInstantiateFlagUpload    CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD
#define gpuGraphInstantiateFlagUseNodePriority  \
        CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
#define gpuGraphInstantiateFlags         CUgraphInstantiate_flags
#define gpuGraphInstantiateWithFlags     cuGraphInstantiateWithFlags
#define gpuGraphKernelNodeCopyAttributes cuGraphKernelNodeCopyAttributes
#define gpuGraphKernelNodeGetAttribute   cuGraphKernelNodeGetAttribute
#define gpuGraphKernelNodeGetParams      cuGraphKernelNodeGetParams
#define gpuGraphKernelNodeSetAttribute   cuGraphKernelNodeSetAttribute
#define gpuGraphKernelNodeSetParams      cuGraphKernelNodeSetParams
#define gpuGraphLaunch                   cuGraphLaunch
#define gpuGraphMemAllocNodeGetParams    cuGraphMemAllocNodeGetParams
#define gpuGraphMemAttrReservedMemCurrent  \
        CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT
#define gpuGraphMemAttrReservedMemHigh   CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH
#define gpuGraphMemAttrUsedMemCurrent    CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT
#define gpuGraphMemAttrUsedMemHigh       CU_GRAPH_MEM_ATTR_USED_MEM_HIGH
#define gpuGraphMemAttributeType         CUgraphMem_attribute
#define gpuGraphMemFreeNodeGetParams     cuGraphMemFreeNodeGetParams
#define gpuGraphMemcpyNodeGetParams      cuGraphMemcpyNodeGetParams
#define gpuGraphMemcpyNodeSetParams      cuGraphMemcpyNodeSetParams
#define gpuGraphMemcpyNodeSetParams1D    cudaGraphMemcpyNodeSetParams1D
#define gpuGraphMemcpyNodeSetParamsFromSymbol  \
        cudaGraphMemcpyNodeSetParamsFromSymbol
#define gpuGraphMemcpyNodeSetParamsToSymbol  \
        cudaGraphMemcpyNodeSetParamsToSymbol
#define gpuGraphMemsetNodeGetParams      cuGraphMemsetNodeGetParams
#define gpuGraphMemsetNodeSetParams      cuGraphMemsetNodeSetParams
#define gpuGraphNodeFindInClone          cuGraphNodeFindInClone
#define gpuGraphNodeGetDependencies      cuGraphNodeGetDependencies
#define gpuGraphNodeGetDependentNodes    cuGraphNodeGetDependentNodes
#define gpuGraphNodeGetEnabled           cuGraphNodeGetEnabled
#define gpuGraphNodeGetType              cuGraphNodeGetType
#define gpuGraphNodeSetEnabled           cuGraphNodeSetEnabled
#define gpuGraphNodeType                 CUgraphNodeType
#define gpuGraphNodeTypeCount            cudaGraphNodeTypeCount
#define gpuGraphNodeTypeEmpty            CU_GRAPH_NODE_TYPE_EMPTY
#define gpuGraphNodeTypeEventRecord      CU_GRAPH_NODE_TYPE_EVENT_RECORD
#define gpuGraphNodeTypeExtSemaphoreSignal  \
        CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL
#define gpuGraphNodeTypeExtSemaphoreWait CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT
#define gpuGraphNodeTypeGraph            CU_GRAPH_NODE_TYPE_GRAPH
#define gpuGraphNodeTypeHost             CU_GRAPH_NODE_TYPE_HOST
#define gpuGraphNodeTypeKernel           CU_GRAPH_NODE_TYPE_KERNEL
#define gpuGraphNodeTypeMemAlloc         CU_GRAPH_NODE_TYPE_MEM_ALLOC
#define gpuGraphNodeTypeMemFree          CU_GRAPH_NODE_TYPE_MEM_FREE
#define gpuGraphNodeTypeMemcpy           CU_GRAPH_NODE_TYPE_MEMCPY
#define gpuGraphNodeTypeMemset           CU_GRAPH_NODE_TYPE_MEMSET
#define gpuGraphNodeTypeWaitEvent        CU_GRAPH_NODE_TYPE_WAIT_EVENT
#define gpuGraphNode_t                   CUgraphNode
#define gpuGraphReleaseUserObject        cuGraphReleaseUserObject
#define gpuGraphRemoveDependencies       cuGraphRemoveDependencies
#define gpuGraphRetainUserObject         cuGraphRetainUserObject
#define gpuGraphUpload                   cuGraphUpload
#define gpuGraphUserObjectMove           CU_GRAPH_USER_OBJECT_MOVE
#define gpuGraph_t                       CUgraph
#define gpuGraphicsGLRegisterBuffer      cuGraphicsGLRegisterBuffer
#define gpuGraphicsGLRegisterImage       cuGraphicsGLRegisterImage
#define gpuGraphicsMapResources          cuGraphicsMapResources
#define gpuGraphicsRegisterFlags         CUgraphicsRegisterFlags
#define gpuGraphicsRegisterFlagsNone     CU_GRAPHICS_REGISTER_FLAGS_NONE
#define gpuGraphicsRegisterFlagsReadOnly CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY
#define gpuGraphicsRegisterFlagsSurfaceLoadStore  \
        CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST
#define gpuGraphicsRegisterFlagsTextureGather  \
        CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER
#define gpuGraphicsRegisterFlagsWriteDiscard  \
        CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
#define gpuGraphicsResource              cudaGraphicsResource
#define gpuGraphicsResourceGetMappedPointer  \
        cuGraphicsResourceGetMappedPointer_v2
#define gpuGraphicsResource_t            CUgraphicsResource
#define gpuGraphicsSubResourceGetMappedArray  \
        cuGraphicsSubResourceGetMappedArray
#define gpuGraphicsUnmapResources        cuGraphicsUnmapResources
#define gpuGraphicsUnregisterResource    cuGraphicsUnregisterResource
#define gpuHostAlloc                     cuMemHostAlloc
#define gpuHostFn_t                      CUhostFn
#define gpuHostFree                      cudaFreeHost
#define gpuHostGetDevicePointer          cuMemHostGetDevicePointer_v2
#define gpuHostGetFlags                  cuMemHostGetFlags
#define gpuHostMalloc                    cudaMallocHost
#define gpuHostMallocDefault             cudaHostAllocDefault
#define gpuHostMallocMapped              cudaHostAllocMapped
#define gpuHostMallocPortable            cudaHostAllocPortable
#define gpuHostMallocWriteCombined       cudaHostAllocWriteCombined
#define gpuHostNodeParams                CUDA_HOST_NODE_PARAMS
#define gpuHostRegister                  cuMemHostRegister_v2
#define gpuHostRegisterDefault           cudaHostRegisterDefault
#define gpuHostRegisterIoMemory          CU_MEMHOSTREGISTER_IOMEMORY
#define gpuHostRegisterMapped            CU_MEMHOSTREGISTER_DEVICEMAP
#define gpuHostRegisterPortable          CU_MEMHOSTREGISTER_PORTABLE
#define gpuHostRegisterReadOnly          CU_MEMHOSTREGISTER_READ_ONLY
#define gpuHostUnregister                cuMemHostUnregister
#define gpuImportExternalMemory          cuImportExternalMemory
#define gpuImportExternalSemaphore       cuImportExternalSemaphore
#define gpuInit                          cuInit
#define gpuInvalidDeviceId               CU_DEVICE_INVALID
#define gpuIpcCloseMemHandle             cuIpcCloseMemHandle
#define gpuIpcEventHandle_st             CUipcEventHandle_st
#define gpuIpcEventHandle_t              CUipcEventHandle
#define gpuIpcGetEventHandle             cuIpcGetEventHandle
#define gpuIpcGetMemHandle               cuIpcGetMemHandle
#define gpuIpcMemHandle_st               CUipcMemHandle_st
#define gpuIpcMemHandle_t                CUipcMemHandle
#define gpuIpcMemLazyEnablePeerAccess    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
#define gpuIpcOpenEventHandle            cuIpcOpenEventHandle
#define gpuIpcOpenMemHandle              cuIpcOpenMemHandle
#define gpuJitOption                     CUjit_option
#define gpuKernelNodeAttrID              CUkernelNodeAttrID
#define gpuKernelNodeAttrValue           CUkernelNodeAttrValue
#define gpuKernelNodeAttributeAccessPolicyWindow  \
        CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW
#define gpuKernelNodeAttributeCooperative  \
        CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE
#define gpuKernelNodeParams              CUDA_KERNEL_NODE_PARAMS
#define gpuLaunchCooperativeKernel       cudaLaunchCooperativeKernel
#define gpuLaunchCooperativeKernelMultiDevice  \
        cudaLaunchCooperativeKernelMultiDevice
#define gpuLaunchHostFunc                cuLaunchHostFunc
#define gpuLaunchKernel                  cudaLaunchKernel
#define gpuLaunchParams                  cudaLaunchParams
#define gpuLimitMallocHeapSize           CU_LIMIT_MALLOC_HEAP_SIZE
#define gpuLimitPrintfFifoSize           CU_LIMIT_PRINTF_FIFO_SIZE
#define gpuLimitStackSize                CU_LIMIT_STACK_SIZE
#define gpuLimit_t                       cudaLimit
#define gpuMalloc                        cuMemAlloc_v2
#define gpuMalloc3D                      cudaMalloc3D
#define gpuMalloc3DArray                 cudaMalloc3DArray
#define gpuMallocArray                   cudaMallocArray
#define gpuMallocAsync                   cuMemAllocAsync
#define gpuMallocFromPoolAsync           cuMemAllocFromPoolAsync
#define gpuMallocManaged                 cuMemAllocManaged
#define gpuMallocMipmappedArray          cudaMallocMipmappedArray
#define gpuMallocPitch                   cudaMallocPitch
#define gpuMemAccessDesc                 CUmemAccessDesc
#define gpuMemAccessFlags                CUmemAccess_flags
#define gpuMemAccessFlagsProtNone        CU_MEM_ACCESS_FLAGS_PROT_NONE
#define gpuMemAccessFlagsProtRead        CU_MEM_ACCESS_FLAGS_PROT_READ
#define gpuMemAccessFlagsProtReadWrite   CU_MEM_ACCESS_FLAGS_PROT_READWRITE
#define gpuMemAddressFree                cuMemAddressFree
#define gpuMemAddressReserve             cuMemAddressReserve
#define gpuMemAdvise                     cuMemAdvise
#define gpuMemAdviseSetAccessedBy        CU_MEM_ADVISE_SET_ACCESSED_BY
#define gpuMemAdviseSetPreferredLocation CU_MEM_ADVISE_SET_PREFERRED_LOCATION
#define gpuMemAdviseSetReadMostly        CU_MEM_ADVISE_SET_READ_MOSTLY
#define gpuMemAdviseUnsetAccessedBy      CU_MEM_ADVISE_UNSET_ACCESSED_BY
#define gpuMemAdviseUnsetPreferredLocation  \
        CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION
#define gpuMemAdviseUnsetReadMostly      CU_MEM_ADVISE_UNSET_READ_MOSTLY
#define gpuMemAllocHost                  cuMemAllocHost_v2
#define gpuMemAllocNodeParams            CUDA_MEM_ALLOC_NODE_PARAMS
#define gpuMemAllocPitch                 cuMemAllocPitch_v2
#define gpuMemAllocationGranularityMinimum  \
        CU_MEM_ALLOC_GRANULARITY_MINIMUM
#define gpuMemAllocationGranularityRecommended  \
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#define gpuMemAllocationGranularity_flags  \
        CUmemAllocationGranularity_flags
#define gpuMemAllocationHandleType       CUmemAllocationHandleType
#define gpuMemAllocationProp             CUmemAllocationProp
#define gpuMemAllocationType             CUmemAllocationType
#define gpuMemAllocationTypeInvalid      CU_MEM_ALLOCATION_TYPE_INVALID
#define gpuMemAllocationTypeMax          CU_MEM_ALLOCATION_TYPE_MAX
#define gpuMemAllocationTypePinned       CU_MEM_ALLOCATION_TYPE_PINNED
#define gpuMemAttachGlobal               CU_MEM_ATTACH_GLOBAL
#define gpuMemAttachHost                 CU_MEM_ATTACH_HOST
#define gpuMemAttachSingle               CU_MEM_ATTACH_SINGLE
#define gpuMemCreate                     cuMemCreate
#define gpuMemExportToShareableHandle    cuMemExportToShareableHandle
#define gpuMemGenericAllocationHandle_t  CUmemGenericAllocationHandle
#define gpuMemGetAccess                  cuMemGetAccess
#define gpuMemGetAddressRange            cuMemGetAddressRange_v2
#define gpuMemGetAllocationGranularity   cuMemGetAllocationGranularity
#define gpuMemGetAllocationPropertiesFromHandle  \
        cuMemGetAllocationPropertiesFromHandle
#define gpuMemGetInfo                    cuMemGetInfo_v2
#define gpuMemHandleType                 CUmemHandleType
#define gpuMemHandleTypeGeneric          CU_MEM_HANDLE_TYPE_GENERIC
#define gpuMemHandleTypeNone             CU_MEM_HANDLE_TYPE_NONE
#define gpuMemHandleTypePosixFileDescriptor  \
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
#define gpuMemHandleTypeWin32            CU_MEM_HANDLE_TYPE_WIN32
#define gpuMemHandleTypeWin32Kmt         CU_MEM_HANDLE_TYPE_WIN32_KMT
#define gpuMemImportFromShareableHandle  cuMemImportFromShareableHandle
#define gpuMemLocation                   CUmemLocation
#define gpuMemLocationType               CUmemLocationType
#define gpuMemLocationTypeDevice         CU_MEM_LOCATION_TYPE_DEVICE
#define gpuMemLocationTypeInvalid        CU_MEM_LOCATION_TYPE_INVALID
#define gpuMemMap                        cuMemMap
#define gpuMemMapArrayAsync              cuMemMapArrayAsync
#define gpuMemOperationType              CUmemOperationType
#define gpuMemOperationTypeMap           CU_MEM_OPERATION_TYPE_MAP
#define gpuMemOperationTypeUnmap         CU_MEM_OPERATION_TYPE_UNMAP
#define gpuMemPoolAttr                   CUmemPool_attribute
#define gpuMemPoolAttrReleaseThreshold   CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
#define gpuMemPoolAttrReservedMemCurrent CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
#define gpuMemPoolAttrReservedMemHigh    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH
#define gpuMemPoolAttrUsedMemCurrent     CU_MEMPOOL_ATTR_USED_MEM_CURRENT
#define gpuMemPoolAttrUsedMemHigh        CU_MEMPOOL_ATTR_USED_MEM_HIGH
#define gpuMemPoolCreate                 cuMemPoolCreate
#define gpuMemPoolDestroy                cuMemPoolDestroy
#define gpuMemPoolExportPointer          cuMemPoolExportPointer
#define gpuMemPoolExportToShareableHandle  \
        cuMemPoolExportToShareableHandle
#define gpuMemPoolGetAccess              cuMemPoolGetAccess
#define gpuMemPoolGetAttribute           cuMemPoolGetAttribute
#define gpuMemPoolImportFromShareableHandle  \
        cuMemPoolImportFromShareableHandle
#define gpuMemPoolImportPointer          cuMemPoolImportPointer
#define gpuMemPoolProps                  CUmemPoolProps
#define gpuMemPoolPtrExportData          CUmemPoolPtrExportData
#define gpuMemPoolReuseAllowInternalDependencies  \
        CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES
#define gpuMemPoolReuseAllowOpportunistic  \
        CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC
#define gpuMemPoolReuseFollowEventDependencies  \
        CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES
#define gpuMemPoolSetAccess              cuMemPoolSetAccess
#define gpuMemPoolSetAttribute           cuMemPoolSetAttribute
#define gpuMemPoolTrimTo                 cuMemPoolTrimTo
#define gpuMemPool_t                     CUmemoryPool
#define gpuMemPrefetchAsync              cuMemPrefetchAsync
#define gpuMemRangeAttribute             CUmem_range_attribute
#define gpuMemRangeAttributeAccessedBy   CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY
#define gpuMemRangeAttributeLastPrefetchLocation  \
        CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION
#define gpuMemRangeAttributePreferredLocation  \
        CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION
#define gpuMemRangeAttributeReadMostly   CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY
#define gpuMemRangeGetAttribute          cuMemRangeGetAttribute
#define gpuMemRangeGetAttributes         cuMemRangeGetAttributes
#define gpuMemRelease                    cuMemRelease
#define gpuMemRetainAllocationHandle     cuMemRetainAllocationHandle
#define gpuMemSetAccess                  cuMemSetAccess
#define gpuMemUnmap                      cuMemUnmap
#define gpuMemcpy                        cudaMemcpy
#define gpuMemcpy2D                      cudaMemcpy2D
#define gpuMemcpy2DAsync                 cudaMemcpy2DAsync
#define gpuMemcpy2DFromArray             cudaMemcpy2DFromArray
#define gpuMemcpy2DFromArrayAsync        cudaMemcpy2DFromArrayAsync
#define gpuMemcpy2DToArray               cudaMemcpy2DToArray
#define gpuMemcpy2DToArrayAsync          cudaMemcpy2DToArrayAsync
#define gpuMemcpy3D                      cudaMemcpy3D
#define gpuMemcpy3DAsync                 cudaMemcpy3DAsync
#define gpuMemcpyAsync                   cudaMemcpyAsync
#define gpuMemcpyAtoH                    cuMemcpyAtoH_v2
#define gpuMemcpyDtoD                    cuMemcpyDtoD_v2
#define gpuMemcpyDtoDAsync               cuMemcpyDtoDAsync_v2
#define gpuMemcpyDtoH                    cuMemcpyDtoH_v2
#define gpuMemcpyDtoHAsync               cuMemcpyDtoHAsync_v2
#define gpuMemcpyFromArray               cudaMemcpyFromArray
#define gpuMemcpyFromSymbol              cudaMemcpyFromSymbol
#define gpuMemcpyFromSymbolAsync         cudaMemcpyFromSymbolAsync
#define gpuMemcpyHtoA                    cuMemcpyHtoA_v2
#define gpuMemcpyHtoD                    cuMemcpyHtoD_v2
#define gpuMemcpyHtoDAsync               cuMemcpyHtoDAsync_v2
#define gpuMemcpyParam2D                 cuMemcpy2D_v2
#define gpuMemcpyParam2DAsync            cuMemcpy2DAsync_v2
#define gpuMemcpyPeer                    cudaMemcpyPeer
#define gpuMemcpyPeerAsync               cudaMemcpyPeerAsync
#define gpuMemcpyToArray                 cudaMemcpyToArray
#define gpuMemcpyToSymbol                cudaMemcpyToSymbol
#define gpuMemcpyToSymbolAsync           cudaMemcpyToSymbolAsync
#define gpuMemoryAdvise                  CUmem_advise
#define gpuMemoryType                    CUmemorytype
#define gpuMemoryTypeArray               CU_MEMORYTYPE_ARRAY
#define gpuMemoryTypeDevice              CU_MEMORYTYPE_DEVICE
#define gpuMemoryTypeHost                CU_MEMORYTYPE_HOST
#define gpuMemoryTypeManaged             cudaMemoryTypeManaged
#define gpuMemoryTypeUnified             CU_MEMORYTYPE_UNIFIED
#define gpuMemset                        cudaMemset
#define gpuMemset2D                      cudaMemset2D
#define gpuMemset2DAsync                 cudaMemset2DAsync
#define gpuMemset3D                      cudaMemset3D
#define gpuMemset3DAsync                 cudaMemset3DAsync
#define gpuMemsetAsync                   cudaMemsetAsync
#define gpuMemsetD16                     cuMemsetD16_v2
#define gpuMemsetD16Async                cuMemsetD16Async
#define gpuMemsetD32                     cuMemsetD32_v2
#define gpuMemsetD32Async                cuMemsetD32Async
#define gpuMemsetD8                      cuMemsetD8_v2
#define gpuMemsetD8Async                 cuMemsetD8Async
#define gpuMemsetParams                  CUDA_MEMSET_NODE_PARAMS
#define gpuMipmappedArrayCreate          cuMipmappedArrayCreate
#define gpuMipmappedArrayDestroy         cuMipmappedArrayDestroy
#define gpuMipmappedArrayGetLevel        cuMipmappedArrayGetLevel
#define gpuModuleGetFunction             cuModuleGetFunction
#define gpuModuleGetGlobal               cuModuleGetGlobal_v2
#define gpuModuleGetTexRef               cuModuleGetTexRef
#define gpuModuleLaunchCooperativeKernel cuLaunchCooperativeKernel
#define gpuModuleLaunchCooperativeKernelMultiDevice  \
        cuLaunchCooperativeKernelMultiDevice
#define gpuModuleLaunchKernel            cuLaunchKernel
#define gpuModuleLoad                    cuModuleLoad
#define gpuModuleLoadData                cuModuleLoadData
#define gpuModuleLoadDataEx              cuModuleLoadDataEx
#define gpuModuleOccupancyMaxActiveBlocksPerMultiprocessor  \
        cuOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags  \
        cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define gpuModuleOccupancyMaxPotentialBlockSize  \
        cuOccupancyMaxPotentialBlockSize
#define gpuModuleOccupancyMaxPotentialBlockSizeWithFlags  \
        cuOccupancyMaxPotentialBlockSizeWithFlags
#define gpuModuleUnload                  cuModuleUnload
#define gpuModule_t                      CUmodule
#define gpuOccupancyDefault              CU_OCCUPANCY_DEFAULT
#define gpuOccupancyDisableCachingOverride  \
        CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor  \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags  \
        cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define gpuOccupancyMaxPotentialBlockSize  \
        cudaOccupancyMaxPotentialBlockSize
#define gpuOccupancyMaxPotentialBlockSizeVariableSMem  \
        cudaOccupancyMaxPotentialBlockSizeVariableSMem
#define gpuOccupancyMaxPotentialBlockSizeVariableSMemWithFlags  \
        cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
#define gpuOccupancyMaxPotentialBlockSizeWithFlags  \
        cudaOccupancyMaxPotentialBlockSizeWithFlags
#define gpuPeekAtLastError               cudaPeekAtLastError
#define gpuPointerAttribute_t            cudaPointerAttributes
#define gpuPointerGetAttribute           cuPointerGetAttribute
#define gpuPointerGetAttributes          cudaPointerGetAttributes
#define gpuPointerSetAttribute           cuPointerSetAttribute
#define gpuProfilerStart                 cudaProfilerStart
#define gpuProfilerStop                  cudaProfilerStop
#define gpuRuntimeGetVersion             cudaRuntimeGetVersion
#define gpuSetDevice                     cudaSetDevice
#define gpuSetDeviceFlags                cudaSetDeviceFlags
#define gpuSharedMemBankSizeDefault      CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE
#define gpuSharedMemBankSizeEightByte    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE
#define gpuSharedMemBankSizeFourByte     CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE
#define gpuSharedMemConfig               CUsharedconfig
#define gpuSignalExternalSemaphoresAsync cuSignalExternalSemaphoresAsync
#define gpuStreamAddCallback             cuStreamAddCallback
#define gpuStreamAddCaptureDependencies  CU_STREAM_ADD_CAPTURE_DEPENDENCIES
#define gpuStreamAttachMemAsync          cuStreamAttachMemAsync
#define gpuStreamBeginCapture            cuStreamBeginCapture_v2
#define gpuStreamCallback_t              CUstreamCallback
#define gpuStreamCaptureMode             CUstreamCaptureMode
#define gpuStreamCaptureModeGlobal       CU_STREAM_CAPTURE_MODE_GLOBAL
#define gpuStreamCaptureModeRelaxed      CU_STREAM_CAPTURE_MODE_RELAXED
#define gpuStreamCaptureModeThreadLocal  CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
#define gpuStreamCaptureStatus           CUstreamCaptureStatus
#define gpuStreamCaptureStatusActive     CU_STREAM_CAPTURE_STATUS_ACTIVE
#define gpuStreamCaptureStatusInvalidated  \
        CU_STREAM_CAPTURE_STATUS_INVALIDATED
#define gpuStreamCaptureStatusNone       CU_STREAM_CAPTURE_STATUS_NONE
#define gpuStreamCreate                  cudaStreamCreate
#define gpuStreamCreateWithFlags         cuStreamCreate
#define gpuStreamCreateWithPriority      cuStreamCreateWithPriority
#define gpuStreamDefault                 CU_STREAM_DEFAULT
#define gpuStreamDestroy                 cuStreamDestroy_v2
#define gpuStreamEndCapture              cuStreamEndCapture
#define gpuStreamGetCaptureInfo          cuStreamGetCaptureInfo
#define gpuStreamGetCaptureInfo_v2       cuStreamGetCaptureInfo_v2
#define gpuStreamGetFlags                cuStreamGetFlags
#define gpuStreamGetPriority             cuStreamGetPriority
#define gpuStreamIsCapturing             cuStreamIsCapturing
#define gpuStreamNonBlocking             CU_STREAM_NON_BLOCKING
#define gpuStreamPerThread               CU_STREAM_PER_THREAD
#define gpuStreamQuery                   cuStreamQuery
#define gpuStreamSetCaptureDependencies  CU_STREAM_SET_CAPTURE_DEPENDENCIES
#define gpuStreamSynchronize             cuStreamSynchronize
#define gpuStreamUpdateCaptureDependencies  \
        cuStreamUpdateCaptureDependencies
#define gpuStreamUpdateCaptureDependenciesFlags  \
        CUstreamUpdateCaptureDependencies_flags
#define gpuStreamWaitEvent               cuStreamWaitEvent
#define gpuStreamWaitValue32             cuStreamWaitValue32
#define gpuStreamWaitValue64             cuStreamWaitValue64
#define gpuStreamWaitValueAnd            CU_STREAM_WAIT_VALUE_AND
#define gpuStreamWaitValueEq             CU_STREAM_WAIT_VALUE_EQ
#define gpuStreamWaitValueGte            CU_STREAM_WAIT_VALUE_GEQ
#define gpuStreamWaitValueNor            CU_STREAM_WAIT_VALUE_NOR
#define gpuStreamWriteValue32            cuStreamWriteValue32
#define gpuStreamWriteValue64            cuStreamWriteValue64
#define gpuStream_t                      CUstream
#define gpuSuccess                       CUDA_SUCCESS
#define gpuTexObjectCreate               cuTexObjectCreate
#define gpuTexObjectDestroy              cuTexObjectDestroy
#define gpuTexObjectGetResourceDesc      cuTexObjectGetResourceDesc
#define gpuTexObjectGetResourceViewDesc  cuTexObjectGetResourceViewDesc
#define gpuTexObjectGetTextureDesc       cuTexObjectGetTextureDesc
#define gpuTexRefGetAddress              cuTexRefGetAddress_v2
#define gpuTexRefGetAddressMode          cuTexRefGetAddressMode
#define gpuTexRefGetFilterMode           cuTexRefGetFilterMode
#define gpuTexRefGetFlags                cuTexRefGetFlags
#define gpuTexRefGetFormat               cuTexRefGetFormat
#define gpuTexRefGetMaxAnisotropy        cuTexRefGetMaxAnisotropy
#define gpuTexRefGetMipMappedArray       cuTexRefGetMipmappedArray
#define gpuTexRefGetMipmapFilterMode     cuTexRefGetMipmapFilterMode
#define gpuTexRefGetMipmapLevelBias      cuTexRefGetMipmapLevelBias
#define gpuTexRefGetMipmapLevelClamp     cuTexRefGetMipmapLevelClamp
#define gpuTexRefSetAddress              cuTexRefSetAddress_v2
#define gpuTexRefSetAddress2D            cuTexRefSetAddress2D_v3
#define gpuTexRefSetAddressMode          cuTexRefSetAddressMode
#define gpuTexRefSetArray                cuTexRefSetArray
#define gpuTexRefSetBorderColor          cuTexRefSetBorderColor
#define gpuTexRefSetFilterMode           cuTexRefSetFilterMode
#define gpuTexRefSetFlags                cuTexRefSetFlags
#define gpuTexRefSetFormat               cuTexRefSetFormat
#define gpuTexRefSetMaxAnisotropy        cuTexRefSetMaxAnisotropy
#define gpuTexRefSetMipmapFilterMode     cuTexRefSetMipmapFilterMode
#define gpuTexRefSetMipmapLevelBias      cuTexRefSetMipmapLevelBias
#define gpuTexRefSetMipmapLevelClamp     cuTexRefSetMipmapLevelClamp
#define gpuTexRefSetMipmappedArray       cuTexRefSetMipmappedArray
#define gpuThreadExchangeStreamCaptureMode  \
        cuThreadExchangeStreamCaptureMode
#define gpuUUID                          CUuuid
#define gpuUUID_t                        CUuuid_st
#define gpuUnbindTexture                 cudaUnbindTexture
#define gpuUserObjectCreate              cuUserObjectCreate
#define gpuUserObjectFlags               CUuserObject_flags
#define gpuUserObjectNoDestructorSync    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC
#define gpuUserObjectRelease             cuUserObjectRelease
#define gpuUserObjectRetain              cuUserObjectRetain
#define gpuUserObjectRetainFlags         CUuserObjectRetain_flags
#define gpuUserObject_t                  CUuserObject
#define gpuWaitExternalSemaphoresAsync   cuWaitExternalSemaphoresAsync

/* channel_descriptor.h */
#define gpuCreateChannelDesc             cudaCreateChannelDesc

/* driver_types.h */
#define GPU_AD_FORMAT_FLOAT              CU_AD_FORMAT_FLOAT
#define GPU_AD_FORMAT_HALF               CU_AD_FORMAT_HALF
#define GPU_AD_FORMAT_SIGNED_INT16       CU_AD_FORMAT_SIGNED_INT16
#define GPU_AD_FORMAT_SIGNED_INT32       CU_AD_FORMAT_SIGNED_INT32
#define GPU_AD_FORMAT_SIGNED_INT8        CU_AD_FORMAT_SIGNED_INT8
#define GPU_AD_FORMAT_UNSIGNED_INT16     CU_AD_FORMAT_UNSIGNED_INT16
#define GPU_AD_FORMAT_UNSIGNED_INT32     CU_AD_FORMAT_UNSIGNED_INT32
#define GPU_AD_FORMAT_UNSIGNED_INT8      CU_AD_FORMAT_UNSIGNED_INT8
#define GPU_ARRAY3D_DESCRIPTOR           CUDA_ARRAY3D_DESCRIPTOR_st
#define GPU_ARRAY_DESCRIPTOR             CUDA_ARRAY_DESCRIPTOR_st
#define GPU_FUNC_ATTRIBUTE_BINARY_VERSION  \
        CU_FUNC_ATTRIBUTE_BINARY_VERSION
#define GPU_FUNC_ATTRIBUTE_CACHE_MODE_CA CU_FUNC_ATTRIBUTE_CACHE_MODE_CA
#define GPU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES  \
        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
#define GPU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES  \
        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define GPU_FUNC_ATTRIBUTE_MAX           CU_FUNC_ATTRIBUTE_MAX
#define GPU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES  \
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define GPU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK  \
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define GPU_FUNC_ATTRIBUTE_NUM_REGS      CU_FUNC_ATTRIBUTE_NUM_REGS
#define GPU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT  \
        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
#define GPU_FUNC_ATTRIBUTE_PTX_VERSION   CU_FUNC_ATTRIBUTE_PTX_VERSION
#define GPU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES  \
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define GPU_MEMCPY3D                     CUDA_MEMCPY3D_st
#define GPU_POINTER_ATTRIBUTE_ACCESS_FLAGS  \
        CU_POINTER_ATTRIBUTE_ACCESS_FLAGS
#define GPU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES  \
        CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
#define GPU_POINTER_ATTRIBUTE_BUFFER_ID  CU_POINTER_ATTRIBUTE_BUFFER_ID
#define GPU_POINTER_ATTRIBUTE_CONTEXT    CU_POINTER_ATTRIBUTE_CONTEXT
#define GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL  \
        CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define GPU_POINTER_ATTRIBUTE_DEVICE_POINTER  \
        CU_POINTER_ATTRIBUTE_DEVICE_POINTER
#define GPU_POINTER_ATTRIBUTE_HOST_POINTER  \
        CU_POINTER_ATTRIBUTE_HOST_POINTER
#define GPU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE  \
        CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
#define GPU_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE  \
        CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE
#define GPU_POINTER_ATTRIBUTE_IS_MANAGED CU_POINTER_ATTRIBUTE_IS_MANAGED
#define GPU_POINTER_ATTRIBUTE_MAPPED     CU_POINTER_ATTRIBUTE_MAPPED
#define GPU_POINTER_ATTRIBUTE_MEMORY_TYPE  \
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE
#define GPU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE  \
        CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#define GPU_POINTER_ATTRIBUTE_P2P_TOKENS CU_POINTER_ATTRIBUTE_P2P_TOKENS
#define GPU_POINTER_ATTRIBUTE_RANGE_SIZE CU_POINTER_ATTRIBUTE_RANGE_SIZE
#define GPU_POINTER_ATTRIBUTE_RANGE_START_ADDR  \
        CU_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define GPU_POINTER_ATTRIBUTE_SYNC_MEMOPS  \
        CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
#define GPU_RESOURCE_DESC                CUDA_RESOURCE_DESC_v1
#define GPU_RESOURCE_DESC_st             CUDA_RESOURCE_DESC_st
#define GPU_RESOURCE_TYPE_ARRAY          CU_RESOURCE_TYPE_ARRAY
#define GPU_RESOURCE_TYPE_LINEAR         CU_RESOURCE_TYPE_LINEAR
#define GPU_RESOURCE_TYPE_MIPMAPPED_ARRAY  \
        CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
#define GPU_RESOURCE_TYPE_PITCH2D        CU_RESOURCE_TYPE_PITCH2D
#define GPU_RESOURCE_VIEW_DESC           CUDA_RESOURCE_VIEW_DESC_v1
#define GPU_RESOURCE_VIEW_DESC_st        CUDA_RESOURCE_VIEW_DESC_st
#define GPU_RES_VIEW_FORMAT_FLOAT_1X16   CU_RES_VIEW_FORMAT_FLOAT_1X16
#define GPU_RES_VIEW_FORMAT_FLOAT_1X32   CU_RES_VIEW_FORMAT_FLOAT_1X32
#define GPU_RES_VIEW_FORMAT_FLOAT_2X16   CU_RES_VIEW_FORMAT_FLOAT_2X16
#define GPU_RES_VIEW_FORMAT_FLOAT_2X32   CU_RES_VIEW_FORMAT_FLOAT_2X32
#define GPU_RES_VIEW_FORMAT_FLOAT_4X16   CU_RES_VIEW_FORMAT_FLOAT_4X16
#define GPU_RES_VIEW_FORMAT_FLOAT_4X32   CU_RES_VIEW_FORMAT_FLOAT_4X32
#define GPU_RES_VIEW_FORMAT_NONE         CU_RES_VIEW_FORMAT_NONE
#define GPU_RES_VIEW_FORMAT_SIGNED_BC4   CU_RES_VIEW_FORMAT_SIGNED_BC4
#define GPU_RES_VIEW_FORMAT_SIGNED_BC5   CU_RES_VIEW_FORMAT_SIGNED_BC5
#define GPU_RES_VIEW_FORMAT_SIGNED_BC6H  CU_RES_VIEW_FORMAT_SIGNED_BC6H
#define GPU_RES_VIEW_FORMAT_SINT_1X16    CU_RES_VIEW_FORMAT_SINT_1X16
#define GPU_RES_VIEW_FORMAT_SINT_1X32    CU_RES_VIEW_FORMAT_SINT_1X32
#define GPU_RES_VIEW_FORMAT_SINT_1X8     CU_RES_VIEW_FORMAT_SINT_1X8
#define GPU_RES_VIEW_FORMAT_SINT_2X16    CU_RES_VIEW_FORMAT_SINT_2X16
#define GPU_RES_VIEW_FORMAT_SINT_2X32    CU_RES_VIEW_FORMAT_SINT_2X32
#define GPU_RES_VIEW_FORMAT_SINT_2X8     CU_RES_VIEW_FORMAT_SINT_2X8
#define GPU_RES_VIEW_FORMAT_SINT_4X16    CU_RES_VIEW_FORMAT_SINT_4X16
#define GPU_RES_VIEW_FORMAT_SINT_4X32    CU_RES_VIEW_FORMAT_SINT_4X32
#define GPU_RES_VIEW_FORMAT_SINT_4X8     CU_RES_VIEW_FORMAT_SINT_4X8
#define GPU_RES_VIEW_FORMAT_UINT_1X16    CU_RES_VIEW_FORMAT_UINT_1X16
#define GPU_RES_VIEW_FORMAT_UINT_1X32    CU_RES_VIEW_FORMAT_UINT_1X32
#define GPU_RES_VIEW_FORMAT_UINT_1X8     CU_RES_VIEW_FORMAT_UINT_1X8
#define GPU_RES_VIEW_FORMAT_UINT_2X16    CU_RES_VIEW_FORMAT_UINT_2X16
#define GPU_RES_VIEW_FORMAT_UINT_2X32    CU_RES_VIEW_FORMAT_UINT_2X32
#define GPU_RES_VIEW_FORMAT_UINT_2X8     CU_RES_VIEW_FORMAT_UINT_2X8
#define GPU_RES_VIEW_FORMAT_UINT_4X16    CU_RES_VIEW_FORMAT_UINT_4X16
#define GPU_RES_VIEW_FORMAT_UINT_4X32    CU_RES_VIEW_FORMAT_UINT_4X32
#define GPU_RES_VIEW_FORMAT_UINT_4X8     CU_RES_VIEW_FORMAT_UINT_4X8
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC1 CU_RES_VIEW_FORMAT_UNSIGNED_BC1
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC2 CU_RES_VIEW_FORMAT_UNSIGNED_BC2
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC3 CU_RES_VIEW_FORMAT_UNSIGNED_BC3
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC4 CU_RES_VIEW_FORMAT_UNSIGNED_BC4
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC5 CU_RES_VIEW_FORMAT_UNSIGNED_BC5
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC6H  \
        CU_RES_VIEW_FORMAT_UNSIGNED_BC6H
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC7 CU_RES_VIEW_FORMAT_UNSIGNED_BC7
#define GPU_TEXTURE_DESC                 CUDA_TEXTURE_DESC_v1
#define GPU_TEXTURE_DESC_st              CUDA_TEXTURE_DESC_st
#define GPU_TRSA_OVERRIDE_FORMAT         CU_TRSA_OVERRIDE_FORMAT
#define GPU_TRSF_NORMALIZED_COORDINATES  CU_TRSF_NORMALIZED_COORDINATES
#define GPU_TRSF_READ_AS_INTEGER         CU_TRSF_READ_AS_INTEGER
#define GPU_TRSF_SRGB                    CU_TRSF_SRGB
#define GPU_TR_ADDRESS_MODE_BORDER       CU_TR_ADDRESS_MODE_BORDER
#define GPU_TR_ADDRESS_MODE_CLAMP        CU_TR_ADDRESS_MODE_CLAMP
#define GPU_TR_ADDRESS_MODE_MIRROR       CU_TR_ADDRESS_MODE_MIRROR
#define GPU_TR_ADDRESS_MODE_WRAP         CU_TR_ADDRESS_MODE_WRAP
#define GPU_TR_FILTER_MODE_LINEAR        CU_TR_FILTER_MODE_LINEAR
#define GPU_TR_FILTER_MODE_POINT         CU_TR_FILTER_MODE_POINT
#define GPUaddress_mode                  CUaddress_mode
#define GPUaddress_mode_enum             CUaddress_mode_enum
#define GPUfilter_mode                   CUfilter_mode
#define GPUfilter_mode_enum              CUfilter_mode_enum
#define GPUresourceViewFormat            CUresourceViewFormat
#define GPUresourceViewFormat_enum       CUresourceViewFormat_enum
#define GPUresourcetype                  CUresourcetype
#define GPUresourcetype_enum             CUresourcetype_enum
#define gpuArray                         cudaArray
#define gpuArray_Format                  CUarray_format
#define gpuArray_const_t                 cudaArray_const_t
#define gpuArray_t                       CUarray
#define gpuChannelFormatDesc             cudaChannelFormatDesc
#define gpuChannelFormatKind             cudaChannelFormatKind
#define gpuChannelFormatKindFloat        cudaChannelFormatKindFloat
#define gpuChannelFormatKindNone         cudaChannelFormatKindNone
#define gpuChannelFormatKindSigned       cudaChannelFormatKindSigned
#define gpuChannelFormatKindUnsigned     cudaChannelFormatKindUnsigned
#define gpuDeviceptr_t                   CUdeviceptr
#define gpuExtent                        cudaExtent
#define gpuFunction_attribute            CUfunction_attribute
#define gpuMemcpy3DParms                 cudaMemcpy3DParms
#define gpuMemcpyDefault                 cudaMemcpyDefault
#define gpuMemcpyDeviceToDevice          cudaMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost            cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice            cudaMemcpyHostToDevice
#define gpuMemcpyHostToHost              cudaMemcpyHostToHost
#define gpuMemcpyKind                    cudaMemcpyKind
#define gpuMipmappedArray                cudaMipmappedArray
#define gpuMipmappedArray_const_t        cudaMipmappedArray_const_t
#define gpuMipmappedArray_t              CUmipmappedArray
#define gpuPitchedPtr                    cudaPitchedPtr
#define gpuPointer_attribute             CUpointer_attribute
#define gpuPos                           cudaPos
#define gpuResViewFormatFloat1           cudaResViewFormatFloat1
#define gpuResViewFormatFloat2           cudaResViewFormatFloat2
#define gpuResViewFormatFloat4           cudaResViewFormatFloat4
#define gpuResViewFormatHalf1            cudaResViewFormatHalf1
#define gpuResViewFormatHalf2            cudaResViewFormatHalf2
#define gpuResViewFormatHalf4            cudaResViewFormatHalf4
#define gpuResViewFormatNone             cudaResViewFormatNone
#define gpuResViewFormatSignedBlockCompressed4  \
        cudaResViewFormatSignedBlockCompressed4
#define gpuResViewFormatSignedBlockCompressed5  \
        cudaResViewFormatSignedBlockCompressed5
#define gpuResViewFormatSignedBlockCompressed6H  \
        cudaResViewFormatSignedBlockCompressed6H
#define gpuResViewFormatSignedChar1      cudaResViewFormatSignedChar1
#define gpuResViewFormatSignedChar2      cudaResViewFormatSignedChar2
#define gpuResViewFormatSignedChar4      cudaResViewFormatSignedChar4
#define gpuResViewFormatSignedInt1       cudaResViewFormatSignedInt1
#define gpuResViewFormatSignedInt2       cudaResViewFormatSignedInt2
#define gpuResViewFormatSignedInt4       cudaResViewFormatSignedInt4
#define gpuResViewFormatSignedShort1     cudaResViewFormatSignedShort1
#define gpuResViewFormatSignedShort2     cudaResViewFormatSignedShort2
#define gpuResViewFormatSignedShort4     cudaResViewFormatSignedShort4
#define gpuResViewFormatUnsignedBlockCompressed1  \
        cudaResViewFormatUnsignedBlockCompressed1
#define gpuResViewFormatUnsignedBlockCompressed2  \
        cudaResViewFormatUnsignedBlockCompressed2
#define gpuResViewFormatUnsignedBlockCompressed3  \
        cudaResViewFormatUnsignedBlockCompressed3
#define gpuResViewFormatUnsignedBlockCompressed4  \
        cudaResViewFormatUnsignedBlockCompressed4
#define gpuResViewFormatUnsignedBlockCompressed5  \
        cudaResViewFormatUnsignedBlockCompressed5
#define gpuResViewFormatUnsignedBlockCompressed6H  \
        cudaResViewFormatUnsignedBlockCompressed6H
#define gpuResViewFormatUnsignedBlockCompressed7  \
        cudaResViewFormatUnsignedBlockCompressed7
#define gpuResViewFormatUnsignedChar1    cudaResViewFormatUnsignedChar1
#define gpuResViewFormatUnsignedChar2    cudaResViewFormatUnsignedChar2
#define gpuResViewFormatUnsignedChar4    cudaResViewFormatUnsignedChar4
#define gpuResViewFormatUnsignedInt1     cudaResViewFormatUnsignedInt1
#define gpuResViewFormatUnsignedInt2     cudaResViewFormatUnsignedInt2
#define gpuResViewFormatUnsignedInt4     cudaResViewFormatUnsignedInt4
#define gpuResViewFormatUnsignedShort1   cudaResViewFormatUnsignedShort1
#define gpuResViewFormatUnsignedShort2   cudaResViewFormatUnsignedShort2
#define gpuResViewFormatUnsignedShort4   cudaResViewFormatUnsignedShort4
#define gpuResourceDesc                  cudaResourceDesc
#define gpuResourceType                  cudaResourceType
#define gpuResourceTypeArray             cudaResourceTypeArray
#define gpuResourceTypeLinear            cudaResourceTypeLinear
#define gpuResourceTypeMipmappedArray    cudaResourceTypeMipmappedArray
#define gpuResourceTypePitch2D           cudaResourceTypePitch2D
#define gpuResourceViewDesc              cudaResourceViewDesc
#define gpuResourceViewFormat            cudaResourceViewFormat
#define gpu_Memcpy2D                     CUDA_MEMCPY2D
#define make_gpuExtent                   make_cudaExtent
#define make_gpuPitchedPtr               make_cudaPitchedPtr
#define make_gpuPos                      make_cudaPos

/* surface_types.h */
#define gpuBoundaryModeClamp             cudaBoundaryModeClamp
#define gpuBoundaryModeTrap              cudaBoundaryModeTrap
#define gpuBoundaryModeZero              cudaBoundaryModeZero
#define gpuSurfaceBoundaryMode           cudaSurfaceBoundaryMode
#define gpuSurfaceObject_t               CUsurfObject

/* texture_types.h */
#define gpuAddressModeBorder             cudaAddressModeBorder
#define gpuAddressModeClamp              cudaAddressModeClamp
#define gpuAddressModeMirror             cudaAddressModeMirror
#define gpuAddressModeWrap               cudaAddressModeWrap
#define gpuFilterModeLinear              cudaFilterModeLinear
#define gpuFilterModePoint               cudaFilterModePoint
#define gpuReadModeElementType           cudaReadModeElementType
#define gpuReadModeNormalizedFloat       cudaReadModeNormalizedFloat
#define gpuTexRef                        CUtexref
#define gpuTextureAddressMode            cudaTextureAddressMode
#define gpuTextureDesc                   cudaTextureDesc
#define gpuTextureFilterMode             cudaTextureFilterMode
#define gpuTextureObject_t               CUtexObject
#define gpuTextureReadMode               cudaTextureReadMode
#define gpuTextureType1D                 cudaTextureType1D
#define gpuTextureType1DLayered          cudaTextureType1DLayered
#define gpuTextureType2D                 cudaTextureType2D
#define gpuTextureType2DLayered          cudaTextureType2DLayered
#define gpuTextureType3D                 cudaTextureType3D
#define gpuTextureTypeCubemap            cudaTextureTypeCubemap
#define gpuTextureTypeCubemapLayered     cudaTextureTypeCubemapLayered


#endif
