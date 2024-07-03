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

#define GPU_IPC_HANDLE_SIZE              CUDA_IPC_HANDLE_SIZE
#define GPU_LAUNCH_PARAM_BUFFER_POINTER  CU_LAUNCH_PARAM_BUFFER_POINTER
#define GPU_LAUNCH_PARAM_BUFFER_SIZE     CU_LAUNCH_PARAM_BUFFER_SIZE
#define GPU_LAUNCH_PARAM_END             CU_LAUNCH_PARAM_END
#define gpuAccessPolicyWindow            cudaAccessPolicyWindow
#define gpuAccessProperty                cudaAccessProperty
#define gpuAccessPropertyNormal          cudaAccessPropertyNormal
#define gpuAccessPropertyPersisting      cudaAccessPropertyPersisting
#define gpuAccessPropertyStreaming       cudaAccessPropertyStreaming
#define gpuArray3DCreate                 cuArray3DCreate_v2
#define gpuArray3DGetDescriptor          cuArray3DGetDescriptor_v2
#define gpuArrayCreate                   cuArrayCreate_v2
#define gpuArrayCubemap                  cudaArrayCubemap
#define gpuArrayDefault                  cudaArrayDefault
#define gpuArrayDestroy                  cuArrayDestroy
#define gpuArrayGetDescriptor            cuArrayGetDescriptor_v2
#define gpuArrayGetInfo                  cudaArrayGetInfo
#define gpuArrayLayered                  cudaArrayLayered
#define gpuArrayMapInfo                  CUarrayMapInfo_v1
#define gpuArraySparseSubresourceType    CUarraySparseSubresourceType
#define gpuArraySparseSubresourceTypeMiptail  \
        CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL
#define gpuArraySparseSubresourceTypeSparseLevel  \
        CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL
#define gpuArraySurfaceLoadStore         cudaArraySurfaceLoadStore
#define gpuArrayTextureGather            cudaArrayTextureGather
#define gpuBindTexture                   cudaBindTexture
#define gpuBindTexture2D                 cudaBindTexture2D
#define gpuBindTextureToArray            cudaBindTextureToArray
#define gpuBindTextureToMipmappedArray   cudaBindTextureToMipmappedArray
#define gpuChooseDevice                  cudaChooseDevice
#define gpuComputeMode                   cudaComputeMode
#define gpuComputeModeDefault            cudaComputeModeDefault
#define gpuComputeModeExclusive          cudaComputeModeExclusive
#define gpuComputeModeExclusiveProcess   cudaComputeModeExclusiveProcess
#define gpuComputeModeProhibited         cudaComputeModeProhibited
#define gpuCooperativeLaunchMultiDeviceNoPostSync  \
        cudaCooperativeLaunchMultiDeviceNoPostSync
#define gpuCooperativeLaunchMultiDeviceNoPreSync  \
        cudaCooperativeLaunchMultiDeviceNoPreSync
#define gpuCpuDeviceId                   cudaCpuDeviceId
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
#define gpuDestroyExternalMemory         cudaDestroyExternalMemory
#define gpuDestroyExternalSemaphore      cudaDestroyExternalSemaphore
#define gpuDestroySurfaceObject          cudaDestroySurfaceObject
#define gpuDestroyTextureObject          cudaDestroyTextureObject
#define gpuDevP2PAttrAccessSupported     cudaDevP2PAttrAccessSupported
#define gpuDevP2PAttrHipArrayAccessSupported  \
        cudaDevP2PAttrCudaArrayAccessSupported
#define gpuDevP2PAttrNativeAtomicSupported  \
        cudaDevP2PAttrNativeAtomicSupported
#define gpuDevP2PAttrPerformanceRank     cudaDevP2PAttrPerformanceRank
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
#define gpuDeviceCanAccessPeer           cudaDeviceCanAccessPeer
#define gpuDeviceComputeCapability       cuDeviceComputeCapability
#define gpuDeviceDisablePeerAccess       cudaDeviceDisablePeerAccess
#define gpuDeviceEnablePeerAccess        cudaDeviceEnablePeerAccess
#define gpuDeviceGet                     cuDeviceGet
#define gpuDeviceGetAttribute            cudaDeviceGetAttribute
#define gpuDeviceGetByPCIBusId           cudaDeviceGetByPCIBusId
#define gpuDeviceGetCacheConfig          cudaDeviceGetCacheConfig
#define gpuDeviceGetDefaultMemPool       cudaDeviceGetDefaultMemPool
#define gpuDeviceGetGraphMemAttribute    cudaDeviceGetGraphMemAttribute
#define gpuDeviceGetLimit                cudaDeviceGetLimit
#define gpuDeviceGetMemPool              cudaDeviceGetMemPool
#define gpuDeviceGetName                 cuDeviceGetName
#define gpuDeviceGetP2PAttribute         cudaDeviceGetP2PAttribute
#define gpuDeviceGetPCIBusId             cudaDeviceGetPCIBusId
#define gpuDeviceGetSharedMemConfig      cudaDeviceGetSharedMemConfig
#define gpuDeviceGetStreamPriorityRange  cudaDeviceGetStreamPriorityRange
#define gpuDeviceGetUuid                 cuDeviceGetUuid_v2
#define gpuDeviceGraphMemTrim            cudaDeviceGraphMemTrim
#define gpuDeviceLmemResizeToMax         cudaDeviceLmemResizeToMax
#define gpuDeviceMapHost                 cudaDeviceMapHost
#define gpuDeviceP2PAttr                 cudaDeviceP2PAttr
#define gpuDevicePrimaryCtxGetState      cuDevicePrimaryCtxGetState
#define gpuDevicePrimaryCtxRelease       cuDevicePrimaryCtxRelease_v2
#define gpuDevicePrimaryCtxReset         cuDevicePrimaryCtxReset_v2
#define gpuDevicePrimaryCtxRetain        cuDevicePrimaryCtxRetain
#define gpuDevicePrimaryCtxSetFlags      cuDevicePrimaryCtxSetFlags_v2
#define gpuDeviceProp_t                  cudaDeviceProp
#define gpuDeviceReset                   cudaDeviceReset
#define gpuDeviceScheduleAuto            cudaDeviceScheduleAuto
#define gpuDeviceScheduleBlockingSync    cudaDeviceScheduleBlockingSync
#define gpuDeviceScheduleMask            cudaDeviceScheduleMask
#define gpuDeviceScheduleSpin            cudaDeviceScheduleSpin
#define gpuDeviceScheduleYield           cudaDeviceScheduleYield
#define gpuDeviceSetCacheConfig          cudaDeviceSetCacheConfig
#define gpuDeviceSetGraphMemAttribute    cudaDeviceSetGraphMemAttribute
#define gpuDeviceSetLimit                cudaDeviceSetLimit
#define gpuDeviceSetMemPool              cudaDeviceSetMemPool
#define gpuDeviceSetSharedMemConfig      cudaDeviceSetSharedMemConfig
#define gpuDeviceSynchronize             cudaDeviceSynchronize
#define gpuDeviceTotalMem                cuDeviceTotalMem_v2
#define gpuDevice_t                      CUdevice_v1
#define gpuDriverGetVersion              cudaDriverGetVersion
#define gpuDrvGetErrorName               cuGetErrorName
#define gpuDrvGetErrorString             cuGetErrorString
#define gpuDrvMemcpy2DUnaligned          cuMemcpy2DUnaligned_v2
#define gpuDrvMemcpy3D                   cuMemcpy3D_v2
#define gpuDrvMemcpy3DAsync              cuMemcpy3DAsync_v2
#define gpuDrvPointerGetAttributes       cuPointerGetAttributes
#define gpuErrorAlreadyAcquired          cudaErrorAlreadyAcquired
#define gpuErrorAlreadyMapped            cudaErrorAlreadyMapped
#define gpuErrorArrayIsMapped            cudaErrorArrayIsMapped
#define gpuErrorAssert                   cudaErrorAssert
#define gpuErrorCapturedEvent            cudaErrorCapturedEvent
#define gpuErrorContextAlreadyCurrent    CUDA_ERROR_CONTEXT_ALREADY_CURRENT
#define gpuErrorContextAlreadyInUse      cudaErrorDeviceAlreadyInUse
#define gpuErrorContextIsDestroyed       cudaErrorContextIsDestroyed
#define gpuErrorCooperativeLaunchTooLarge  \
        cudaErrorCooperativeLaunchTooLarge
#define gpuErrorDeinitialized            cudaErrorCudartUnloading
#define gpuErrorECCNotCorrectable        cudaErrorECCUncorrectable
#define gpuErrorFileNotFound             cudaErrorFileNotFound
#define gpuErrorGraphExecUpdateFailure   cudaErrorGraphExecUpdateFailure
#define gpuErrorHostMemoryAlreadyRegistered  \
        cudaErrorHostMemoryAlreadyRegistered
#define gpuErrorHostMemoryNotRegistered  cudaErrorHostMemoryNotRegistered
#define gpuErrorIllegalAddress           cudaErrorIllegalAddress
#define gpuErrorIllegalState             cudaErrorIllegalState
#define gpuErrorInsufficientDriver       cudaErrorInsufficientDriver
#define gpuErrorInvalidConfiguration     cudaErrorInvalidConfiguration
#define gpuErrorInvalidContext           cudaErrorDeviceUninitialized
#define gpuErrorInvalidDevice            cudaErrorInvalidDevice
#define gpuErrorInvalidDeviceFunction    cudaErrorInvalidDeviceFunction
#define gpuErrorInvalidDevicePointer     cudaErrorInvalidDevicePointer
#define gpuErrorInvalidGraphicsContext   cudaErrorInvalidGraphicsContext
#define gpuErrorInvalidHandle            cudaErrorInvalidResourceHandle
#define gpuErrorInvalidImage             cudaErrorInvalidKernelImage
#define gpuErrorInvalidKernelFile        cudaErrorInvalidPtx
#define gpuErrorInvalidMemcpyDirection   cudaErrorInvalidMemcpyDirection
#define gpuErrorInvalidPitchValue        cudaErrorInvalidPitchValue
#define gpuErrorInvalidSource            cudaErrorInvalidSource
#define gpuErrorInvalidSymbol            cudaErrorInvalidSymbol
#define gpuErrorInvalidValue             cudaErrorInvalidValue
#define gpuErrorLaunchFailure            cudaErrorLaunchFailure
#define gpuErrorLaunchOutOfResources     cudaErrorLaunchOutOfResources
#define gpuErrorLaunchTimeOut            cudaErrorLaunchTimeout
#define gpuErrorMapFailed                cudaErrorMapBufferObjectFailed
#define gpuErrorMissingConfiguration     cudaErrorMissingConfiguration
#define gpuErrorNoBinaryForGpu           cudaErrorNoKernelImageForDevice
#define gpuErrorNoDevice                 cudaErrorNoDevice
#define gpuErrorNotFound                 cudaErrorSymbolNotFound
#define gpuErrorNotInitialized           cudaErrorInitializationError
#define gpuErrorNotMapped                cudaErrorNotMapped
#define gpuErrorNotMappedAsArray         cudaErrorNotMappedAsArray
#define gpuErrorNotMappedAsPointer       cudaErrorNotMappedAsPointer
#define gpuErrorNotReady                 cudaErrorNotReady
#define gpuErrorNotSupported             cudaErrorNotSupported
#define gpuErrorOperatingSystem          cudaErrorOperatingSystem
#define gpuErrorOutOfMemory              cudaErrorMemoryAllocation
#define gpuErrorPeerAccessAlreadyEnabled cudaErrorPeerAccessAlreadyEnabled
#define gpuErrorPeerAccessNotEnabled     cudaErrorPeerAccessNotEnabled
#define gpuErrorPeerAccessUnsupported    cudaErrorPeerAccessUnsupported
#define gpuErrorPriorLaunchFailure       cudaErrorPriorLaunchFailure
#define gpuErrorProfilerAlreadyStarted   cudaErrorProfilerAlreadyStarted
#define gpuErrorProfilerAlreadyStopped   cudaErrorProfilerAlreadyStopped
#define gpuErrorProfilerDisabled         cudaErrorProfilerDisabled
#define gpuErrorProfilerNotInitialized   cudaErrorProfilerNotInitialized
#define gpuErrorSetOnActiveProcess       cudaErrorSetOnActiveProcess
#define gpuErrorSharedObjectInitFailed   cudaErrorSharedObjectInitFailed
#define gpuErrorSharedObjectSymbolNotFound  \
        cudaErrorSharedObjectSymbolNotFound
#define gpuErrorStreamCaptureImplicit    cudaErrorStreamCaptureImplicit
#define gpuErrorStreamCaptureInvalidated cudaErrorStreamCaptureInvalidated
#define gpuErrorStreamCaptureIsolation   cudaErrorStreamCaptureIsolation
#define gpuErrorStreamCaptureMerge       cudaErrorStreamCaptureMerge
#define gpuErrorStreamCaptureUnjoined    cudaErrorStreamCaptureUnjoined
#define gpuErrorStreamCaptureUnmatched   cudaErrorStreamCaptureUnmatched
#define gpuErrorStreamCaptureUnsupported cudaErrorStreamCaptureUnsupported
#define gpuErrorStreamCaptureWrongThread cudaErrorStreamCaptureWrongThread
#define gpuErrorUnknown                  cudaErrorUnknown
#define gpuErrorUnmapFailed              cudaErrorUnmapBufferObjectFailed
#define gpuErrorUnsupportedLimit         cudaErrorUnsupportedLimit
#define gpuError_t                       cudaError_t
#define gpuEventBlockingSync             cudaEventBlockingSync
#define gpuEventCreate                   cudaEventCreate
#define gpuEventCreateWithFlags          cudaEventCreateWithFlags
#define gpuEventDefault                  cudaEventDefault
#define gpuEventDestroy                  cudaEventDestroy
#define gpuEventDisableTiming            cudaEventDisableTiming
#define gpuEventElapsedTime              cudaEventElapsedTime
#define gpuEventInterprocess             cudaEventInterprocess
#define gpuEventQuery                    cudaEventQuery
#define gpuEventRecord                   cudaEventRecord
#define gpuEventSynchronize              cudaEventSynchronize
#define gpuEvent_t                       cudaEvent_t
#define gpuExternalMemoryBufferDesc      cudaExternalMemoryBufferDesc
#define gpuExternalMemoryBufferDesc_st   CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
#define gpuExternalMemoryDedicated       cudaExternalMemoryDedicated
#define gpuExternalMemoryGetMappedBuffer cudaExternalMemoryGetMappedBuffer
#define gpuExternalMemoryHandleDesc      cudaExternalMemoryHandleDesc
#define gpuExternalMemoryHandleDesc_st   CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
#define gpuExternalMemoryHandleType      cudaExternalMemoryHandleType
#define gpuExternalMemoryHandleTypeD3D11Resource  \
        cudaExternalMemoryHandleTypeD3D11Resource
#define gpuExternalMemoryHandleTypeD3D11ResourceKmt  \
        cudaExternalMemoryHandleTypeD3D11ResourceKmt
#define gpuExternalMemoryHandleTypeD3D12Heap  \
        cudaExternalMemoryHandleTypeD3D12Heap
#define gpuExternalMemoryHandleTypeD3D12Resource  \
        cudaExternalMemoryHandleTypeD3D12Resource
#define gpuExternalMemoryHandleTypeOpaqueFd  \
        cudaExternalMemoryHandleTypeOpaqueFd
#define gpuExternalMemoryHandleTypeOpaqueWin32  \
        cudaExternalMemoryHandleTypeOpaqueWin32
#define gpuExternalMemoryHandleTypeOpaqueWin32Kmt  \
        cudaExternalMemoryHandleTypeOpaqueWin32Kmt
#define gpuExternalMemoryHandleType_enum CUexternalMemoryHandleType_enum
#define gpuExternalMemory_t              cudaExternalMemory_t
#define gpuExternalSemaphoreHandleDesc   cudaExternalSemaphoreHandleDesc
#define gpuExternalSemaphoreHandleDesc_st  \
        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
#define gpuExternalSemaphoreHandleType   cudaExternalSemaphoreHandleType
#define gpuExternalSemaphoreHandleTypeD3D12Fence  \
        cudaExternalSemaphoreHandleTypeD3D12Fence
#define gpuExternalSemaphoreHandleTypeOpaqueFd  \
        cudaExternalSemaphoreHandleTypeOpaqueFd
#define gpuExternalSemaphoreHandleTypeOpaqueWin32  \
        cudaExternalSemaphoreHandleTypeOpaqueWin32
#define gpuExternalSemaphoreHandleTypeOpaqueWin32Kmt  \
        cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
#define gpuExternalSemaphoreHandleType_enum  \
        CUexternalSemaphoreHandleType_enum
#define gpuExternalSemaphoreSignalParams cudaExternalSemaphoreSignalParams_v1
#define gpuExternalSemaphoreSignalParams_st  \
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
#define gpuExternalSemaphoreWaitParams   cudaExternalSemaphoreWaitParams_v1
#define gpuExternalSemaphoreWaitParams_st  \
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
#define gpuExternalSemaphore_t           cudaExternalSemaphore_t
#define gpuFree                          cudaFree
#define gpuFreeArray                     cudaFreeArray
#define gpuFreeAsync                     cudaFreeAsync
#define gpuFreeMipmappedArray            cudaFreeMipmappedArray
#define gpuFuncAttribute                 cudaFuncAttribute
#define gpuFuncAttributeMax              cudaFuncAttributeMax
#define gpuFuncAttributeMaxDynamicSharedMemorySize  \
        cudaFuncAttributeMaxDynamicSharedMemorySize
#define gpuFuncAttributePreferredSharedMemoryCarveout  \
        cudaFuncAttributePreferredSharedMemoryCarveout
#define gpuFuncAttributes                cudaFuncAttributes
#define gpuFuncCachePreferEqual          cudaFuncCachePreferEqual
#define gpuFuncCachePreferL1             cudaFuncCachePreferL1
#define gpuFuncCachePreferNone           cudaFuncCachePreferNone
#define gpuFuncCachePreferShared         cudaFuncCachePreferShared
#define gpuFuncCache_t                   cudaFuncCache
#define gpuFuncGetAttribute              cuFuncGetAttribute
#define gpuFuncGetAttributes             cudaFuncGetAttributes
#define gpuFuncSetAttribute              cudaFuncSetAttribute
#define gpuFuncSetCacheConfig            cudaFuncSetCacheConfig
#define gpuFuncSetSharedMemConfig        cudaFuncSetSharedMemConfig
#define gpuFunctionLaunchParams          CUDA_LAUNCH_PARAMS_v1
#define gpuFunctionLaunchParams_t        CUDA_LAUNCH_PARAMS_st
#define gpuFunction_t                    cudaFunction_t
#define gpuGLDeviceList                  CUGLDeviceList
#define gpuGLDeviceListAll               CU_GL_DEVICE_LIST_ALL
#define gpuGLDeviceListCurrentFrame      CU_GL_DEVICE_LIST_CURRENT_FRAME
#define gpuGLDeviceListNextFrame         CU_GL_DEVICE_LIST_NEXT_FRAME
#define gpuGLGetDevices                  cuGLGetDevices
#define gpuGetChannelDesc                cudaGetChannelDesc
#define gpuGetDevice                     cudaGetDevice
#define gpuGetDeviceCount                cudaGetDeviceCount
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
#define gpuGraphAddChildGraphNode        cudaGraphAddChildGraphNode
#define gpuGraphAddDependencies          cudaGraphAddDependencies
#define gpuGraphAddEmptyNode             cudaGraphAddEmptyNode
#define gpuGraphAddEventRecordNode       cudaGraphAddEventRecordNode
#define gpuGraphAddEventWaitNode         cudaGraphAddEventWaitNode
#define gpuGraphAddHostNode              cudaGraphAddHostNode
#define gpuGraphAddKernelNode            cudaGraphAddKernelNode
#define gpuGraphAddMemAllocNode          cudaGraphAddMemAllocNode
#define gpuGraphAddMemFreeNode           cudaGraphAddMemFreeNode
#define gpuGraphAddMemcpyNode            cudaGraphAddMemcpyNode
#define gpuGraphAddMemcpyNode1D          cudaGraphAddMemcpyNode1D
#define gpuGraphAddMemcpyNodeFromSymbol  cudaGraphAddMemcpyNodeFromSymbol
#define gpuGraphAddMemcpyNodeToSymbol    cudaGraphAddMemcpyNodeToSymbol
#define gpuGraphAddMemsetNode            cudaGraphAddMemsetNode
#define gpuGraphChildGraphNodeGetGraph   cudaGraphChildGraphNodeGetGraph
#define gpuGraphClone                    cudaGraphClone
#define gpuGraphCreate                   cudaGraphCreate
#define gpuGraphDebugDotFlags            cudaGraphDebugDotFlags
#define gpuGraphDebugDotFlagsEventNodeParams  \
        cudaGraphDebugDotFlagsEventNodeParams
#define gpuGraphDebugDotFlagsExtSemasSignalNodeParams  \
        cudaGraphDebugDotFlagsExtSemasSignalNodeParams
#define gpuGraphDebugDotFlagsExtSemasWaitNodeParams  \
        cudaGraphDebugDotFlagsExtSemasWaitNodeParams
#define gpuGraphDebugDotFlagsHandles     cudaGraphDebugDotFlagsHandles
#define gpuGraphDebugDotFlagsHostNodeParams  \
        cudaGraphDebugDotFlagsHostNodeParams
#define gpuGraphDebugDotFlagsKernelNodeAttributes  \
        cudaGraphDebugDotFlagsKernelNodeAttributes
#define gpuGraphDebugDotFlagsKernelNodeParams  \
        cudaGraphDebugDotFlagsKernelNodeParams
#define gpuGraphDebugDotFlagsMemcpyNodeParams  \
        cudaGraphDebugDotFlagsMemcpyNodeParams
#define gpuGraphDebugDotFlagsMemsetNodeParams  \
        cudaGraphDebugDotFlagsMemsetNodeParams
#define gpuGraphDebugDotFlagsVerbose     cudaGraphDebugDotFlagsVerbose
#define gpuGraphDebugDotPrint            cudaGraphDebugDotPrint
#define gpuGraphDestroy                  cudaGraphDestroy
#define gpuGraphDestroyNode              cudaGraphDestroyNode
#define gpuGraphEventRecordNodeGetEvent  cudaGraphEventRecordNodeGetEvent
#define gpuGraphEventRecordNodeSetEvent  cudaGraphEventRecordNodeSetEvent
#define gpuGraphEventWaitNodeGetEvent    cudaGraphEventWaitNodeGetEvent
#define gpuGraphEventWaitNodeSetEvent    cudaGraphEventWaitNodeSetEvent
#define gpuGraphExecChildGraphNodeSetParams  \
        cudaGraphExecChildGraphNodeSetParams
#define gpuGraphExecDestroy              cudaGraphExecDestroy
#define gpuGraphExecEventRecordNodeSetEvent  \
        cudaGraphExecEventRecordNodeSetEvent
#define gpuGraphExecEventWaitNodeSetEvent  \
        cudaGraphExecEventWaitNodeSetEvent
#define gpuGraphExecHostNodeSetParams    cudaGraphExecHostNodeSetParams
#define gpuGraphExecKernelNodeSetParams  cudaGraphExecKernelNodeSetParams
#define gpuGraphExecMemcpyNodeSetParams  cudaGraphExecMemcpyNodeSetParams
#define gpuGraphExecMemcpyNodeSetParams1D  \
        cudaGraphExecMemcpyNodeSetParams1D
#define gpuGraphExecMemcpyNodeSetParamsFromSymbol  \
        cudaGraphExecMemcpyNodeSetParamsFromSymbol
#define gpuGraphExecMemcpyNodeSetParamsToSymbol  \
        cudaGraphExecMemcpyNodeSetParamsToSymbol
#define gpuGraphExecMemsetNodeSetParams  cudaGraphExecMemsetNodeSetParams
#define gpuGraphExecUpdate               cudaGraphExecUpdate
#define gpuGraphExecUpdateError          cudaGraphExecUpdateError
#define gpuGraphExecUpdateErrorFunctionChanged  \
        cudaGraphExecUpdateErrorFunctionChanged
#define gpuGraphExecUpdateErrorNodeTypeChanged  \
        cudaGraphExecUpdateErrorNodeTypeChanged
#define gpuGraphExecUpdateErrorNotSupported  \
        cudaGraphExecUpdateErrorNotSupported
#define gpuGraphExecUpdateErrorParametersChanged  \
        cudaGraphExecUpdateErrorParametersChanged
#define gpuGraphExecUpdateErrorTopologyChanged  \
        cudaGraphExecUpdateErrorTopologyChanged
#define gpuGraphExecUpdateErrorUnsupportedFunctionChange  \
        cudaGraphExecUpdateErrorUnsupportedFunctionChange
#define gpuGraphExecUpdateResult         cudaGraphExecUpdateResult
#define gpuGraphExecUpdateSuccess        cudaGraphExecUpdateSuccess
#define gpuGraphExec_t                   cudaGraphExec_t
#define gpuGraphGetEdges                 cudaGraphGetEdges
#define gpuGraphGetNodes                 cudaGraphGetNodes
#define gpuGraphGetRootNodes             cudaGraphGetRootNodes
#define gpuGraphHostNodeGetParams        cudaGraphHostNodeGetParams
#define gpuGraphHostNodeSetParams        cudaGraphHostNodeSetParams
#define gpuGraphInstantiate              cudaGraphInstantiate
#define gpuGraphInstantiateFlagAutoFreeOnLaunch  \
        cudaGraphInstantiateFlagAutoFreeOnLaunch
#define gpuGraphInstantiateFlagDeviceLaunch  \
        cudaGraphInstantiateFlagDeviceLaunch
#define gpuGraphInstantiateFlagUpload    cudaGraphInstantiateFlagUpload
#define gpuGraphInstantiateFlagUseNodePriority  \
        cudaGraphInstantiateFlagUseNodePriority
#define gpuGraphInstantiateFlags         cudaGraphInstantiateFlags
#define gpuGraphInstantiateWithFlags     cudaGraphInstantiateWithFlags
#define gpuGraphKernelNodeCopyAttributes cudaGraphKernelNodeCopyAttributes
#define gpuGraphKernelNodeGetAttribute   cudaGraphKernelNodeGetAttribute
#define gpuGraphKernelNodeGetParams      cudaGraphKernelNodeGetParams
#define gpuGraphKernelNodeSetAttribute   cudaGraphKernelNodeSetAttribute
#define gpuGraphKernelNodeSetParams      cudaGraphKernelNodeSetParams
#define gpuGraphLaunch                   cudaGraphLaunch
#define gpuGraphMemAllocNodeGetParams    cudaGraphMemAllocNodeGetParams
#define gpuGraphMemAttrReservedMemCurrent  \
        cudaGraphMemAttrReservedMemCurrent
#define gpuGraphMemAttrReservedMemHigh   cudaGraphMemAttrReservedMemHigh
#define gpuGraphMemAttrUsedMemCurrent    cudaGraphMemAttrUsedMemCurrent
#define gpuGraphMemAttrUsedMemHigh       cudaGraphMemAttrUsedMemHigh
#define gpuGraphMemAttributeType         cudaGraphMemAttributeType
#define gpuGraphMemFreeNodeGetParams     cudaGraphMemFreeNodeGetParams
#define gpuGraphMemcpyNodeGetParams      cudaGraphMemcpyNodeGetParams
#define gpuGraphMemcpyNodeSetParams      cudaGraphMemcpyNodeSetParams
#define gpuGraphMemcpyNodeSetParams1D    cudaGraphMemcpyNodeSetParams1D
#define gpuGraphMemcpyNodeSetParamsFromSymbol  \
        cudaGraphMemcpyNodeSetParamsFromSymbol
#define gpuGraphMemcpyNodeSetParamsToSymbol  \
        cudaGraphMemcpyNodeSetParamsToSymbol
#define gpuGraphMemsetNodeGetParams      cudaGraphMemsetNodeGetParams
#define gpuGraphMemsetNodeSetParams      cudaGraphMemsetNodeSetParams
#define gpuGraphNodeFindInClone          cudaGraphNodeFindInClone
#define gpuGraphNodeGetDependencies      cudaGraphNodeGetDependencies
#define gpuGraphNodeGetDependentNodes    cudaGraphNodeGetDependentNodes
#define gpuGraphNodeGetEnabled           cudaGraphNodeGetEnabled
#define gpuGraphNodeGetType              cudaGraphNodeGetType
#define gpuGraphNodeSetEnabled           cudaGraphNodeSetEnabled
#define gpuGraphNodeType                 cudaGraphNodeType
#define gpuGraphNodeTypeCount            cudaGraphNodeTypeCount
#define gpuGraphNodeTypeEmpty            cudaGraphNodeTypeEmpty
#define gpuGraphNodeTypeEventRecord      cudaGraphNodeTypeEventRecord
#define gpuGraphNodeTypeExtSemaphoreSignal  \
        cudaGraphNodeTypeExtSemaphoreSignal
#define gpuGraphNodeTypeExtSemaphoreWait cudaGraphNodeTypeExtSemaphoreWait
#define gpuGraphNodeTypeGraph            cudaGraphNodeTypeGraph
#define gpuGraphNodeTypeHost             cudaGraphNodeTypeHost
#define gpuGraphNodeTypeKernel           cudaGraphNodeTypeKernel
#define gpuGraphNodeTypeMemAlloc         cudaGraphNodeTypeMemAlloc
#define gpuGraphNodeTypeMemFree          cudaGraphNodeTypeMemFree
#define gpuGraphNodeTypeMemcpy           cudaGraphNodeTypeMemcpy
#define gpuGraphNodeTypeMemset           cudaGraphNodeTypeMemset
#define gpuGraphNodeTypeWaitEvent        cudaGraphNodeTypeWaitEvent
#define gpuGraphNode_t                   cudaGraphNode_t
#define gpuGraphReleaseUserObject        cudaGraphReleaseUserObject
#define gpuGraphRemoveDependencies       cudaGraphRemoveDependencies
#define gpuGraphRetainUserObject         cudaGraphRetainUserObject
#define gpuGraphUpload                   cudaGraphUpload
#define gpuGraphUserObjectMove           cudaGraphUserObjectMove
#define gpuGraph_t                       cudaGraph_t
#define gpuGraphicsGLRegisterBuffer      cuGraphicsGLRegisterBuffer
#define gpuGraphicsGLRegisterImage       cuGraphicsGLRegisterImage
#define gpuGraphicsMapResources          cudaGraphicsMapResources
#define gpuGraphicsRegisterFlags         cudaGraphicsRegisterFlags
#define gpuGraphicsRegisterFlagsNone     cudaGraphicsRegisterFlagsNone
#define gpuGraphicsRegisterFlagsReadOnly cudaGraphicsRegisterFlagsReadOnly
#define gpuGraphicsRegisterFlagsSurfaceLoadStore  \
        cudaGraphicsRegisterFlagsSurfaceLoadStore
#define gpuGraphicsRegisterFlagsTextureGather  \
        cudaGraphicsRegisterFlagsTextureGather
#define gpuGraphicsRegisterFlagsWriteDiscard  \
        cudaGraphicsRegisterFlagsWriteDiscard
#define gpuGraphicsResource              cudaGraphicsResource
#define gpuGraphicsResourceGetMappedPointer  \
        cudaGraphicsResourceGetMappedPointer
#define gpuGraphicsResource_t            cudaGraphicsResource_t
#define gpuGraphicsSubResourceGetMappedArray  \
        cudaGraphicsSubResourceGetMappedArray
#define gpuGraphicsUnmapResources        cudaGraphicsUnmapResources
#define gpuGraphicsUnregisterResource    cudaGraphicsUnregisterResource
#define gpuHostAlloc                     cudaHostAlloc
#define gpuHostFn_t                      cudaHostFn_t
#define gpuHostFree                      cudaFreeHost
#define gpuHostGetDevicePointer          cudaHostGetDevicePointer
#define gpuHostGetFlags                  cudaHostGetFlags
#define gpuHostMalloc                    cudaMallocHost
#define gpuHostMallocDefault             cudaHostAllocDefault
#define gpuHostMallocMapped              cudaHostAllocMapped
#define gpuHostMallocPortable            cudaHostAllocPortable
#define gpuHostMallocWriteCombined       cudaHostAllocWriteCombined
#define gpuHostNodeParams                cudaHostNodeParams
#define gpuHostRegister                  cudaHostRegister
#define gpuHostRegisterDefault           cudaHostRegisterDefault
#define gpuHostRegisterIoMemory          cudaHostRegisterIoMemory
#define gpuHostRegisterMapped            cudaHostRegisterMapped
#define gpuHostRegisterPortable          cudaHostRegisterPortable
#define gpuHostRegisterReadOnly          cudaHostRegisterReadOnly
#define gpuHostUnregister                cudaHostUnregister
#define gpuImportExternalMemory          cudaImportExternalMemory
#define gpuImportExternalSemaphore       cudaImportExternalSemaphore
#define gpuInit                          cuInit
#define gpuInvalidDeviceId               cudaInvalidDeviceId
#define gpuIpcCloseMemHandle             cudaIpcCloseMemHandle
#define gpuIpcEventHandle_st             cudaIpcEventHandle_st
#define gpuIpcEventHandle_t              cudaIpcEventHandle_t
#define gpuIpcGetEventHandle             cudaIpcGetEventHandle
#define gpuIpcGetMemHandle               cudaIpcGetMemHandle
#define gpuIpcMemHandle_st               cudaIpcMemHandle_st
#define gpuIpcMemHandle_t                cudaIpcMemHandle_t
#define gpuIpcMemLazyEnablePeerAccess    cudaIpcMemLazyEnablePeerAccess
#define gpuIpcOpenEventHandle            cudaIpcOpenEventHandle
#define gpuIpcOpenMemHandle              cudaIpcOpenMemHandle
#define gpuJitOption                     CUjit_option
#define gpuKernelNodeAttrID              cudaKernelNodeAttrID
#define gpuKernelNodeAttrValue           cudaKernelNodeAttrValue
#define gpuKernelNodeAttributeAccessPolicyWindow  \
        cudaKernelNodeAttributeAccessPolicyWindow
#define gpuKernelNodeAttributeCooperative  \
        cudaKernelNodeAttributeCooperative
#define gpuKernelNodeParams              cudaKernelNodeParams
#define gpuLaunchCooperativeKernel       cudaLaunchCooperativeKernel
#define gpuLaunchCooperativeKernelMultiDevice  \
        cudaLaunchCooperativeKernelMultiDevice
#define gpuLaunchHostFunc                cudaLaunchHostFunc
#define gpuLaunchKernel                  cudaLaunchKernel
#define gpuLaunchParams                  cudaLaunchParams
#define gpuLimitMallocHeapSize           cudaLimitMallocHeapSize
#define gpuLimitPrintfFifoSize           cudaLimitPrintfFifoSize
#define gpuLimitStackSize                cudaLimitStackSize
#define gpuLimit_t                       cudaLimit
#define gpuMalloc                        cudaMalloc
#define gpuMalloc3D                      cudaMalloc3D
#define gpuMalloc3DArray                 cudaMalloc3DArray
#define gpuMallocArray                   cudaMallocArray
#define gpuMallocAsync                   cudaMallocAsync
#define gpuMallocFromPoolAsync           cudaMallocFromPoolAsync
#define gpuMallocManaged                 cudaMallocManaged
#define gpuMallocMipmappedArray          cudaMallocMipmappedArray
#define gpuMallocPitch                   cudaMallocPitch
#define gpuMemAccessDesc                 cudaMemAccessDesc
#define gpuMemAccessFlags                cudaMemAccessFlags
#define gpuMemAccessFlagsProtNone        cudaMemAccessFlagsProtNone
#define gpuMemAccessFlagsProtRead        cudaMemAccessFlagsProtRead
#define gpuMemAccessFlagsProtReadWrite   cudaMemAccessFlagsProtReadWrite
#define gpuMemAddressFree                cuMemAddressFree
#define gpuMemAddressReserve             cuMemAddressReserve
#define gpuMemAdvise                     cudaMemAdvise
#define gpuMemAdviseSetAccessedBy        cudaMemAdviseSetAccessedBy
#define gpuMemAdviseSetPreferredLocation cudaMemAdviseSetPreferredLocation
#define gpuMemAdviseSetReadMostly        cudaMemAdviseSetReadMostly
#define gpuMemAdviseUnsetAccessedBy      cudaMemAdviseUnsetAccessedBy
#define gpuMemAdviseUnsetPreferredLocation  \
        cudaMemAdviseUnsetPreferredLocation
#define gpuMemAdviseUnsetReadMostly      cudaMemAdviseUnsetReadMostly
#define gpuMemAllocHost                  cuMemAllocHost_v2
#define gpuMemAllocNodeParams            cudaMemAllocNodeParams
#define gpuMemAllocPitch                 cuMemAllocPitch_v2
#define gpuMemAllocationGranularityMinimum  \
        CU_MEM_ALLOC_GRANULARITY_MINIMUM
#define gpuMemAllocationGranularityRecommended  \
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#define gpuMemAllocationGranularity_flags  \
        CUmemAllocationGranularity_flags
#define gpuMemAllocationHandleType       cudaMemAllocationHandleType
#define gpuMemAllocationProp             CUmemAllocationProp_v1
#define gpuMemAllocationType             cudaMemAllocationType
#define gpuMemAllocationTypeInvalid      cudaMemAllocationTypeInvalid
#define gpuMemAllocationTypeMax          cudaMemAllocationTypeMax
#define gpuMemAllocationTypePinned       cudaMemAllocationTypePinned
#define gpuMemAttachGlobal               cudaMemAttachGlobal
#define gpuMemAttachHost                 cudaMemAttachHost
#define gpuMemAttachSingle               cudaMemAttachSingle
#define gpuMemCreate                     cuMemCreate
#define gpuMemExportToShareableHandle    cuMemExportToShareableHandle
#define gpuMemGenericAllocationHandle_t  CUmemGenericAllocationHandle_v1
#define gpuMemGetAccess                  cuMemGetAccess
#define gpuMemGetAddressRange            cuMemGetAddressRange_v2
#define gpuMemGetAllocationGranularity   cuMemGetAllocationGranularity
#define gpuMemGetAllocationPropertiesFromHandle  \
        cuMemGetAllocationPropertiesFromHandle
#define gpuMemGetInfo                    cudaMemGetInfo
#define gpuMemHandleType                 CUmemHandleType
#define gpuMemHandleTypeGeneric          CU_MEM_HANDLE_TYPE_GENERIC
#define gpuMemHandleTypeNone             cudaMemHandleTypeNone
#define gpuMemHandleTypePosixFileDescriptor  \
        cudaMemHandleTypePosixFileDescriptor
#define gpuMemHandleTypeWin32            cudaMemHandleTypeWin32
#define gpuMemHandleTypeWin32Kmt         cudaMemHandleTypeWin32Kmt
#define gpuMemImportFromShareableHandle  cuMemImportFromShareableHandle
#define gpuMemLocation                   cudaMemLocation
#define gpuMemLocationType               cudaMemLocationType
#define gpuMemLocationTypeDevice         cudaMemLocationTypeDevice
#define gpuMemLocationTypeInvalid        cudaMemLocationTypeInvalid
#define gpuMemMap                        cuMemMap
#define gpuMemMapArrayAsync              cuMemMapArrayAsync
#define gpuMemOperationType              CUmemOperationType
#define gpuMemOperationTypeMap           CU_MEM_OPERATION_TYPE_MAP
#define gpuMemOperationTypeUnmap         CU_MEM_OPERATION_TYPE_UNMAP
#define gpuMemPoolAttr                   cudaMemPoolAttr
#define gpuMemPoolAttrReleaseThreshold   cudaMemPoolAttrReleaseThreshold
#define gpuMemPoolAttrReservedMemCurrent cudaMemPoolAttrReservedMemCurrent
#define gpuMemPoolAttrReservedMemHigh    cudaMemPoolAttrReservedMemHigh
#define gpuMemPoolAttrUsedMemCurrent     cudaMemPoolAttrUsedMemCurrent
#define gpuMemPoolAttrUsedMemHigh        cudaMemPoolAttrUsedMemHigh
#define gpuMemPoolCreate                 cudaMemPoolCreate
#define gpuMemPoolDestroy                cudaMemPoolDestroy
#define gpuMemPoolExportPointer          cudaMemPoolExportPointer
#define gpuMemPoolExportToShareableHandle  \
        cudaMemPoolExportToShareableHandle
#define gpuMemPoolGetAccess              cudaMemPoolGetAccess
#define gpuMemPoolGetAttribute           cudaMemPoolGetAttribute
#define gpuMemPoolImportFromShareableHandle  \
        cudaMemPoolImportFromShareableHandle
#define gpuMemPoolImportPointer          cudaMemPoolImportPointer
#define gpuMemPoolProps                  cudaMemPoolProps
#define gpuMemPoolPtrExportData          cudaMemPoolPtrExportData
#define gpuMemPoolReuseAllowInternalDependencies  \
        cudaMemPoolReuseAllowInternalDependencies
#define gpuMemPoolReuseAllowOpportunistic  \
        cudaMemPoolReuseAllowOpportunistic
#define gpuMemPoolReuseFollowEventDependencies  \
        cudaMemPoolReuseFollowEventDependencies
#define gpuMemPoolSetAccess              cudaMemPoolSetAccess
#define gpuMemPoolSetAttribute           cudaMemPoolSetAttribute
#define gpuMemPoolTrimTo                 cudaMemPoolTrimTo
#define gpuMemPool_t                     cudaMemPool_t
#define gpuMemPrefetchAsync              cudaMemPrefetchAsync
#define gpuMemRangeAttribute             cudaMemRangeAttribute
#define gpuMemRangeAttributeAccessedBy   cudaMemRangeAttributeAccessedBy
#define gpuMemRangeAttributeLastPrefetchLocation  \
        cudaMemRangeAttributeLastPrefetchLocation
#define gpuMemRangeAttributePreferredLocation  \
        cudaMemRangeAttributePreferredLocation
#define gpuMemRangeAttributeReadMostly   cudaMemRangeAttributeReadMostly
#define gpuMemRangeGetAttribute          cudaMemRangeGetAttribute
#define gpuMemRangeGetAttributes         cudaMemRangeGetAttributes
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
#define gpuMemoryAdvise                  cudaMemoryAdvise
#define gpuMemoryType                    cudaMemoryType
#define gpuMemoryTypeArray               CU_MEMORYTYPE_ARRAY
#define gpuMemoryTypeDevice              cudaMemoryTypeDevice
#define gpuMemoryTypeHost                cudaMemoryTypeHost
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
#define gpuMemsetParams                  cudaMemsetParams
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
#define gpuOccupancyDefault              cudaOccupancyDefault
#define gpuOccupancyDisableCachingOverride  \
        cudaOccupancyDisableCachingOverride
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
#define gpuSharedMemBankSizeDefault      cudaSharedMemBankSizeDefault
#define gpuSharedMemBankSizeEightByte    cudaSharedMemBankSizeEightByte
#define gpuSharedMemBankSizeFourByte     cudaSharedMemBankSizeFourByte
#define gpuSharedMemConfig               cudaSharedMemConfig
#define gpuSignalExternalSemaphoresAsync cudaSignalExternalSemaphoresAsync
#define gpuStreamAddCallback             cudaStreamAddCallback
#define gpuStreamAddCaptureDependencies  cudaStreamAddCaptureDependencies
#define gpuStreamAttachMemAsync          cudaStreamAttachMemAsync
#define gpuStreamBeginCapture            cudaStreamBeginCapture
#define gpuStreamCallback_t              cudaStreamCallback_t
#define gpuStreamCaptureMode             cudaStreamCaptureMode
#define gpuStreamCaptureModeGlobal       cudaStreamCaptureModeGlobal
#define gpuStreamCaptureModeRelaxed      cudaStreamCaptureModeRelaxed
#define gpuStreamCaptureModeThreadLocal  cudaStreamCaptureModeThreadLocal
#define gpuStreamCaptureStatus           cudaStreamCaptureStatus
#define gpuStreamCaptureStatusActive     cudaStreamCaptureStatusActive
#define gpuStreamCaptureStatusInvalidated  \
        cudaStreamCaptureStatusInvalidated
#define gpuStreamCaptureStatusNone       cudaStreamCaptureStatusNone
#define gpuStreamCreate                  cudaStreamCreate
#define gpuStreamCreateWithFlags         cudaStreamCreateWithFlags
#define gpuStreamCreateWithPriority      cudaStreamCreateWithPriority
#define gpuStreamDefault                 cudaStreamDefault
#define gpuStreamDestroy                 cudaStreamDestroy
#define gpuStreamEndCapture              cudaStreamEndCapture
#define gpuStreamGetCaptureInfo          cudaStreamGetCaptureInfo
#define gpuStreamGetCaptureInfo_v2       cuStreamGetCaptureInfo_v2
#define gpuStreamGetFlags                cudaStreamGetFlags
#define gpuStreamGetPriority             cudaStreamGetPriority
#define gpuStreamIsCapturing             cudaStreamIsCapturing
#define gpuStreamNonBlocking             cudaStreamNonBlocking
#define gpuStreamPerThread               cudaStreamPerThread
#define gpuStreamQuery                   cudaStreamQuery
#define gpuStreamSetCaptureDependencies  cudaStreamSetCaptureDependencies
#define gpuStreamSynchronize             cudaStreamSynchronize
#define gpuStreamUpdateCaptureDependencies  \
        cuStreamUpdateCaptureDependencies
#define gpuStreamUpdateCaptureDependenciesFlags  \
        cudaStreamUpdateCaptureDependenciesFlags
#define gpuStreamWaitEvent               cudaStreamWaitEvent
#define gpuStreamWaitValue32             cuStreamWaitValue32_v2
#define gpuStreamWaitValue64             cuStreamWaitValue64_v2
#define gpuStreamWaitValueAnd            CU_STREAM_WAIT_VALUE_AND
#define gpuStreamWaitValueEq             CU_STREAM_WAIT_VALUE_EQ
#define gpuStreamWaitValueGte            CU_STREAM_WAIT_VALUE_GEQ
#define gpuStreamWaitValueNor            CU_STREAM_WAIT_VALUE_NOR
#define gpuStreamWriteValue32            cuStreamWriteValue32_v2
#define gpuStreamWriteValue64            cuStreamWriteValue64_v2
#define gpuStream_t                      cudaStream_t
#define gpuSuccess                       cudaSuccess
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
        cudaThreadExchangeStreamCaptureMode
#define gpuUUID                          CUuuid
#define gpuUUID_t                        CUuuid_st
#define gpuUnbindTexture                 cudaUnbindTexture
#define gpuUserObjectCreate              cudaUserObjectCreate
#define gpuUserObjectFlags               cudaUserObjectFlags
#define gpuUserObjectNoDestructorSync    cudaUserObjectNoDestructorSync
#define gpuUserObjectRelease             cudaUserObjectRelease
#define gpuUserObjectRetain              cudaUserObjectRetain
#define gpuUserObjectRetainFlags         cudaUserObjectRetainFlags
#define gpuUserObject_t                  cudaUserObject_t
#define gpuWaitExternalSemaphoresAsync   cudaWaitExternalSemaphoresAsync

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
#define GPU_ARRAY3D_DESCRIPTOR           CUDA_ARRAY3D_DESCRIPTOR_v2
#define GPU_ARRAY_DESCRIPTOR             CUDA_ARRAY_DESCRIPTOR_v2
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
#define GPU_MEMCPY3D                     CUDA_MEMCPY3D_v2
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
#define gpuArray_t                       cudaArray_t
#define gpuChannelFormatDesc             cudaChannelFormatDesc
#define gpuChannelFormatKind             cudaChannelFormatKind
#define gpuChannelFormatKindFloat        cudaChannelFormatKindFloat
#define gpuChannelFormatKindNone         cudaChannelFormatKindNone
#define gpuChannelFormatKindSigned       cudaChannelFormatKindSigned
#define gpuChannelFormatKindUnsigned     cudaChannelFormatKindUnsigned
#define gpuDeviceptr_t                   CUdeviceptr_v2
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
#define gpuMipmappedArray_t              cudaMipmappedArray_t
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
#define gpu_Memcpy2D                     CUDA_MEMCPY2D_v2
#define make_gpuExtent                   make_cudaExtent
#define make_gpuPitchedPtr               make_cudaPitchedPtr
#define make_gpuPos                      make_cudaPos

/* surface_types.h */
#define gpuBoundaryModeClamp             cudaBoundaryModeClamp
#define gpuBoundaryModeTrap              cudaBoundaryModeTrap
#define gpuBoundaryModeZero              cudaBoundaryModeZero
#define gpuSurfaceBoundaryMode           cudaSurfaceBoundaryMode
#define gpuSurfaceObject_t               cudaSurfaceObject_t

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
#define gpuTextureObject_t               cudaTextureObject_t
#define gpuTextureReadMode               cudaTextureReadMode
#define gpuTextureType1D                 cudaTextureType1D
#define gpuTextureType1DLayered          cudaTextureType1DLayered
#define gpuTextureType2D                 cudaTextureType2D
#define gpuTextureType2DLayered          cudaTextureType2DLayered
#define gpuTextureType3D                 cudaTextureType3D
#define gpuTextureTypeCubemap            cudaTextureTypeCubemap
#define gpuTextureTypeCubemapLayered     cudaTextureTypeCubemapLayered

#endif
