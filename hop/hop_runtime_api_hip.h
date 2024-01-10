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

#ifndef __HOP_HOP_RUNTIME_API_HIP_H__
#define __HOP_HOP_RUNTIME_API_HIP_H__

#include <hip/hip_runtime_api.h>

#define GPU_IPC_HANDLE_SIZE              HIP_IPC_HANDLE_SIZE
#define GPU_LAUNCH_PARAM_BUFFER_POINTER  HIP_LAUNCH_PARAM_BUFFER_POINTER
#define GPU_LAUNCH_PARAM_BUFFER_SIZE     HIP_LAUNCH_PARAM_BUFFER_SIZE
#define GPU_LAUNCH_PARAM_END             HIP_LAUNCH_PARAM_END
#define gpuAccessPolicyWindow            hipAccessPolicyWindow
#define gpuAccessProperty                hipAccessProperty
#define gpuAccessPropertyNormal          hipAccessPropertyNormal
#define gpuAccessPropertyPersisting      hipAccessPropertyPersisting
#define gpuAccessPropertyStreaming       hipAccessPropertyStreaming
#define gpuArray3DCreate                 hipArray3DCreate
#define gpuArray3DGetDescriptor          hipArray3DGetDescriptor
#define gpuArrayCreate                   hipArrayCreate
#define gpuArrayCubemap                  hipArrayCubemap
#define gpuArrayDefault                  hipArrayDefault
#define gpuArrayDestroy                  hipArrayDestroy
#define gpuArrayGetDescriptor            hipArrayGetDescriptor
#define gpuArrayGetInfo                  hipArrayGetInfo
#define gpuArrayLayered                  hipArrayLayered
#define gpuArrayMapInfo                  hipArrayMapInfo
#define gpuArraySparseSubresourceType    hipArraySparseSubresourceType
#define gpuArraySparseSubresourceTypeMiptail  \
        hipArraySparseSubresourceTypeMiptail
#define gpuArraySparseSubresourceTypeSparseLevel  \
        hipArraySparseSubresourceTypeSparseLevel
#define gpuArraySurfaceLoadStore         hipArraySurfaceLoadStore
#define gpuArrayTextureGather            hipArrayTextureGather
#define gpuBindTexture                   hipBindTexture
#define gpuBindTexture2D                 hipBindTexture2D
#define gpuBindTextureToArray            hipBindTextureToArray
#define gpuBindTextureToMipmappedArray   hipBindTextureToMipmappedArray
#define gpuChooseDevice                  hipChooseDevice
#define gpuComputeMode                   hipComputeMode
#define gpuComputeModeDefault            hipComputeModeDefault
#define gpuComputeModeExclusive          hipComputeModeExclusive
#define gpuComputeModeExclusiveProcess   hipComputeModeExclusiveProcess
#define gpuComputeModeProhibited         hipComputeModeProhibited
#define gpuCooperativeLaunchMultiDeviceNoPostSync  \
        hipCooperativeLaunchMultiDeviceNoPostSync
#define gpuCooperativeLaunchMultiDeviceNoPreSync  \
        hipCooperativeLaunchMultiDeviceNoPreSync
#define gpuCpuDeviceId                   hipCpuDeviceId
#define gpuCreateSurfaceObject           hipCreateSurfaceObject
#define gpuCreateTextureObject           hipCreateTextureObject
#define gpuCtxCreate                     hipCtxCreate
#define gpuCtxDestroy                    hipCtxDestroy
#define gpuCtxDisablePeerAccess          hipCtxDisablePeerAccess
#define gpuCtxEnablePeerAccess           hipCtxEnablePeerAccess
#define gpuCtxGetApiVersion              hipCtxGetApiVersion
#define gpuCtxGetCacheConfig             hipCtxGetCacheConfig
#define gpuCtxGetCurrent                 hipCtxGetCurrent
#define gpuCtxGetDevice                  hipCtxGetDevice
#define gpuCtxGetFlags                   hipCtxGetFlags
#define gpuCtxGetSharedMemConfig         hipCtxGetSharedMemConfig
#define gpuCtxPopCurrent                 hipCtxPopCurrent
#define gpuCtxPushCurrent                hipCtxPushCurrent
#define gpuCtxSetCacheConfig             hipCtxSetCacheConfig
#define gpuCtxSetCurrent                 hipCtxSetCurrent
#define gpuCtxSetSharedMemConfig         hipCtxSetSharedMemConfig
#define gpuCtxSynchronize                hipCtxSynchronize
#define gpuCtx_t                         hipCtx_t
#define gpuDestroyExternalMemory         hipDestroyExternalMemory
#define gpuDestroyExternalSemaphore      hipDestroyExternalSemaphore
#define gpuDestroySurfaceObject          hipDestroySurfaceObject
#define gpuDestroyTextureObject          hipDestroyTextureObject
#define gpuDevP2PAttrAccessSupported     hipDevP2PAttrAccessSupported
#define gpuDevP2PAttrHipArrayAccessSupported  \
        hipDevP2PAttrHipArrayAccessSupported
#define gpuDevP2PAttrNativeAtomicSupported  \
        hipDevP2PAttrNativeAtomicSupported
#define gpuDevP2PAttrPerformanceRank     hipDevP2PAttrPerformanceRank
#define gpuDeviceAttributeAsyncEngineCount  \
        hipDeviceAttributeAsyncEngineCount
#define gpuDeviceAttributeCanMapHostMemory  \
        hipDeviceAttributeCanMapHostMemory
#define gpuDeviceAttributeCanUseHostPointerForRegisteredMem  \
        hipDeviceAttributeCanUseHostPointerForRegisteredMem
#define gpuDeviceAttributeCanUseStreamWaitValue  \
        hipDeviceAttributeCanUseStreamWaitValue
#define gpuDeviceAttributeClockRate      hipDeviceAttributeClockRate
#define gpuDeviceAttributeComputeCapabilityMajor  \
        hipDeviceAttributeComputeCapabilityMajor
#define gpuDeviceAttributeComputeCapabilityMinor  \
        hipDeviceAttributeComputeCapabilityMinor
#define gpuDeviceAttributeComputeMode    hipDeviceAttributeComputeMode
#define gpuDeviceAttributeComputePreemptionSupported  \
        hipDeviceAttributeComputePreemptionSupported
#define gpuDeviceAttributeConcurrentKernels  \
        hipDeviceAttributeConcurrentKernels
#define gpuDeviceAttributeConcurrentManagedAccess  \
        hipDeviceAttributeConcurrentManagedAccess
#define gpuDeviceAttributeCooperativeLaunch  \
        hipDeviceAttributeCooperativeLaunch
#define gpuDeviceAttributeCooperativeMultiDeviceLaunch  \
        hipDeviceAttributeCooperativeMultiDeviceLaunch
#define gpuDeviceAttributeDirectManagedMemAccessFromHost  \
        hipDeviceAttributeDirectManagedMemAccessFromHost
#define gpuDeviceAttributeEccEnabled     hipDeviceAttributeEccEnabled
#define gpuDeviceAttributeGlobalL1CacheSupported  \
        hipDeviceAttributeGlobalL1CacheSupported
#define gpuDeviceAttributeHostNativeAtomicSupported  \
        hipDeviceAttributeHostNativeAtomicSupported
#define gpuDeviceAttributeIntegrated     hipDeviceAttributeIntegrated
#define gpuDeviceAttributeIsMultiGpuBoard  \
        hipDeviceAttributeIsMultiGpuBoard
#define gpuDeviceAttributeKernelExecTimeout  \
        hipDeviceAttributeKernelExecTimeout
#define gpuDeviceAttributeL2CacheSize    hipDeviceAttributeL2CacheSize
#define gpuDeviceAttributeLocalL1CacheSupported  \
        hipDeviceAttributeLocalL1CacheSupported
#define gpuDeviceAttributeManagedMemory  hipDeviceAttributeManagedMemory
#define gpuDeviceAttributeMaxBlockDimX   hipDeviceAttributeMaxBlockDimX
#define gpuDeviceAttributeMaxBlockDimY   hipDeviceAttributeMaxBlockDimY
#define gpuDeviceAttributeMaxBlockDimZ   hipDeviceAttributeMaxBlockDimZ
#define gpuDeviceAttributeMaxBlocksPerMultiProcessor  \
        hipDeviceAttributeMaxBlocksPerMultiProcessor
#define gpuDeviceAttributeMaxGridDimX    hipDeviceAttributeMaxGridDimX
#define gpuDeviceAttributeMaxGridDimY    hipDeviceAttributeMaxGridDimY
#define gpuDeviceAttributeMaxGridDimZ    hipDeviceAttributeMaxGridDimZ
#define gpuDeviceAttributeMaxPitch       hipDeviceAttributeMaxPitch
#define gpuDeviceAttributeMaxRegistersPerBlock  \
        hipDeviceAttributeMaxRegistersPerBlock
#define gpuDeviceAttributeMaxRegistersPerMultiprocessor  \
        hipDeviceAttributeMaxRegistersPerMultiprocessor
#define gpuDeviceAttributeMaxSharedMemoryPerBlock  \
        hipDeviceAttributeMaxSharedMemoryPerBlock
#define gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor  \
        hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define gpuDeviceAttributeMaxSurface1D   hipDeviceAttributeMaxSurface1D
#define gpuDeviceAttributeMaxSurface1DLayered  \
        hipDeviceAttributeMaxSurface1DLayered
#define gpuDeviceAttributeMaxSurface2D   hipDeviceAttributeMaxSurface2D
#define gpuDeviceAttributeMaxSurface2DLayered  \
        hipDeviceAttributeMaxSurface2DLayered
#define gpuDeviceAttributeMaxSurface3D   hipDeviceAttributeMaxSurface3D
#define gpuDeviceAttributeMaxSurfaceCubemap  \
        hipDeviceAttributeMaxSurfaceCubemap
#define gpuDeviceAttributeMaxSurfaceCubemapLayered  \
        hipDeviceAttributeMaxSurfaceCubemapLayered
#define gpuDeviceAttributeMaxTexture1DLayered  \
        hipDeviceAttributeMaxTexture1DLayered
#define gpuDeviceAttributeMaxTexture1DLinear  \
        hipDeviceAttributeMaxTexture1DLinear
#define gpuDeviceAttributeMaxTexture1DMipmap  \
        hipDeviceAttributeMaxTexture1DMipmap
#define gpuDeviceAttributeMaxTexture1DWidth  \
        hipDeviceAttributeMaxTexture1DWidth
#define gpuDeviceAttributeMaxTexture2DGather  \
        hipDeviceAttributeMaxTexture2DGather
#define gpuDeviceAttributeMaxTexture2DHeight  \
        hipDeviceAttributeMaxTexture2DHeight
#define gpuDeviceAttributeMaxTexture2DLayered  \
        hipDeviceAttributeMaxTexture2DLayered
#define gpuDeviceAttributeMaxTexture2DLinear  \
        hipDeviceAttributeMaxTexture2DLinear
#define gpuDeviceAttributeMaxTexture2DMipmap  \
        hipDeviceAttributeMaxTexture2DMipmap
#define gpuDeviceAttributeMaxTexture2DWidth  \
        hipDeviceAttributeMaxTexture2DWidth
#define gpuDeviceAttributeMaxTexture3DAlt  \
        hipDeviceAttributeMaxTexture3DAlt
#define gpuDeviceAttributeMaxTexture3DDepth  \
        hipDeviceAttributeMaxTexture3DDepth
#define gpuDeviceAttributeMaxTexture3DHeight  \
        hipDeviceAttributeMaxTexture3DHeight
#define gpuDeviceAttributeMaxTexture3DWidth  \
        hipDeviceAttributeMaxTexture3DWidth
#define gpuDeviceAttributeMaxTextureCubemap  \
        hipDeviceAttributeMaxTextureCubemap
#define gpuDeviceAttributeMaxTextureCubemapLayered  \
        hipDeviceAttributeMaxTextureCubemapLayered
#define gpuDeviceAttributeMaxThreadsPerBlock  \
        hipDeviceAttributeMaxThreadsPerBlock
#define gpuDeviceAttributeMaxThreadsPerMultiProcessor  \
        hipDeviceAttributeMaxThreadsPerMultiProcessor
#define gpuDeviceAttributeMemoryBusWidth hipDeviceAttributeMemoryBusWidth
#define gpuDeviceAttributeMemoryClockRate  \
        hipDeviceAttributeMemoryClockRate
#define gpuDeviceAttributeMemoryPoolsSupported  \
        hipDeviceAttributeMemoryPoolsSupported
#define gpuDeviceAttributeMultiGpuBoardGroupID  \
        hipDeviceAttributeMultiGpuBoardGroupID
#define gpuDeviceAttributeMultiprocessorCount  \
        hipDeviceAttributeMultiprocessorCount
#define gpuDeviceAttributePageableMemoryAccess  \
        hipDeviceAttributePageableMemoryAccess
#define gpuDeviceAttributePageableMemoryAccessUsesHostPageTables  \
        hipDeviceAttributePageableMemoryAccessUsesHostPageTables
#define gpuDeviceAttributePciBusId       hipDeviceAttributePciBusId
#define gpuDeviceAttributePciDeviceId    hipDeviceAttributePciDeviceId
#define gpuDeviceAttributePciDomainID    hipDeviceAttributePciDomainID
#define gpuDeviceAttributeSharedMemPerBlockOptin  \
        hipDeviceAttributeSharedMemPerBlockOptin
#define gpuDeviceAttributeSingleToDoublePrecisionPerfRatio  \
        hipDeviceAttributeSingleToDoublePrecisionPerfRatio
#define gpuDeviceAttributeStreamPrioritiesSupported  \
        hipDeviceAttributeStreamPrioritiesSupported
#define gpuDeviceAttributeSurfaceAlignment  \
        hipDeviceAttributeSurfaceAlignment
#define gpuDeviceAttributeTccDriver      hipDeviceAttributeTccDriver
#define gpuDeviceAttributeTextureAlignment  \
        hipDeviceAttributeTextureAlignment
#define gpuDeviceAttributeTexturePitchAlignment  \
        hipDeviceAttributeTexturePitchAlignment
#define gpuDeviceAttributeTotalConstantMemory  \
        hipDeviceAttributeTotalConstantMemory
#define gpuDeviceAttributeUnifiedAddressing  \
        hipDeviceAttributeUnifiedAddressing
#define gpuDeviceAttributeVirtualMemoryManagementSupported  \
        hipDeviceAttributeVirtualMemoryManagementSupported
#define gpuDeviceAttributeWarpSize       hipDeviceAttributeWarpSize
#define gpuDeviceAttribute_t             hipDeviceAttribute_t
#define gpuDeviceCanAccessPeer           hipDeviceCanAccessPeer
#define gpuDeviceComputeCapability       hipDeviceComputeCapability
#define gpuDeviceDisablePeerAccess       hipDeviceDisablePeerAccess
#define gpuDeviceEnablePeerAccess        hipDeviceEnablePeerAccess
#define gpuDeviceGet                     hipDeviceGet
#define gpuDeviceGetAttribute            hipDeviceGetAttribute
#define gpuDeviceGetByPCIBusId           hipDeviceGetByPCIBusId
#define gpuDeviceGetCacheConfig          hipDeviceGetCacheConfig
#define gpuDeviceGetDefaultMemPool       hipDeviceGetDefaultMemPool
#define gpuDeviceGetGraphMemAttribute    hipDeviceGetGraphMemAttribute
#define gpuDeviceGetLimit                hipDeviceGetLimit
#define gpuDeviceGetMemPool              hipDeviceGetMemPool
#define gpuDeviceGetName                 hipDeviceGetName
#define gpuDeviceGetP2PAttribute         hipDeviceGetP2PAttribute
#define gpuDeviceGetPCIBusId             hipDeviceGetPCIBusId
#define gpuDeviceGetSharedMemConfig      hipDeviceGetSharedMemConfig
#define gpuDeviceGetStreamPriorityRange  hipDeviceGetStreamPriorityRange
#define gpuDeviceGetUuid                 hipDeviceGetUuid
#define gpuDeviceGraphMemTrim            hipDeviceGraphMemTrim
#define gpuDeviceLmemResizeToMax         hipDeviceLmemResizeToMax
#define gpuDeviceMapHost                 hipDeviceMapHost
#define gpuDeviceP2PAttr                 hipDeviceP2PAttr
#define gpuDevicePrimaryCtxGetState      hipDevicePrimaryCtxGetState
#define gpuDevicePrimaryCtxRelease       hipDevicePrimaryCtxRelease
#define gpuDevicePrimaryCtxReset         hipDevicePrimaryCtxReset
#define gpuDevicePrimaryCtxRetain        hipDevicePrimaryCtxRetain
#define gpuDevicePrimaryCtxSetFlags      hipDevicePrimaryCtxSetFlags
#define gpuDeviceProp_t                  hipDeviceProp_t
#define gpuDeviceReset                   hipDeviceReset
#define gpuDeviceScheduleAuto            hipDeviceScheduleAuto
#define gpuDeviceScheduleBlockingSync    hipDeviceScheduleBlockingSync
#define gpuDeviceScheduleMask            hipDeviceScheduleMask
#define gpuDeviceScheduleSpin            hipDeviceScheduleSpin
#define gpuDeviceScheduleYield           hipDeviceScheduleYield
#define gpuDeviceSetCacheConfig          hipDeviceSetCacheConfig
#define gpuDeviceSetGraphMemAttribute    hipDeviceSetGraphMemAttribute
#define gpuDeviceSetLimit                hipDeviceSetLimit
#define gpuDeviceSetMemPool              hipDeviceSetMemPool
#define gpuDeviceSetSharedMemConfig      hipDeviceSetSharedMemConfig
#define gpuDeviceSynchronize             hipDeviceSynchronize
#define gpuDeviceTotalMem                hipDeviceTotalMem
#define gpuDevice_t                      hipDevice_t
#define gpuDriverGetVersion              hipDriverGetVersion
#define gpuDrvGetErrorName               hipDrvGetErrorName
#define gpuDrvGetErrorString             hipDrvGetErrorString
#define gpuDrvMemcpy2DUnaligned          hipDrvMemcpy2DUnaligned
#define gpuDrvMemcpy3D                   hipDrvMemcpy3D
#define gpuDrvMemcpy3DAsync              hipDrvMemcpy3DAsync
#define gpuDrvPointerGetAttributes       hipDrvPointerGetAttributes
#define gpuErrorAlreadyAcquired          hipErrorAlreadyAcquired
#define gpuErrorAlreadyMapped            hipErrorAlreadyMapped
#define gpuErrorArrayIsMapped            hipErrorArrayIsMapped
#define gpuErrorAssert                   hipErrorAssert
#define gpuErrorCapturedEvent            hipErrorCapturedEvent
#define gpuErrorContextAlreadyCurrent    hipErrorContextAlreadyCurrent
#define gpuErrorContextAlreadyInUse      hipErrorContextAlreadyInUse
#define gpuErrorContextIsDestroyed       hipErrorContextIsDestroyed
#define gpuErrorCooperativeLaunchTooLarge  \
        hipErrorCooperativeLaunchTooLarge
#define gpuErrorDeinitialized            hipErrorDeinitialized
#define gpuErrorECCNotCorrectable        hipErrorECCNotCorrectable
#define gpuErrorFileNotFound             hipErrorFileNotFound
#define gpuErrorGraphExecUpdateFailure   hipErrorGraphExecUpdateFailure
#define gpuErrorHostMemoryAlreadyRegistered  \
        hipErrorHostMemoryAlreadyRegistered
#define gpuErrorHostMemoryNotRegistered  hipErrorHostMemoryNotRegistered
#define gpuErrorIllegalAddress           hipErrorIllegalAddress
#define gpuErrorIllegalState             hipErrorIllegalState
#define gpuErrorInsufficientDriver       hipErrorInsufficientDriver
#define gpuErrorInvalidConfiguration     hipErrorInvalidConfiguration
#define gpuErrorInvalidContext           hipErrorInvalidContext
#define gpuErrorInvalidDevice            hipErrorInvalidDevice
#define gpuErrorInvalidDeviceFunction    hipErrorInvalidDeviceFunction
#define gpuErrorInvalidDevicePointer     hipErrorInvalidDevicePointer
#define gpuErrorInvalidGraphicsContext   hipErrorInvalidGraphicsContext
#define gpuErrorInvalidHandle            hipErrorInvalidHandle
#define gpuErrorInvalidImage             hipErrorInvalidImage
#define gpuErrorInvalidKernelFile        hipErrorInvalidKernelFile
#define gpuErrorInvalidMemcpyDirection   hipErrorInvalidMemcpyDirection
#define gpuErrorInvalidPitchValue        hipErrorInvalidPitchValue
#define gpuErrorInvalidSource            hipErrorInvalidSource
#define gpuErrorInvalidSymbol            hipErrorInvalidSymbol
#define gpuErrorInvalidValue             hipErrorInvalidValue
#define gpuErrorLaunchFailure            hipErrorLaunchFailure
#define gpuErrorLaunchOutOfResources     hipErrorLaunchOutOfResources
#define gpuErrorLaunchTimeOut            hipErrorLaunchTimeOut
#define gpuErrorMapFailed                hipErrorMapFailed
#define gpuErrorMissingConfiguration     hipErrorMissingConfiguration
#define gpuErrorNoBinaryForGpu           hipErrorNoBinaryForGpu
#define gpuErrorNoDevice                 hipErrorNoDevice
#define gpuErrorNotFound                 hipErrorNotFound
#define gpuErrorNotInitialized           hipErrorNotInitialized
#define gpuErrorNotMapped                hipErrorNotMapped
#define gpuErrorNotMappedAsArray         hipErrorNotMappedAsArray
#define gpuErrorNotMappedAsPointer       hipErrorNotMappedAsPointer
#define gpuErrorNotReady                 hipErrorNotReady
#define gpuErrorNotSupported             hipErrorNotSupported
#define gpuErrorOperatingSystem          hipErrorOperatingSystem
#define gpuErrorOutOfMemory              hipErrorOutOfMemory
#define gpuErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define gpuErrorPeerAccessNotEnabled     hipErrorPeerAccessNotEnabled
#define gpuErrorPeerAccessUnsupported    hipErrorPeerAccessUnsupported
#define gpuErrorPriorLaunchFailure       hipErrorPriorLaunchFailure
#define gpuErrorProfilerAlreadyStarted   hipErrorProfilerAlreadyStarted
#define gpuErrorProfilerAlreadyStopped   hipErrorProfilerAlreadyStopped
#define gpuErrorProfilerDisabled         hipErrorProfilerDisabled
#define gpuErrorProfilerNotInitialized   hipErrorProfilerNotInitialized
#define gpuErrorSetOnActiveProcess       hipErrorSetOnActiveProcess
#define gpuErrorSharedObjectInitFailed   hipErrorSharedObjectInitFailed
#define gpuErrorSharedObjectSymbolNotFound  \
        hipErrorSharedObjectSymbolNotFound
#define gpuErrorStreamCaptureImplicit    hipErrorStreamCaptureImplicit
#define gpuErrorStreamCaptureInvalidated hipErrorStreamCaptureInvalidated
#define gpuErrorStreamCaptureIsolation   hipErrorStreamCaptureIsolation
#define gpuErrorStreamCaptureMerge       hipErrorStreamCaptureMerge
#define gpuErrorStreamCaptureUnjoined    hipErrorStreamCaptureUnjoined
#define gpuErrorStreamCaptureUnmatched   hipErrorStreamCaptureUnmatched
#define gpuErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define gpuErrorStreamCaptureWrongThread hipErrorStreamCaptureWrongThread
#define gpuErrorUnknown                  hipErrorUnknown
#define gpuErrorUnmapFailed              hipErrorUnmapFailed
#define gpuErrorUnsupportedLimit         hipErrorUnsupportedLimit
#define gpuError_t                       hipError_t
#define gpuEventBlockingSync             hipEventBlockingSync
#define gpuEventCreate                   hipEventCreate
#define gpuEventCreateWithFlags          hipEventCreateWithFlags
#define gpuEventDefault                  hipEventDefault
#define gpuEventDestroy                  hipEventDestroy
#define gpuEventDisableTiming            hipEventDisableTiming
#define gpuEventElapsedTime              hipEventElapsedTime
#define gpuEventInterprocess             hipEventInterprocess
#define gpuEventQuery                    hipEventQuery
#define gpuEventRecord                   hipEventRecord
#define gpuEventSynchronize              hipEventSynchronize
#define gpuEvent_t                       hipEvent_t
#define gpuExternalMemoryBufferDesc      hipExternalMemoryBufferDesc
#define gpuExternalMemoryBufferDesc_st   hipExternalMemoryBufferDesc_st
#define gpuExternalMemoryDedicated       hipExternalMemoryDedicated
#define gpuExternalMemoryGetMappedBuffer hipExternalMemoryGetMappedBuffer
#define gpuExternalMemoryHandleDesc      hipExternalMemoryHandleDesc
#define gpuExternalMemoryHandleDesc_st   hipExternalMemoryHandleDesc_st
#define gpuExternalMemoryHandleType      hipExternalMemoryHandleType
#define gpuExternalMemoryHandleTypeD3D11Resource  \
        hipExternalMemoryHandleTypeD3D11Resource
#define gpuExternalMemoryHandleTypeD3D11ResourceKmt  \
        hipExternalMemoryHandleTypeD3D11ResourceKmt
#define gpuExternalMemoryHandleTypeD3D12Heap  \
        hipExternalMemoryHandleTypeD3D12Heap
#define gpuExternalMemoryHandleTypeD3D12Resource  \
        hipExternalMemoryHandleTypeD3D12Resource
#define gpuExternalMemoryHandleTypeOpaqueFd  \
        hipExternalMemoryHandleTypeOpaqueFd
#define gpuExternalMemoryHandleTypeOpaqueWin32  \
        hipExternalMemoryHandleTypeOpaqueWin32
#define gpuExternalMemoryHandleTypeOpaqueWin32Kmt  \
        hipExternalMemoryHandleTypeOpaqueWin32Kmt
#define gpuExternalMemoryHandleType_enum hipExternalMemoryHandleType_enum
#define gpuExternalMemory_t              hipExternalMemory_t
#define gpuExternalSemaphoreHandleDesc   hipExternalSemaphoreHandleDesc
#define gpuExternalSemaphoreHandleDesc_st  \
        hipExternalSemaphoreHandleDesc_st
#define gpuExternalSemaphoreHandleType   hipExternalSemaphoreHandleType
#define gpuExternalSemaphoreHandleTypeD3D12Fence  \
        hipExternalSemaphoreHandleTypeD3D12Fence
#define gpuExternalSemaphoreHandleTypeOpaqueFd  \
        hipExternalSemaphoreHandleTypeOpaqueFd
#define gpuExternalSemaphoreHandleTypeOpaqueWin32  \
        hipExternalSemaphoreHandleTypeOpaqueWin32
#define gpuExternalSemaphoreHandleTypeOpaqueWin32Kmt  \
        hipExternalSemaphoreHandleTypeOpaqueWin32Kmt
#define gpuExternalSemaphoreHandleType_enum  \
        hipExternalSemaphoreHandleType_enum
#define gpuExternalSemaphoreSignalParams hipExternalSemaphoreSignalParams
#define gpuExternalSemaphoreSignalParams_st  \
        hipExternalSemaphoreSignalParams_st
#define gpuExternalSemaphoreWaitParams   hipExternalSemaphoreWaitParams
#define gpuExternalSemaphoreWaitParams_st  \
        hipExternalSemaphoreWaitParams_st
#define gpuExternalSemaphore_t           hipExternalSemaphore_t
#define gpuFree                          hipFree
#define gpuFreeArray                     hipFreeArray
#define gpuFreeAsync                     hipFreeAsync
#define gpuFreeMipmappedArray            hipFreeMipmappedArray
#define gpuFuncAttribute                 hipFuncAttribute
#define gpuFuncAttributeMax              hipFuncAttributeMax
#define gpuFuncAttributeMaxDynamicSharedMemorySize  \
        hipFuncAttributeMaxDynamicSharedMemorySize
#define gpuFuncAttributePreferredSharedMemoryCarveout  \
        hipFuncAttributePreferredSharedMemoryCarveout
#define gpuFuncAttributes                hipFuncAttributes
#define gpuFuncCachePreferEqual          hipFuncCachePreferEqual
#define gpuFuncCachePreferL1             hipFuncCachePreferL1
#define gpuFuncCachePreferNone           hipFuncCachePreferNone
#define gpuFuncCachePreferShared         hipFuncCachePreferShared
#define gpuFuncCache_t                   hipFuncCache_t
#define gpuFuncGetAttribute              hipFuncGetAttribute
#define gpuFuncGetAttributes             hipFuncGetAttributes
#define gpuFuncSetAttribute              hipFuncSetAttribute
#define gpuFuncSetCacheConfig            hipFuncSetCacheConfig
#define gpuFuncSetSharedMemConfig        hipFuncSetSharedMemConfig
#define gpuFunctionLaunchParams          hipFunctionLaunchParams
#define gpuFunctionLaunchParams_t        hipFunctionLaunchParams_t
#define gpuFunction_t                    hipFunction_t
#define gpuGLDeviceList                  hipGLDeviceList
#define gpuGLDeviceListAll               hipGLDeviceListAll
#define gpuGLDeviceListCurrentFrame      hipGLDeviceListCurrentFrame
#define gpuGLDeviceListNextFrame         hipGLDeviceListNextFrame
#define gpuGLGetDevices                  hipGLGetDevices
#define gpuGetChannelDesc                hipGetChannelDesc
#define gpuGetDevice                     hipGetDevice
#define gpuGetDeviceCount                hipGetDeviceCount
#define gpuGetDeviceFlags                hipGetDeviceFlags
#define gpuGetDeviceProperties           hipGetDeviceProperties
#define gpuGetErrorName                  hipGetErrorName
#define gpuGetErrorString                hipGetErrorString
#define gpuGetLastError                  hipGetLastError
#define gpuGetMipmappedArrayLevel        hipGetMipmappedArrayLevel
#define gpuGetSymbolAddress              hipGetSymbolAddress
#define gpuGetSymbolSize                 hipGetSymbolSize
#define gpuGetTextureAlignmentOffset     hipGetTextureAlignmentOffset
#define gpuGetTextureObjectResourceDesc  hipGetTextureObjectResourceDesc
#define gpuGetTextureObjectResourceViewDesc  \
        hipGetTextureObjectResourceViewDesc
#define gpuGetTextureObjectTextureDesc   hipGetTextureObjectTextureDesc
#define gpuGetTextureReference           hipGetTextureReference
#define gpuGraphAddChildGraphNode        hipGraphAddChildGraphNode
#define gpuGraphAddDependencies          hipGraphAddDependencies
#define gpuGraphAddEmptyNode             hipGraphAddEmptyNode
#define gpuGraphAddEventRecordNode       hipGraphAddEventRecordNode
#define gpuGraphAddEventWaitNode         hipGraphAddEventWaitNode
#define gpuGraphAddHostNode              hipGraphAddHostNode
#define gpuGraphAddKernelNode            hipGraphAddKernelNode
#define gpuGraphAddMemAllocNode          hipGraphAddMemAllocNode
#define gpuGraphAddMemFreeNode           hipGraphAddMemFreeNode
#define gpuGraphAddMemcpyNode            hipGraphAddMemcpyNode
#define gpuGraphAddMemcpyNode1D          hipGraphAddMemcpyNode1D
#define gpuGraphAddMemcpyNodeFromSymbol  hipGraphAddMemcpyNodeFromSymbol
#define gpuGraphAddMemcpyNodeToSymbol    hipGraphAddMemcpyNodeToSymbol
#define gpuGraphAddMemsetNode            hipGraphAddMemsetNode
#define gpuGraphChildGraphNodeGetGraph   hipGraphChildGraphNodeGetGraph
#define gpuGraphClone                    hipGraphClone
#define gpuGraphCreate                   hipGraphCreate
#define gpuGraphDebugDotFlags            hipGraphDebugDotFlags
#define gpuGraphDebugDotFlagsEventNodeParams  \
        hipGraphDebugDotFlagsEventNodeParams
#define gpuGraphDebugDotFlagsExtSemasSignalNodeParams  \
        hipGraphDebugDotFlagsExtSemasSignalNodeParams
#define gpuGraphDebugDotFlagsExtSemasWaitNodeParams  \
        hipGraphDebugDotFlagsExtSemasWaitNodeParams
#define gpuGraphDebugDotFlagsHandles     hipGraphDebugDotFlagsHandles
#define gpuGraphDebugDotFlagsHostNodeParams  \
        hipGraphDebugDotFlagsHostNodeParams
#define gpuGraphDebugDotFlagsKernelNodeAttributes  \
        hipGraphDebugDotFlagsKernelNodeAttributes
#define gpuGraphDebugDotFlagsKernelNodeParams  \
        hipGraphDebugDotFlagsKernelNodeParams
#define gpuGraphDebugDotFlagsMemcpyNodeParams  \
        hipGraphDebugDotFlagsMemcpyNodeParams
#define gpuGraphDebugDotFlagsMemsetNodeParams  \
        hipGraphDebugDotFlagsMemsetNodeParams
#define gpuGraphDebugDotFlagsVerbose     hipGraphDebugDotFlagsVerbose
#define gpuGraphDebugDotPrint            hipGraphDebugDotPrint
#define gpuGraphDestroy                  hipGraphDestroy
#define gpuGraphDestroyNode              hipGraphDestroyNode
#define gpuGraphEventRecordNodeGetEvent  hipGraphEventRecordNodeGetEvent
#define gpuGraphEventRecordNodeSetEvent  hipGraphEventRecordNodeSetEvent
#define gpuGraphEventWaitNodeGetEvent    hipGraphEventWaitNodeGetEvent
#define gpuGraphEventWaitNodeSetEvent    hipGraphEventWaitNodeSetEvent
#define gpuGraphExecChildGraphNodeSetParams  \
        hipGraphExecChildGraphNodeSetParams
#define gpuGraphExecDestroy              hipGraphExecDestroy
#define gpuGraphExecEventRecordNodeSetEvent  \
        hipGraphExecEventRecordNodeSetEvent
#define gpuGraphExecEventWaitNodeSetEvent  \
        hipGraphExecEventWaitNodeSetEvent
#define gpuGraphExecHostNodeSetParams    hipGraphExecHostNodeSetParams
#define gpuGraphExecKernelNodeSetParams  hipGraphExecKernelNodeSetParams
#define gpuGraphExecMemcpyNodeSetParams  hipGraphExecMemcpyNodeSetParams
#define gpuGraphExecMemcpyNodeSetParams1D  \
        hipGraphExecMemcpyNodeSetParams1D
#define gpuGraphExecMemcpyNodeSetParamsFromSymbol  \
        hipGraphExecMemcpyNodeSetParamsFromSymbol
#define gpuGraphExecMemcpyNodeSetParamsToSymbol  \
        hipGraphExecMemcpyNodeSetParamsToSymbol
#define gpuGraphExecMemsetNodeSetParams  hipGraphExecMemsetNodeSetParams
#define gpuGraphExecUpdate               hipGraphExecUpdate
#define gpuGraphExecUpdateError          hipGraphExecUpdateError
#define gpuGraphExecUpdateErrorFunctionChanged  \
        hipGraphExecUpdateErrorFunctionChanged
#define gpuGraphExecUpdateErrorNodeTypeChanged  \
        hipGraphExecUpdateErrorNodeTypeChanged
#define gpuGraphExecUpdateErrorNotSupported  \
        hipGraphExecUpdateErrorNotSupported
#define gpuGraphExecUpdateErrorParametersChanged  \
        hipGraphExecUpdateErrorParametersChanged
#define gpuGraphExecUpdateErrorTopologyChanged  \
        hipGraphExecUpdateErrorTopologyChanged
#define gpuGraphExecUpdateErrorUnsupportedFunctionChange  \
        hipGraphExecUpdateErrorUnsupportedFunctionChange
#define gpuGraphExecUpdateResult         hipGraphExecUpdateResult
#define gpuGraphExecUpdateSuccess        hipGraphExecUpdateSuccess
#define gpuGraphExec_t                   hipGraphExec_t
#define gpuGraphGetEdges                 hipGraphGetEdges
#define gpuGraphGetNodes                 hipGraphGetNodes
#define gpuGraphGetRootNodes             hipGraphGetRootNodes
#define gpuGraphHostNodeGetParams        hipGraphHostNodeGetParams
#define gpuGraphHostNodeSetParams        hipGraphHostNodeSetParams
#define gpuGraphInstantiate              hipGraphInstantiate
#define gpuGraphInstantiateFlagAutoFreeOnLaunch  \
        hipGraphInstantiateFlagAutoFreeOnLaunch
#define gpuGraphInstantiateFlagDeviceLaunch  \
        hipGraphInstantiateFlagDeviceLaunch
#define gpuGraphInstantiateFlagUpload    hipGraphInstantiateFlagUpload
#define gpuGraphInstantiateFlagUseNodePriority  \
        hipGraphInstantiateFlagUseNodePriority
#define gpuGraphInstantiateFlags         hipGraphInstantiateFlags
#define gpuGraphInstantiateWithFlags     hipGraphInstantiateWithFlags
#define gpuGraphKernelNodeCopyAttributes hipGraphKernelNodeCopyAttributes
#define gpuGraphKernelNodeGetAttribute   hipGraphKernelNodeGetAttribute
#define gpuGraphKernelNodeGetParams      hipGraphKernelNodeGetParams
#define gpuGraphKernelNodeSetAttribute   hipGraphKernelNodeSetAttribute
#define gpuGraphKernelNodeSetParams      hipGraphKernelNodeSetParams
#define gpuGraphLaunch                   hipGraphLaunch
#define gpuGraphMemAllocNodeGetParams    hipGraphMemAllocNodeGetParams
#define gpuGraphMemAttrReservedMemCurrent  \
        hipGraphMemAttrReservedMemCurrent
#define gpuGraphMemAttrReservedMemHigh   hipGraphMemAttrReservedMemHigh
#define gpuGraphMemAttrUsedMemCurrent    hipGraphMemAttrUsedMemCurrent
#define gpuGraphMemAttrUsedMemHigh       hipGraphMemAttrUsedMemHigh
#define gpuGraphMemAttributeType         hipGraphMemAttributeType
#define gpuGraphMemFreeNodeGetParams     hipGraphMemFreeNodeGetParams
#define gpuGraphMemcpyNodeGetParams      hipGraphMemcpyNodeGetParams
#define gpuGraphMemcpyNodeSetParams      hipGraphMemcpyNodeSetParams
#define gpuGraphMemcpyNodeSetParams1D    hipGraphMemcpyNodeSetParams1D
#define gpuGraphMemcpyNodeSetParamsFromSymbol  \
        hipGraphMemcpyNodeSetParamsFromSymbol
#define gpuGraphMemcpyNodeSetParamsToSymbol  \
        hipGraphMemcpyNodeSetParamsToSymbol
#define gpuGraphMemsetNodeGetParams      hipGraphMemsetNodeGetParams
#define gpuGraphMemsetNodeSetParams      hipGraphMemsetNodeSetParams
#define gpuGraphNodeFindInClone          hipGraphNodeFindInClone
#define gpuGraphNodeGetDependencies      hipGraphNodeGetDependencies
#define gpuGraphNodeGetDependentNodes    hipGraphNodeGetDependentNodes
#define gpuGraphNodeGetEnabled           hipGraphNodeGetEnabled
#define gpuGraphNodeGetType              hipGraphNodeGetType
#define gpuGraphNodeSetEnabled           hipGraphNodeSetEnabled
#define gpuGraphNodeType                 hipGraphNodeType
#define gpuGraphNodeTypeCount            hipGraphNodeTypeCount
#define gpuGraphNodeTypeEmpty            hipGraphNodeTypeEmpty
#define gpuGraphNodeTypeEventRecord      hipGraphNodeTypeEventRecord
#define gpuGraphNodeTypeExtSemaphoreSignal  \
        hipGraphNodeTypeExtSemaphoreSignal
#define gpuGraphNodeTypeExtSemaphoreWait hipGraphNodeTypeExtSemaphoreWait
#define gpuGraphNodeTypeGraph            hipGraphNodeTypeGraph
#define gpuGraphNodeTypeHost             hipGraphNodeTypeHost
#define gpuGraphNodeTypeKernel           hipGraphNodeTypeKernel
#define gpuGraphNodeTypeMemAlloc         hipGraphNodeTypeMemAlloc
#define gpuGraphNodeTypeMemFree          hipGraphNodeTypeMemFree
#define gpuGraphNodeTypeMemcpy           hipGraphNodeTypeMemcpy
#define gpuGraphNodeTypeMemset           hipGraphNodeTypeMemset
#define gpuGraphNodeTypeWaitEvent        hipGraphNodeTypeWaitEvent
#define gpuGraphNode_t                   hipGraphNode_t
#define gpuGraphReleaseUserObject        hipGraphReleaseUserObject
#define gpuGraphRemoveDependencies       hipGraphRemoveDependencies
#define gpuGraphRetainUserObject         hipGraphRetainUserObject
#define gpuGraphUpload                   hipGraphUpload
#define gpuGraphUserObjectMove           hipGraphUserObjectMove
#define gpuGraph_t                       hipGraph_t
#define gpuGraphicsGLRegisterBuffer      hipGraphicsGLRegisterBuffer
#define gpuGraphicsGLRegisterImage       hipGraphicsGLRegisterImage
#define gpuGraphicsMapResources          hipGraphicsMapResources
#define gpuGraphicsRegisterFlags         hipGraphicsRegisterFlags
#define gpuGraphicsRegisterFlagsNone     hipGraphicsRegisterFlagsNone
#define gpuGraphicsRegisterFlagsReadOnly hipGraphicsRegisterFlagsReadOnly
#define gpuGraphicsRegisterFlagsSurfaceLoadStore  \
        hipGraphicsRegisterFlagsSurfaceLoadStore
#define gpuGraphicsRegisterFlagsTextureGather  \
        hipGraphicsRegisterFlagsTextureGather
#define gpuGraphicsRegisterFlagsWriteDiscard  \
        hipGraphicsRegisterFlagsWriteDiscard
#define gpuGraphicsResource              hipGraphicsResource
#define gpuGraphicsResourceGetMappedPointer  \
        hipGraphicsResourceGetMappedPointer
#define gpuGraphicsResource_t            hipGraphicsResource_t
#define gpuGraphicsSubResourceGetMappedArray  \
        hipGraphicsSubResourceGetMappedArray
#define gpuGraphicsUnmapResources        hipGraphicsUnmapResources
#define gpuGraphicsUnregisterResource    hipGraphicsUnregisterResource
#define gpuHostAlloc                     hipHostAlloc
#define gpuHostFn_t                      hipHostFn_t
#define gpuHostFree                      hipHostFree
#define gpuHostGetDevicePointer          hipHostGetDevicePointer
#define gpuHostGetFlags                  hipHostGetFlags
#define gpuHostMalloc                    hipHostMalloc
#define gpuHostMallocDefault             hipHostMallocDefault
#define gpuHostMallocMapped              hipHostMallocMapped
#define gpuHostMallocPortable            hipHostMallocPortable
#define gpuHostMallocWriteCombined       hipHostMallocWriteCombined
#define gpuHostNodeParams                hipHostNodeParams
#define gpuHostRegister                  hipHostRegister
#define gpuHostRegisterDefault           hipHostRegisterDefault
#define gpuHostRegisterIoMemory          hipHostRegisterIoMemory
#define gpuHostRegisterMapped            hipHostRegisterMapped
#define gpuHostRegisterPortable          hipHostRegisterPortable
#define gpuHostRegisterReadOnly          hipHostRegisterReadOnly
#define gpuHostUnregister                hipHostUnregister
#define gpuImportExternalMemory          hipImportExternalMemory
#define gpuImportExternalSemaphore       hipImportExternalSemaphore
#define gpuInit                          hipInit
#define gpuInvalidDeviceId               hipInvalidDeviceId
#define gpuIpcCloseMemHandle             hipIpcCloseMemHandle
#define gpuIpcEventHandle_st             hipIpcEventHandle_st
#define gpuIpcEventHandle_t              hipIpcEventHandle_t
#define gpuIpcGetEventHandle             hipIpcGetEventHandle
#define gpuIpcGetMemHandle               hipIpcGetMemHandle
#define gpuIpcMemHandle_st               hipIpcMemHandle_st
#define gpuIpcMemHandle_t                hipIpcMemHandle_t
#define gpuIpcMemLazyEnablePeerAccess    hipIpcMemLazyEnablePeerAccess
#define gpuIpcOpenEventHandle            hipIpcOpenEventHandle
#define gpuIpcOpenMemHandle              hipIpcOpenMemHandle
#define gpuJitOption                     hipJitOption
#define gpuKernelNodeAttrID              hipKernelNodeAttrID
#define gpuKernelNodeAttrValue           hipKernelNodeAttrValue
#define gpuKernelNodeAttributeAccessPolicyWindow  \
        hipKernelNodeAttributeAccessPolicyWindow
#define gpuKernelNodeAttributeCooperative  \
        hipKernelNodeAttributeCooperative
#define gpuKernelNodeParams              hipKernelNodeParams
#define gpuLaunchCooperativeKernel       hipLaunchCooperativeKernel
#define gpuLaunchCooperativeKernelMultiDevice  \
        hipLaunchCooperativeKernelMultiDevice
#define gpuLaunchHostFunc                hipLaunchHostFunc
#define gpuLaunchKernel                  hipLaunchKernel
#define gpuLaunchParams                  hipLaunchParams
#define gpuLimitMallocHeapSize           hipLimitMallocHeapSize
#define gpuLimitPrintfFifoSize           hipLimitPrintfFifoSize
#define gpuLimitStackSize                hipLimitStackSize
#define gpuLimit_t                       hipLimit_t
#define gpuMalloc                        hipMalloc
#define gpuMalloc3D                      hipMalloc3D
#define gpuMalloc3DArray                 hipMalloc3DArray
#define gpuMallocArray                   hipMallocArray
#define gpuMallocAsync                   hipMallocAsync
#define gpuMallocFromPoolAsync           hipMallocFromPoolAsync
#define gpuMallocManaged                 hipMallocManaged
#define gpuMallocMipmappedArray          hipMallocMipmappedArray
#define gpuMallocPitch                   hipMallocPitch
#define gpuMemAccessDesc                 hipMemAccessDesc
#define gpuMemAccessFlags                hipMemAccessFlags
#define gpuMemAccessFlagsProtNone        hipMemAccessFlagsProtNone
#define gpuMemAccessFlagsProtRead        hipMemAccessFlagsProtRead
#define gpuMemAccessFlagsProtReadWrite   hipMemAccessFlagsProtReadWrite
#define gpuMemAddressFree                hipMemAddressFree
#define gpuMemAddressReserve             hipMemAddressReserve
#define gpuMemAdvise                     hipMemAdvise
#define gpuMemAdviseSetAccessedBy        hipMemAdviseSetAccessedBy
#define gpuMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation
#define gpuMemAdviseSetReadMostly        hipMemAdviseSetReadMostly
#define gpuMemAdviseUnsetAccessedBy      hipMemAdviseUnsetAccessedBy
#define gpuMemAdviseUnsetPreferredLocation  \
        hipMemAdviseUnsetPreferredLocation
#define gpuMemAdviseUnsetReadMostly      hipMemAdviseUnsetReadMostly
#define gpuMemAllocHost                  hipMemAllocHost
#define gpuMemAllocNodeParams            hipMemAllocNodeParams
#define gpuMemAllocPitch                 hipMemAllocPitch
#define gpuMemAllocationGranularityMinimum  \
        hipMemAllocationGranularityMinimum
#define gpuMemAllocationGranularityRecommended  \
        hipMemAllocationGranularityRecommended
#define gpuMemAllocationGranularity_flags  \
        hipMemAllocationGranularity_flags
#define gpuMemAllocationHandleType       hipMemAllocationHandleType
#define gpuMemAllocationProp             hipMemAllocationProp
#define gpuMemAllocationType             hipMemAllocationType
#define gpuMemAllocationTypeInvalid      hipMemAllocationTypeInvalid
#define gpuMemAllocationTypeMax          hipMemAllocationTypeMax
#define gpuMemAllocationTypePinned       hipMemAllocationTypePinned
#define gpuMemAttachGlobal               hipMemAttachGlobal
#define gpuMemAttachHost                 hipMemAttachHost
#define gpuMemAttachSingle               hipMemAttachSingle
#define gpuMemCreate                     hipMemCreate
#define gpuMemExportToShareableHandle    hipMemExportToShareableHandle
#define gpuMemGenericAllocationHandle_t  hipMemGenericAllocationHandle_t
#define gpuMemGetAccess                  hipMemGetAccess
#define gpuMemGetAddressRange            hipMemGetAddressRange
#define gpuMemGetAllocationGranularity   hipMemGetAllocationGranularity
#define gpuMemGetAllocationPropertiesFromHandle  \
        hipMemGetAllocationPropertiesFromHandle
#define gpuMemGetInfo                    hipMemGetInfo
#define gpuMemHandleType                 hipMemHandleType
#define gpuMemHandleTypeGeneric          hipMemHandleTypeGeneric
#define gpuMemHandleTypeNone             hipMemHandleTypeNone
#define gpuMemHandleTypePosixFileDescriptor  \
        hipMemHandleTypePosixFileDescriptor
#define gpuMemHandleTypeWin32            hipMemHandleTypeWin32
#define gpuMemHandleTypeWin32Kmt         hipMemHandleTypeWin32Kmt
#define gpuMemImportFromShareableHandle  hipMemImportFromShareableHandle
#define gpuMemLocation                   hipMemLocation
#define gpuMemLocationType               hipMemLocationType
#define gpuMemLocationTypeDevice         hipMemLocationTypeDevice
#define gpuMemLocationTypeInvalid        hipMemLocationTypeInvalid
#define gpuMemMap                        hipMemMap
#define gpuMemMapArrayAsync              hipMemMapArrayAsync
#define gpuMemOperationType              hipMemOperationType
#define gpuMemOperationTypeMap           hipMemOperationTypeMap
#define gpuMemOperationTypeUnmap         hipMemOperationTypeUnmap
#define gpuMemPoolAttr                   hipMemPoolAttr
#define gpuMemPoolAttrReleaseThreshold   hipMemPoolAttrReleaseThreshold
#define gpuMemPoolAttrReservedMemCurrent hipMemPoolAttrReservedMemCurrent
#define gpuMemPoolAttrReservedMemHigh    hipMemPoolAttrReservedMemHigh
#define gpuMemPoolAttrUsedMemCurrent     hipMemPoolAttrUsedMemCurrent
#define gpuMemPoolAttrUsedMemHigh        hipMemPoolAttrUsedMemHigh
#define gpuMemPoolCreate                 hipMemPoolCreate
#define gpuMemPoolDestroy                hipMemPoolDestroy
#define gpuMemPoolExportPointer          hipMemPoolExportPointer
#define gpuMemPoolExportToShareableHandle  \
        hipMemPoolExportToShareableHandle
#define gpuMemPoolGetAccess              hipMemPoolGetAccess
#define gpuMemPoolGetAttribute           hipMemPoolGetAttribute
#define gpuMemPoolImportFromShareableHandle  \
        hipMemPoolImportFromShareableHandle
#define gpuMemPoolImportPointer          hipMemPoolImportPointer
#define gpuMemPoolProps                  hipMemPoolProps
#define gpuMemPoolPtrExportData          hipMemPoolPtrExportData
#define gpuMemPoolReuseAllowInternalDependencies  \
        hipMemPoolReuseAllowInternalDependencies
#define gpuMemPoolReuseAllowOpportunistic  \
        hipMemPoolReuseAllowOpportunistic
#define gpuMemPoolReuseFollowEventDependencies  \
        hipMemPoolReuseFollowEventDependencies
#define gpuMemPoolSetAccess              hipMemPoolSetAccess
#define gpuMemPoolSetAttribute           hipMemPoolSetAttribute
#define gpuMemPoolTrimTo                 hipMemPoolTrimTo
#define gpuMemPool_t                     hipMemPool_t
#define gpuMemPrefetchAsync              hipMemPrefetchAsync
#define gpuMemRangeAttribute             hipMemRangeAttribute
#define gpuMemRangeAttributeAccessedBy   hipMemRangeAttributeAccessedBy
#define gpuMemRangeAttributeLastPrefetchLocation  \
        hipMemRangeAttributeLastPrefetchLocation
#define gpuMemRangeAttributePreferredLocation  \
        hipMemRangeAttributePreferredLocation
#define gpuMemRangeAttributeReadMostly   hipMemRangeAttributeReadMostly
#define gpuMemRangeGetAttribute          hipMemRangeGetAttribute
#define gpuMemRangeGetAttributes         hipMemRangeGetAttributes
#define gpuMemRelease                    hipMemRelease
#define gpuMemRetainAllocationHandle     hipMemRetainAllocationHandle
#define gpuMemSetAccess                  hipMemSetAccess
#define gpuMemUnmap                      hipMemUnmap
#define gpuMemcpy                        hipMemcpy
#define gpuMemcpy2D                      hipMemcpy2D
#define gpuMemcpy2DAsync                 hipMemcpy2DAsync
#define gpuMemcpy2DFromArray             hipMemcpy2DFromArray
#define gpuMemcpy2DFromArrayAsync        hipMemcpy2DFromArrayAsync
#define gpuMemcpy2DToArray               hipMemcpy2DToArray
#define gpuMemcpy2DToArrayAsync          hipMemcpy2DToArrayAsync
#define gpuMemcpy3D                      hipMemcpy3D
#define gpuMemcpy3DAsync                 hipMemcpy3DAsync
#define gpuMemcpyAsync                   hipMemcpyAsync
#define gpuMemcpyAtoH                    hipMemcpyAtoH
#define gpuMemcpyDtoD                    hipMemcpyDtoD
#define gpuMemcpyDtoDAsync               hipMemcpyDtoDAsync
#define gpuMemcpyDtoH                    hipMemcpyDtoH
#define gpuMemcpyDtoHAsync               hipMemcpyDtoHAsync
#define gpuMemcpyFromArray               hipMemcpyFromArray
#define gpuMemcpyFromSymbol              hipMemcpyFromSymbol
#define gpuMemcpyFromSymbolAsync         hipMemcpyFromSymbolAsync
#define gpuMemcpyHtoA                    hipMemcpyHtoA
#define gpuMemcpyHtoD                    hipMemcpyHtoD
#define gpuMemcpyHtoDAsync               hipMemcpyHtoDAsync
#define gpuMemcpyParam2D                 hipMemcpyParam2D
#define gpuMemcpyParam2DAsync            hipMemcpyParam2DAsync
#define gpuMemcpyPeer                    hipMemcpyPeer
#define gpuMemcpyPeerAsync               hipMemcpyPeerAsync
#define gpuMemcpyToArray                 hipMemcpyToArray
#define gpuMemcpyToSymbol                hipMemcpyToSymbol
#define gpuMemcpyToSymbolAsync           hipMemcpyToSymbolAsync
#define gpuMemoryAdvise                  hipMemoryAdvise
#define gpuMemoryType                    hipMemoryType
#define gpuMemoryTypeArray               hipMemoryTypeArray
#define gpuMemoryTypeDevice              hipMemoryTypeDevice
#define gpuMemoryTypeHost                hipMemoryTypeHost
#define gpuMemoryTypeManaged             hipMemoryTypeManaged
#define gpuMemoryTypeUnified             hipMemoryTypeUnified
#define gpuMemset                        hipMemset
#define gpuMemset2D                      hipMemset2D
#define gpuMemset2DAsync                 hipMemset2DAsync
#define gpuMemset3D                      hipMemset3D
#define gpuMemset3DAsync                 hipMemset3DAsync
#define gpuMemsetAsync                   hipMemsetAsync
#define gpuMemsetD16                     hipMemsetD16
#define gpuMemsetD16Async                hipMemsetD16Async
#define gpuMemsetD32                     hipMemsetD32
#define gpuMemsetD32Async                hipMemsetD32Async
#define gpuMemsetD8                      hipMemsetD8
#define gpuMemsetD8Async                 hipMemsetD8Async
#define gpuMemsetParams                  hipMemsetParams
#define gpuMipmappedArrayCreate          hipMipmappedArrayCreate
#define gpuMipmappedArrayDestroy         hipMipmappedArrayDestroy
#define gpuMipmappedArrayGetLevel        hipMipmappedArrayGetLevel
#define gpuModuleGetFunction             hipModuleGetFunction
#define gpuModuleGetGlobal               hipModuleGetGlobal
#define gpuModuleGetTexRef               hipModuleGetTexRef
#define gpuModuleLaunchCooperativeKernel hipModuleLaunchCooperativeKernel
#define gpuModuleLaunchCooperativeKernelMultiDevice  \
        hipModuleLaunchCooperativeKernelMultiDevice
#define gpuModuleLaunchKernel            hipModuleLaunchKernel
#define gpuModuleLoad                    hipModuleLoad
#define gpuModuleLoadData                hipModuleLoadData
#define gpuModuleLoadDataEx              hipModuleLoadDataEx
#define gpuModuleOccupancyMaxActiveBlocksPerMultiprocessor  \
        hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags  \
        hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define gpuModuleOccupancyMaxPotentialBlockSize  \
        hipModuleOccupancyMaxPotentialBlockSize
#define gpuModuleOccupancyMaxPotentialBlockSizeWithFlags  \
        hipModuleOccupancyMaxPotentialBlockSizeWithFlags
#define gpuModuleUnload                  hipModuleUnload
#define gpuModule_t                      hipModule_t
#define gpuOccupancyDefault              hipOccupancyDefault
#define gpuOccupancyDisableCachingOverride  \
        hipOccupancyDisableCachingOverride
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor  \
        hipOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags  \
        hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define gpuOccupancyMaxPotentialBlockSize  \
        hipOccupancyMaxPotentialBlockSize
#define gpuOccupancyMaxPotentialBlockSizeVariableSMem  \
        hipOccupancyMaxPotentialBlockSizeVariableSMem
#define gpuOccupancyMaxPotentialBlockSizeVariableSMemWithFlags  \
        hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
#define gpuOccupancyMaxPotentialBlockSizeWithFlags  \
        hipOccupancyMaxPotentialBlockSizeWithFlags
#define gpuPeekAtLastError               hipPeekAtLastError
#define gpuPointerAttribute_t            hipPointerAttribute_t
#define gpuPointerGetAttribute           hipPointerGetAttribute
#define gpuPointerGetAttributes          hipPointerGetAttributes
#define gpuPointerSetAttribute           hipPointerSetAttribute
#define gpuProfilerStart                 hipProfilerStart
#define gpuProfilerStop                  hipProfilerStop
#define gpuRuntimeGetVersion             hipRuntimeGetVersion
#define gpuSetDevice                     hipSetDevice
#define gpuSetDeviceFlags                hipSetDeviceFlags
#define gpuSharedMemBankSizeDefault      hipSharedMemBankSizeDefault
#define gpuSharedMemBankSizeEightByte    hipSharedMemBankSizeEightByte
#define gpuSharedMemBankSizeFourByte     hipSharedMemBankSizeFourByte
#define gpuSharedMemConfig               hipSharedMemConfig
#define gpuSignalExternalSemaphoresAsync hipSignalExternalSemaphoresAsync
#define gpuStreamAddCallback             hipStreamAddCallback
#define gpuStreamAddCaptureDependencies  hipStreamAddCaptureDependencies
#define gpuStreamAttachMemAsync          hipStreamAttachMemAsync
#define gpuStreamBeginCapture            hipStreamBeginCapture
#define gpuStreamCallback_t              hipStreamCallback_t
#define gpuStreamCaptureMode             hipStreamCaptureMode
#define gpuStreamCaptureModeGlobal       hipStreamCaptureModeGlobal
#define gpuStreamCaptureModeRelaxed      hipStreamCaptureModeRelaxed
#define gpuStreamCaptureModeThreadLocal  hipStreamCaptureModeThreadLocal
#define gpuStreamCaptureStatus           hipStreamCaptureStatus
#define gpuStreamCaptureStatusActive     hipStreamCaptureStatusActive
#define gpuStreamCaptureStatusInvalidated  \
        hipStreamCaptureStatusInvalidated
#define gpuStreamCaptureStatusNone       hipStreamCaptureStatusNone
#define gpuStreamCreate                  hipStreamCreate
#define gpuStreamCreateWithFlags         hipStreamCreateWithFlags
#define gpuStreamCreateWithPriority      hipStreamCreateWithPriority
#define gpuStreamDefault                 hipStreamDefault
#define gpuStreamDestroy                 hipStreamDestroy
#define gpuStreamEndCapture              hipStreamEndCapture
#define gpuStreamGetCaptureInfo          hipStreamGetCaptureInfo
#define gpuStreamGetCaptureInfo_v2       hipStreamGetCaptureInfo_v2
#define gpuStreamGetFlags                hipStreamGetFlags
#define gpuStreamGetPriority             hipStreamGetPriority
#define gpuStreamIsCapturing             hipStreamIsCapturing
#define gpuStreamNonBlocking             hipStreamNonBlocking
#define gpuStreamPerThread               hipStreamPerThread
#define gpuStreamQuery                   hipStreamQuery
#define gpuStreamSetCaptureDependencies  hipStreamSetCaptureDependencies
#define gpuStreamSynchronize             hipStreamSynchronize
#define gpuStreamUpdateCaptureDependencies  \
        hipStreamUpdateCaptureDependencies
#define gpuStreamUpdateCaptureDependenciesFlags  \
        hipStreamUpdateCaptureDependenciesFlags
#define gpuStreamWaitEvent               hipStreamWaitEvent
#define gpuStreamWaitValue32             hipStreamWaitValue32
#define gpuStreamWaitValue64             hipStreamWaitValue64
#define gpuStreamWaitValueAnd            hipStreamWaitValueAnd
#define gpuStreamWaitValueEq             hipStreamWaitValueEq
#define gpuStreamWaitValueGte            hipStreamWaitValueGte
#define gpuStreamWaitValueNor            hipStreamWaitValueNor
#define gpuStreamWriteValue32            hipStreamWriteValue32
#define gpuStreamWriteValue64            hipStreamWriteValue64
#define gpuStream_t                      hipStream_t
#define gpuSuccess                       hipSuccess
#define gpuTexObjectCreate               hipTexObjectCreate
#define gpuTexObjectDestroy              hipTexObjectDestroy
#define gpuTexObjectGetResourceDesc      hipTexObjectGetResourceDesc
#define gpuTexObjectGetResourceViewDesc  hipTexObjectGetResourceViewDesc
#define gpuTexObjectGetTextureDesc       hipTexObjectGetTextureDesc
#define gpuTexRefGetAddress              hipTexRefGetAddress
#define gpuTexRefGetAddressMode          hipTexRefGetAddressMode
#define gpuTexRefGetFilterMode           hipTexRefGetFilterMode
#define gpuTexRefGetFlags                hipTexRefGetFlags
#define gpuTexRefGetFormat               hipTexRefGetFormat
#define gpuTexRefGetMaxAnisotropy        hipTexRefGetMaxAnisotropy
#define gpuTexRefGetMipMappedArray       hipTexRefGetMipMappedArray
#define gpuTexRefGetMipmapFilterMode     hipTexRefGetMipmapFilterMode
#define gpuTexRefGetMipmapLevelBias      hipTexRefGetMipmapLevelBias
#define gpuTexRefGetMipmapLevelClamp     hipTexRefGetMipmapLevelClamp
#define gpuTexRefSetAddress              hipTexRefSetAddress
#define gpuTexRefSetAddress2D            hipTexRefSetAddress2D
#define gpuTexRefSetAddressMode          hipTexRefSetAddressMode
#define gpuTexRefSetArray                hipTexRefSetArray
#define gpuTexRefSetBorderColor          hipTexRefSetBorderColor
#define gpuTexRefSetFilterMode           hipTexRefSetFilterMode
#define gpuTexRefSetFlags                hipTexRefSetFlags
#define gpuTexRefSetFormat               hipTexRefSetFormat
#define gpuTexRefSetMaxAnisotropy        hipTexRefSetMaxAnisotropy
#define gpuTexRefSetMipmapFilterMode     hipTexRefSetMipmapFilterMode
#define gpuTexRefSetMipmapLevelBias      hipTexRefSetMipmapLevelBias
#define gpuTexRefSetMipmapLevelClamp     hipTexRefSetMipmapLevelClamp
#define gpuTexRefSetMipmappedArray       hipTexRefSetMipmappedArray
#define gpuThreadExchangeStreamCaptureMode  \
        hipThreadExchangeStreamCaptureMode
#define gpuUUID                          hipUUID
#define gpuUUID_t                        hipUUID_t
#define gpuUnbindTexture                 hipUnbindTexture
#define gpuUserObjectCreate              hipUserObjectCreate
#define gpuUserObjectFlags               hipUserObjectFlags
#define gpuUserObjectNoDestructorSync    hipUserObjectNoDestructorSync
#define gpuUserObjectRelease             hipUserObjectRelease
#define gpuUserObjectRetain              hipUserObjectRetain
#define gpuUserObjectRetainFlags         hipUserObjectRetainFlags
#define gpuUserObject_t                  hipUserObject_t
#define gpuWaitExternalSemaphoresAsync   hipWaitExternalSemaphoresAsync

/* channel_descriptor.h */
#define gpuCreateChannelDesc             hipCreateChannelDesc

/* driver_types.h */
#define GPU_AD_FORMAT_FLOAT              HIP_AD_FORMAT_FLOAT
#define GPU_AD_FORMAT_HALF               HIP_AD_FORMAT_HALF
#define GPU_AD_FORMAT_SIGNED_INT16       HIP_AD_FORMAT_SIGNED_INT16
#define GPU_AD_FORMAT_SIGNED_INT32       HIP_AD_FORMAT_SIGNED_INT32
#define GPU_AD_FORMAT_SIGNED_INT8        HIP_AD_FORMAT_SIGNED_INT8
#define GPU_AD_FORMAT_UNSIGNED_INT16     HIP_AD_FORMAT_UNSIGNED_INT16
#define GPU_AD_FORMAT_UNSIGNED_INT32     HIP_AD_FORMAT_UNSIGNED_INT32
#define GPU_AD_FORMAT_UNSIGNED_INT8      HIP_AD_FORMAT_UNSIGNED_INT8
#define GPU_ARRAY3D_DESCRIPTOR           HIP_ARRAY3D_DESCRIPTOR
#define GPU_ARRAY_DESCRIPTOR             HIP_ARRAY_DESCRIPTOR
#define GPU_FUNC_ATTRIBUTE_BINARY_VERSION  \
        HIP_FUNC_ATTRIBUTE_BINARY_VERSION
#define GPU_FUNC_ATTRIBUTE_CACHE_MODE_CA HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA
#define GPU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES  \
        HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
#define GPU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES  \
        HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define GPU_FUNC_ATTRIBUTE_MAX           HIP_FUNC_ATTRIBUTE_MAX
#define GPU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES  \
        HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define GPU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK  \
        HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define GPU_FUNC_ATTRIBUTE_NUM_REGS      HIP_FUNC_ATTRIBUTE_NUM_REGS
#define GPU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT  \
        HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
#define GPU_FUNC_ATTRIBUTE_PTX_VERSION   HIP_FUNC_ATTRIBUTE_PTX_VERSION
#define GPU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES  \
        HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define GPU_MEMCPY3D                     HIP_MEMCPY3D
#define GPU_POINTER_ATTRIBUTE_ACCESS_FLAGS  \
        HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
#define GPU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES  \
        HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
#define GPU_POINTER_ATTRIBUTE_BUFFER_ID  HIP_POINTER_ATTRIBUTE_BUFFER_ID
#define GPU_POINTER_ATTRIBUTE_CONTEXT    HIP_POINTER_ATTRIBUTE_CONTEXT
#define GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL  \
        HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define GPU_POINTER_ATTRIBUTE_DEVICE_POINTER  \
        HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
#define GPU_POINTER_ATTRIBUTE_HOST_POINTER  \
        HIP_POINTER_ATTRIBUTE_HOST_POINTER
#define GPU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE  \
        HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
#define GPU_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE  \
        HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
#define GPU_POINTER_ATTRIBUTE_IS_MANAGED HIP_POINTER_ATTRIBUTE_IS_MANAGED
#define GPU_POINTER_ATTRIBUTE_MAPPED     HIP_POINTER_ATTRIBUTE_MAPPED
#define GPU_POINTER_ATTRIBUTE_MEMORY_TYPE  \
        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
#define GPU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE  \
        HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#define GPU_POINTER_ATTRIBUTE_P2P_TOKENS HIP_POINTER_ATTRIBUTE_P2P_TOKENS
#define GPU_POINTER_ATTRIBUTE_RANGE_SIZE HIP_POINTER_ATTRIBUTE_RANGE_SIZE
#define GPU_POINTER_ATTRIBUTE_RANGE_START_ADDR  \
        HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define GPU_POINTER_ATTRIBUTE_SYNC_MEMOPS  \
        HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
#define GPU_RESOURCE_DESC                HIP_RESOURCE_DESC
#define GPU_RESOURCE_DESC_st             HIP_RESOURCE_DESC_st
#define GPU_RESOURCE_TYPE_ARRAY          HIP_RESOURCE_TYPE_ARRAY
#define GPU_RESOURCE_TYPE_LINEAR         HIP_RESOURCE_TYPE_LINEAR
#define GPU_RESOURCE_TYPE_MIPMAPPED_ARRAY  \
        HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY
#define GPU_RESOURCE_TYPE_PITCH2D        HIP_RESOURCE_TYPE_PITCH2D
#define GPU_RESOURCE_VIEW_DESC           HIP_RESOURCE_VIEW_DESC
#define GPU_RESOURCE_VIEW_DESC_st        HIP_RESOURCE_VIEW_DESC_st
#define GPU_RES_VIEW_FORMAT_FLOAT_1X16   HIP_RES_VIEW_FORMAT_FLOAT_1X16
#define GPU_RES_VIEW_FORMAT_FLOAT_1X32   HIP_RES_VIEW_FORMAT_FLOAT_1X32
#define GPU_RES_VIEW_FORMAT_FLOAT_2X16   HIP_RES_VIEW_FORMAT_FLOAT_2X16
#define GPU_RES_VIEW_FORMAT_FLOAT_2X32   HIP_RES_VIEW_FORMAT_FLOAT_2X32
#define GPU_RES_VIEW_FORMAT_FLOAT_4X16   HIP_RES_VIEW_FORMAT_FLOAT_4X16
#define GPU_RES_VIEW_FORMAT_FLOAT_4X32   HIP_RES_VIEW_FORMAT_FLOAT_4X32
#define GPU_RES_VIEW_FORMAT_NONE         HIP_RES_VIEW_FORMAT_NONE
#define GPU_RES_VIEW_FORMAT_SIGNED_BC4   HIP_RES_VIEW_FORMAT_SIGNED_BC4
#define GPU_RES_VIEW_FORMAT_SIGNED_BC5   HIP_RES_VIEW_FORMAT_SIGNED_BC5
#define GPU_RES_VIEW_FORMAT_SIGNED_BC6H  HIP_RES_VIEW_FORMAT_SIGNED_BC6H
#define GPU_RES_VIEW_FORMAT_SINT_1X16    HIP_RES_VIEW_FORMAT_SINT_1X16
#define GPU_RES_VIEW_FORMAT_SINT_1X32    HIP_RES_VIEW_FORMAT_SINT_1X32
#define GPU_RES_VIEW_FORMAT_SINT_1X8     HIP_RES_VIEW_FORMAT_SINT_1X8
#define GPU_RES_VIEW_FORMAT_SINT_2X16    HIP_RES_VIEW_FORMAT_SINT_2X16
#define GPU_RES_VIEW_FORMAT_SINT_2X32    HIP_RES_VIEW_FORMAT_SINT_2X32
#define GPU_RES_VIEW_FORMAT_SINT_2X8     HIP_RES_VIEW_FORMAT_SINT_2X8
#define GPU_RES_VIEW_FORMAT_SINT_4X16    HIP_RES_VIEW_FORMAT_SINT_4X16
#define GPU_RES_VIEW_FORMAT_SINT_4X32    HIP_RES_VIEW_FORMAT_SINT_4X32
#define GPU_RES_VIEW_FORMAT_SINT_4X8     HIP_RES_VIEW_FORMAT_SINT_4X8
#define GPU_RES_VIEW_FORMAT_UINT_1X16    HIP_RES_VIEW_FORMAT_UINT_1X16
#define GPU_RES_VIEW_FORMAT_UINT_1X32    HIP_RES_VIEW_FORMAT_UINT_1X32
#define GPU_RES_VIEW_FORMAT_UINT_1X8     HIP_RES_VIEW_FORMAT_UINT_1X8
#define GPU_RES_VIEW_FORMAT_UINT_2X16    HIP_RES_VIEW_FORMAT_UINT_2X16
#define GPU_RES_VIEW_FORMAT_UINT_2X32    HIP_RES_VIEW_FORMAT_UINT_2X32
#define GPU_RES_VIEW_FORMAT_UINT_2X8     HIP_RES_VIEW_FORMAT_UINT_2X8
#define GPU_RES_VIEW_FORMAT_UINT_4X16    HIP_RES_VIEW_FORMAT_UINT_4X16
#define GPU_RES_VIEW_FORMAT_UINT_4X32    HIP_RES_VIEW_FORMAT_UINT_4X32
#define GPU_RES_VIEW_FORMAT_UINT_4X8     HIP_RES_VIEW_FORMAT_UINT_4X8
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC1 HIP_RES_VIEW_FORMAT_UNSIGNED_BC1
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC2 HIP_RES_VIEW_FORMAT_UNSIGNED_BC2
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC3 HIP_RES_VIEW_FORMAT_UNSIGNED_BC3
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC4 HIP_RES_VIEW_FORMAT_UNSIGNED_BC4
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC5 HIP_RES_VIEW_FORMAT_UNSIGNED_BC5
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC6H  \
        HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H
#define GPU_RES_VIEW_FORMAT_UNSIGNED_BC7 HIP_RES_VIEW_FORMAT_UNSIGNED_BC7
#define GPU_TEXTURE_DESC                 HIP_TEXTURE_DESC
#define GPU_TEXTURE_DESC_st              HIP_TEXTURE_DESC_st
#define GPU_TRSA_OVERRIDE_FORMAT         HIP_TRSA_OVERRIDE_FORMAT
#define GPU_TRSF_NORMALIZED_COORDINATES  HIP_TRSF_NORMALIZED_COORDINATES
#define GPU_TRSF_READ_AS_INTEGER         HIP_TRSF_READ_AS_INTEGER
#define GPU_TRSF_SRGB                    HIP_TRSF_SRGB
#define GPU_TR_ADDRESS_MODE_BORDER       HIP_TR_ADDRESS_MODE_BORDER
#define GPU_TR_ADDRESS_MODE_CLAMP        HIP_TR_ADDRESS_MODE_CLAMP
#define GPU_TR_ADDRESS_MODE_MIRROR       HIP_TR_ADDRESS_MODE_MIRROR
#define GPU_TR_ADDRESS_MODE_WRAP         HIP_TR_ADDRESS_MODE_WRAP
#define GPU_TR_FILTER_MODE_LINEAR        HIP_TR_FILTER_MODE_LINEAR
#define GPU_TR_FILTER_MODE_POINT         HIP_TR_FILTER_MODE_POINT
#define GPUaddress_mode                  HIPaddress_mode
#define GPUaddress_mode_enum             HIPaddress_mode_enum
#define GPUfilter_mode                   HIPfilter_mode
#define GPUfilter_mode_enum              HIPfilter_mode_enum
#define GPUresourceViewFormat            HIPresourceViewFormat
#define GPUresourceViewFormat_enum       HIPresourceViewFormat_enum
#define GPUresourcetype                  HIPresourcetype
#define GPUresourcetype_enum             HIPresourcetype_enum
#define gpuArray                         hipArray
#define gpuArray_Format                  hipArray_Format
#define gpuArray_const_t                 hipArray_const_t
#define gpuArray_t                       hipArray_t
#define gpuChannelFormatDesc             hipChannelFormatDesc
#define gpuChannelFormatKind             hipChannelFormatKind
#define gpuChannelFormatKindFloat        hipChannelFormatKindFloat
#define gpuChannelFormatKindNone         hipChannelFormatKindNone
#define gpuChannelFormatKindSigned       hipChannelFormatKindSigned
#define gpuChannelFormatKindUnsigned     hipChannelFormatKindUnsigned
#define gpuDeviceptr_t                   hipDeviceptr_t
#define gpuExtent                        hipExtent
#define gpuFunction_attribute            hipFunction_attribute
#define gpuMemcpy3DParms                 hipMemcpy3DParms
#define gpuMemcpyDefault                 hipMemcpyDefault
#define gpuMemcpyDeviceToDevice          hipMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost            hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice            hipMemcpyHostToDevice
#define gpuMemcpyHostToHost              hipMemcpyHostToHost
#define gpuMemcpyKind                    hipMemcpyKind
#define gpuMipmappedArray                hipMipmappedArray
#define gpuMipmappedArray_const_t        hipMipmappedArray_const_t
#define gpuMipmappedArray_t              hipMipmappedArray_t
#define gpuPitchedPtr                    hipPitchedPtr
#define gpuPointer_attribute             hipPointer_attribute
#define gpuPos                           hipPos
#define gpuResViewFormatFloat1           hipResViewFormatFloat1
#define gpuResViewFormatFloat2           hipResViewFormatFloat2
#define gpuResViewFormatFloat4           hipResViewFormatFloat4
#define gpuResViewFormatHalf1            hipResViewFormatHalf1
#define gpuResViewFormatHalf2            hipResViewFormatHalf2
#define gpuResViewFormatHalf4            hipResViewFormatHalf4
#define gpuResViewFormatNone             hipResViewFormatNone
#define gpuResViewFormatSignedBlockCompressed4  \
        hipResViewFormatSignedBlockCompressed4
#define gpuResViewFormatSignedBlockCompressed5  \
        hipResViewFormatSignedBlockCompressed5
#define gpuResViewFormatSignedBlockCompressed6H  \
        hipResViewFormatSignedBlockCompressed6H
#define gpuResViewFormatSignedChar1      hipResViewFormatSignedChar1
#define gpuResViewFormatSignedChar2      hipResViewFormatSignedChar2
#define gpuResViewFormatSignedChar4      hipResViewFormatSignedChar4
#define gpuResViewFormatSignedInt1       hipResViewFormatSignedInt1
#define gpuResViewFormatSignedInt2       hipResViewFormatSignedInt2
#define gpuResViewFormatSignedInt4       hipResViewFormatSignedInt4
#define gpuResViewFormatSignedShort1     hipResViewFormatSignedShort1
#define gpuResViewFormatSignedShort2     hipResViewFormatSignedShort2
#define gpuResViewFormatSignedShort4     hipResViewFormatSignedShort4
#define gpuResViewFormatUnsignedBlockCompressed1  \
        hipResViewFormatUnsignedBlockCompressed1
#define gpuResViewFormatUnsignedBlockCompressed2  \
        hipResViewFormatUnsignedBlockCompressed2
#define gpuResViewFormatUnsignedBlockCompressed3  \
        hipResViewFormatUnsignedBlockCompressed3
#define gpuResViewFormatUnsignedBlockCompressed4  \
        hipResViewFormatUnsignedBlockCompressed4
#define gpuResViewFormatUnsignedBlockCompressed5  \
        hipResViewFormatUnsignedBlockCompressed5
#define gpuResViewFormatUnsignedBlockCompressed6H  \
        hipResViewFormatUnsignedBlockCompressed6H
#define gpuResViewFormatUnsignedBlockCompressed7  \
        hipResViewFormatUnsignedBlockCompressed7
#define gpuResViewFormatUnsignedChar1    hipResViewFormatUnsignedChar1
#define gpuResViewFormatUnsignedChar2    hipResViewFormatUnsignedChar2
#define gpuResViewFormatUnsignedChar4    hipResViewFormatUnsignedChar4
#define gpuResViewFormatUnsignedInt1     hipResViewFormatUnsignedInt1
#define gpuResViewFormatUnsignedInt2     hipResViewFormatUnsignedInt2
#define gpuResViewFormatUnsignedInt4     hipResViewFormatUnsignedInt4
#define gpuResViewFormatUnsignedShort1   hipResViewFormatUnsignedShort1
#define gpuResViewFormatUnsignedShort2   hipResViewFormatUnsignedShort2
#define gpuResViewFormatUnsignedShort4   hipResViewFormatUnsignedShort4
#define gpuResourceDesc                  hipResourceDesc
#define gpuResourceType                  hipResourceType
#define gpuResourceTypeArray             hipResourceTypeArray
#define gpuResourceTypeLinear            hipResourceTypeLinear
#define gpuResourceTypeMipmappedArray    hipResourceTypeMipmappedArray
#define gpuResourceTypePitch2D           hipResourceTypePitch2D
#define gpuResourceViewDesc              hipResourceViewDesc
#define gpuResourceViewFormat            hipResourceViewFormat
#define gpu_Memcpy2D                     hip_Memcpy2D
#define make_gpuExtent                   make_hipExtent
#define make_gpuPitchedPtr               make_hipPitchedPtr
#define make_gpuPos                      make_hipPos

/* surface_types.h */
#define gpuBoundaryModeClamp             hipBoundaryModeClamp
#define gpuBoundaryModeTrap              hipBoundaryModeTrap
#define gpuBoundaryModeZero              hipBoundaryModeZero
#define gpuSurfaceBoundaryMode           hipSurfaceBoundaryMode
#define gpuSurfaceObject_t               hipSurfaceObject_t

/* texture_types.h */
#define gpuAddressModeBorder             hipAddressModeBorder
#define gpuAddressModeClamp              hipAddressModeClamp
#define gpuAddressModeMirror             hipAddressModeMirror
#define gpuAddressModeWrap               hipAddressModeWrap
#define gpuFilterModeLinear              hipFilterModeLinear
#define gpuFilterModePoint               hipFilterModePoint
#define gpuReadModeElementType           hipReadModeElementType
#define gpuReadModeNormalizedFloat       hipReadModeNormalizedFloat
#define gpuTexRef                        hipTexRef
#define gpuTextureAddressMode            hipTextureAddressMode
#define gpuTextureDesc                   hipTextureDesc
#define gpuTextureFilterMode             hipTextureFilterMode
#define gpuTextureObject_t               hipTextureObject_t
#define gpuTextureReadMode               hipTextureReadMode
#define gpuTextureType1D                 hipTextureType1D
#define gpuTextureType1DLayered          hipTextureType1DLayered
#define gpuTextureType2D                 hipTextureType2D
#define gpuTextureType2DLayered          hipTextureType2DLayered
#define gpuTextureType3D                 hipTextureType3D
#define gpuTextureTypeCubemap            hipTextureTypeCubemap
#define gpuTextureTypeCubemapLayered     hipTextureTypeCubemapLayered

#endif
