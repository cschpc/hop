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

#ifndef __HOP_SOURCE_HIP_HIP_RUNTIME_API_H__
#define __HOP_SOURCE_HIP_HIP_RUNTIME_API_H__

#if !defined(HOP_SOURCE_HIP)
#define HOP_SOURCE_HIP
#endif

#define HIP_IPC_HANDLE_SIZE              GPU_IPC_HANDLE_SIZE
#define HIP_LAUNCH_PARAM_BUFFER_POINTER  GPU_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_SIZE     GPU_LAUNCH_PARAM_BUFFER_SIZE
#define HIP_LAUNCH_PARAM_END             GPU_LAUNCH_PARAM_END
#define hipAccessPolicyWindow            gpuAccessPolicyWindow
#define hipAccessProperty                gpuAccessProperty
#define hipAccessPropertyNormal          gpuAccessPropertyNormal
#define hipAccessPropertyPersisting      gpuAccessPropertyPersisting
#define hipAccessPropertyStreaming       gpuAccessPropertyStreaming
#define hipArray3DCreate                 gpuArray3DCreate
#define hipArray3DGetDescriptor          gpuArray3DGetDescriptor
#define hipArrayCreate                   gpuArrayCreate
#define hipArrayCubemap                  gpuArrayCubemap
#define hipArrayDefault                  gpuArrayDefault
#define hipArrayDestroy                  gpuArrayDestroy
#define hipArrayGetDescriptor            gpuArrayGetDescriptor
#define hipArrayGetInfo                  gpuArrayGetInfo
#define hipArrayLayered                  gpuArrayLayered
#define hipArrayMapInfo                  gpuArrayMapInfo
#define hipArraySparseSubresourceType    gpuArraySparseSubresourceType
#define hipArraySparseSubresourceTypeMiptail  \
        gpuArraySparseSubresourceTypeMiptail
#define hipArraySparseSubresourceTypeSparseLevel  \
        gpuArraySparseSubresourceTypeSparseLevel
#define hipArraySurfaceLoadStore         gpuArraySurfaceLoadStore
#define hipArrayTextureGather            gpuArrayTextureGather
#define hipBindTexture                   gpuBindTexture
#define hipBindTexture2D                 gpuBindTexture2D
#define hipBindTextureToArray            gpuBindTextureToArray
#define hipBindTextureToMipmappedArray   gpuBindTextureToMipmappedArray
#define hipChooseDevice                  gpuChooseDevice
#define hipComputeMode                   gpuComputeMode
#define hipComputeModeDefault            gpuComputeModeDefault
#define hipComputeModeExclusive          gpuComputeModeExclusive
#define hipComputeModeExclusiveProcess   gpuComputeModeExclusiveProcess
#define hipComputeModeProhibited         gpuComputeModeProhibited
#define hipCooperativeLaunchMultiDeviceNoPostSync  \
        gpuCooperativeLaunchMultiDeviceNoPostSync
#define hipCooperativeLaunchMultiDeviceNoPreSync  \
        gpuCooperativeLaunchMultiDeviceNoPreSync
#define hipCpuDeviceId                   gpuCpuDeviceId
#define hipCreateSurfaceObject           gpuCreateSurfaceObject
#define hipCreateTextureObject           gpuCreateTextureObject
#define hipCtxCreate                     gpuCtxCreate
#define hipCtxDestroy                    gpuCtxDestroy
#define hipCtxDisablePeerAccess          gpuCtxDisablePeerAccess
#define hipCtxEnablePeerAccess           gpuCtxEnablePeerAccess
#define hipCtxGetApiVersion              gpuCtxGetApiVersion
#define hipCtxGetCacheConfig             gpuCtxGetCacheConfig
#define hipCtxGetCurrent                 gpuCtxGetCurrent
#define hipCtxGetDevice                  gpuCtxGetDevice
#define hipCtxGetFlags                   gpuCtxGetFlags
#define hipCtxGetSharedMemConfig         gpuCtxGetSharedMemConfig
#define hipCtxPopCurrent                 gpuCtxPopCurrent
#define hipCtxPushCurrent                gpuCtxPushCurrent
#define hipCtxSetCacheConfig             gpuCtxSetCacheConfig
#define hipCtxSetCurrent                 gpuCtxSetCurrent
#define hipCtxSetSharedMemConfig         gpuCtxSetSharedMemConfig
#define hipCtxSynchronize                gpuCtxSynchronize
#define hipCtx_t                         gpuCtx_t
#define hipDestroyExternalMemory         gpuDestroyExternalMemory
#define hipDestroyExternalSemaphore      gpuDestroyExternalSemaphore
#define hipDestroySurfaceObject          gpuDestroySurfaceObject
#define hipDestroyTextureObject          gpuDestroyTextureObject
#define hipDevP2PAttrAccessSupported     gpuDevP2PAttrAccessSupported
#define hipDevP2PAttrHipArrayAccessSupported  \
        gpuDevP2PAttrHipArrayAccessSupported
#define hipDevP2PAttrNativeAtomicSupported  \
        gpuDevP2PAttrNativeAtomicSupported
#define hipDevP2PAttrPerformanceRank     gpuDevP2PAttrPerformanceRank
#define hipDeviceAttributeAsyncEngineCount  \
        gpuDeviceAttributeAsyncEngineCount
#define hipDeviceAttributeCanMapHostMemory  \
        gpuDeviceAttributeCanMapHostMemory
#define hipDeviceAttributeCanUseHostPointerForRegisteredMem  \
        gpuDeviceAttributeCanUseHostPointerForRegisteredMem
#define hipDeviceAttributeCanUseStreamWaitValue  \
        gpuDeviceAttributeCanUseStreamWaitValue
#define hipDeviceAttributeClockRate      gpuDeviceAttributeClockRate
#define hipDeviceAttributeComputeCapabilityMajor  \
        gpuDeviceAttributeComputeCapabilityMajor
#define hipDeviceAttributeComputeCapabilityMinor  \
        gpuDeviceAttributeComputeCapabilityMinor
#define hipDeviceAttributeComputeMode    gpuDeviceAttributeComputeMode
#define hipDeviceAttributeComputePreemptionSupported  \
        gpuDeviceAttributeComputePreemptionSupported
#define hipDeviceAttributeConcurrentKernels  \
        gpuDeviceAttributeConcurrentKernels
#define hipDeviceAttributeConcurrentManagedAccess  \
        gpuDeviceAttributeConcurrentManagedAccess
#define hipDeviceAttributeCooperativeLaunch  \
        gpuDeviceAttributeCooperativeLaunch
#define hipDeviceAttributeCooperativeMultiDeviceLaunch  \
        gpuDeviceAttributeCooperativeMultiDeviceLaunch
#define hipDeviceAttributeDirectManagedMemAccessFromHost  \
        gpuDeviceAttributeDirectManagedMemAccessFromHost
#define hipDeviceAttributeEccEnabled     gpuDeviceAttributeEccEnabled
#define hipDeviceAttributeGlobalL1CacheSupported  \
        gpuDeviceAttributeGlobalL1CacheSupported
#define hipDeviceAttributeHostNativeAtomicSupported  \
        gpuDeviceAttributeHostNativeAtomicSupported
#define hipDeviceAttributeIntegrated     gpuDeviceAttributeIntegrated
#define hipDeviceAttributeIsMultiGpuBoard  \
        gpuDeviceAttributeIsMultiGpuBoard
#define hipDeviceAttributeKernelExecTimeout  \
        gpuDeviceAttributeKernelExecTimeout
#define hipDeviceAttributeL2CacheSize    gpuDeviceAttributeL2CacheSize
#define hipDeviceAttributeLocalL1CacheSupported  \
        gpuDeviceAttributeLocalL1CacheSupported
#define hipDeviceAttributeManagedMemory  gpuDeviceAttributeManagedMemory
#define hipDeviceAttributeMaxBlockDimX   gpuDeviceAttributeMaxBlockDimX
#define hipDeviceAttributeMaxBlockDimY   gpuDeviceAttributeMaxBlockDimY
#define hipDeviceAttributeMaxBlockDimZ   gpuDeviceAttributeMaxBlockDimZ
#define hipDeviceAttributeMaxBlocksPerMultiProcessor  \
        gpuDeviceAttributeMaxBlocksPerMultiProcessor
#define hipDeviceAttributeMaxGridDimX    gpuDeviceAttributeMaxGridDimX
#define hipDeviceAttributeMaxGridDimY    gpuDeviceAttributeMaxGridDimY
#define hipDeviceAttributeMaxGridDimZ    gpuDeviceAttributeMaxGridDimZ
#define hipDeviceAttributeMaxPitch       gpuDeviceAttributeMaxPitch
#define hipDeviceAttributeMaxRegistersPerBlock  \
        gpuDeviceAttributeMaxRegistersPerBlock
#define hipDeviceAttributeMaxRegistersPerMultiprocessor  \
        gpuDeviceAttributeMaxRegistersPerMultiprocessor
#define hipDeviceAttributeMaxSharedMemoryPerBlock  \
        gpuDeviceAttributeMaxSharedMemoryPerBlock
#define hipDeviceAttributeMaxSharedMemoryPerMultiprocessor  \
        gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define hipDeviceAttributeMaxSurface1D   gpuDeviceAttributeMaxSurface1D
#define hipDeviceAttributeMaxSurface1DLayered  \
        gpuDeviceAttributeMaxSurface1DLayered
#define hipDeviceAttributeMaxSurface2D   gpuDeviceAttributeMaxSurface2D
#define hipDeviceAttributeMaxSurface2DLayered  \
        gpuDeviceAttributeMaxSurface2DLayered
#define hipDeviceAttributeMaxSurface3D   gpuDeviceAttributeMaxSurface3D
#define hipDeviceAttributeMaxSurfaceCubemap  \
        gpuDeviceAttributeMaxSurfaceCubemap
#define hipDeviceAttributeMaxSurfaceCubemapLayered  \
        gpuDeviceAttributeMaxSurfaceCubemapLayered
#define hipDeviceAttributeMaxTexture1DLayered  \
        gpuDeviceAttributeMaxTexture1DLayered
#define hipDeviceAttributeMaxTexture1DLinear  \
        gpuDeviceAttributeMaxTexture1DLinear
#define hipDeviceAttributeMaxTexture1DMipmap  \
        gpuDeviceAttributeMaxTexture1DMipmap
#define hipDeviceAttributeMaxTexture1DWidth  \
        gpuDeviceAttributeMaxTexture1DWidth
#define hipDeviceAttributeMaxTexture2DGather  \
        gpuDeviceAttributeMaxTexture2DGather
#define hipDeviceAttributeMaxTexture2DHeight  \
        gpuDeviceAttributeMaxTexture2DHeight
#define hipDeviceAttributeMaxTexture2DLayered  \
        gpuDeviceAttributeMaxTexture2DLayered
#define hipDeviceAttributeMaxTexture2DLinear  \
        gpuDeviceAttributeMaxTexture2DLinear
#define hipDeviceAttributeMaxTexture2DMipmap  \
        gpuDeviceAttributeMaxTexture2DMipmap
#define hipDeviceAttributeMaxTexture2DWidth  \
        gpuDeviceAttributeMaxTexture2DWidth
#define hipDeviceAttributeMaxTexture3DAlt  \
        gpuDeviceAttributeMaxTexture3DAlt
#define hipDeviceAttributeMaxTexture3DDepth  \
        gpuDeviceAttributeMaxTexture3DDepth
#define hipDeviceAttributeMaxTexture3DHeight  \
        gpuDeviceAttributeMaxTexture3DHeight
#define hipDeviceAttributeMaxTexture3DWidth  \
        gpuDeviceAttributeMaxTexture3DWidth
#define hipDeviceAttributeMaxTextureCubemap  \
        gpuDeviceAttributeMaxTextureCubemap
#define hipDeviceAttributeMaxTextureCubemapLayered  \
        gpuDeviceAttributeMaxTextureCubemapLayered
#define hipDeviceAttributeMaxThreadsPerBlock  \
        gpuDeviceAttributeMaxThreadsPerBlock
#define hipDeviceAttributeMaxThreadsPerMultiProcessor  \
        gpuDeviceAttributeMaxThreadsPerMultiProcessor
#define hipDeviceAttributeMemoryBusWidth gpuDeviceAttributeMemoryBusWidth
#define hipDeviceAttributeMemoryClockRate  \
        gpuDeviceAttributeMemoryClockRate
#define hipDeviceAttributeMemoryPoolsSupported  \
        gpuDeviceAttributeMemoryPoolsSupported
#define hipDeviceAttributeMultiGpuBoardGroupID  \
        gpuDeviceAttributeMultiGpuBoardGroupID
#define hipDeviceAttributeMultiprocessorCount  \
        gpuDeviceAttributeMultiprocessorCount
#define hipDeviceAttributePageableMemoryAccess  \
        gpuDeviceAttributePageableMemoryAccess
#define hipDeviceAttributePageableMemoryAccessUsesHostPageTables  \
        gpuDeviceAttributePageableMemoryAccessUsesHostPageTables
#define hipDeviceAttributePciBusId       gpuDeviceAttributePciBusId
#define hipDeviceAttributePciDeviceId    gpuDeviceAttributePciDeviceId
#define hipDeviceAttributePciDomainID    gpuDeviceAttributePciDomainID
#define hipDeviceAttributeSharedMemPerBlockOptin  \
        gpuDeviceAttributeSharedMemPerBlockOptin
#define hipDeviceAttributeSingleToDoublePrecisionPerfRatio  \
        gpuDeviceAttributeSingleToDoublePrecisionPerfRatio
#define hipDeviceAttributeStreamPrioritiesSupported  \
        gpuDeviceAttributeStreamPrioritiesSupported
#define hipDeviceAttributeSurfaceAlignment  \
        gpuDeviceAttributeSurfaceAlignment
#define hipDeviceAttributeTccDriver      gpuDeviceAttributeTccDriver
#define hipDeviceAttributeTextureAlignment  \
        gpuDeviceAttributeTextureAlignment
#define hipDeviceAttributeTexturePitchAlignment  \
        gpuDeviceAttributeTexturePitchAlignment
#define hipDeviceAttributeTotalConstantMemory  \
        gpuDeviceAttributeTotalConstantMemory
#define hipDeviceAttributeUnifiedAddressing  \
        gpuDeviceAttributeUnifiedAddressing
#define hipDeviceAttributeVirtualMemoryManagementSupported  \
        gpuDeviceAttributeVirtualMemoryManagementSupported
#define hipDeviceAttributeWarpSize       gpuDeviceAttributeWarpSize
#define hipDeviceAttribute_t             gpuDeviceAttribute_t
#define hipDeviceCanAccessPeer           gpuDeviceCanAccessPeer
#define hipDeviceComputeCapability       gpuDeviceComputeCapability
#define hipDeviceDisablePeerAccess       gpuDeviceDisablePeerAccess
#define hipDeviceEnablePeerAccess        gpuDeviceEnablePeerAccess
#define hipDeviceGet                     gpuDeviceGet
#define hipDeviceGetAttribute            gpuDeviceGetAttribute
#define hipDeviceGetByPCIBusId           gpuDeviceGetByPCIBusId
#define hipDeviceGetCacheConfig          gpuDeviceGetCacheConfig
#define hipDeviceGetDefaultMemPool       gpuDeviceGetDefaultMemPool
#define hipDeviceGetGraphMemAttribute    gpuDeviceGetGraphMemAttribute
#define hipDeviceGetLimit                gpuDeviceGetLimit
#define hipDeviceGetMemPool              gpuDeviceGetMemPool
#define hipDeviceGetName                 gpuDeviceGetName
#define hipDeviceGetP2PAttribute         gpuDeviceGetP2PAttribute
#define hipDeviceGetPCIBusId             gpuDeviceGetPCIBusId
#define hipDeviceGetSharedMemConfig      gpuDeviceGetSharedMemConfig
#define hipDeviceGetStreamPriorityRange  gpuDeviceGetStreamPriorityRange
#define hipDeviceGetUuid                 gpuDeviceGetUuid
#define hipDeviceGraphMemTrim            gpuDeviceGraphMemTrim
#define hipDeviceLmemResizeToMax         gpuDeviceLmemResizeToMax
#define hipDeviceMapHost                 gpuDeviceMapHost
#define hipDeviceP2PAttr                 gpuDeviceP2PAttr
#define hipDevicePrimaryCtxGetState      gpuDevicePrimaryCtxGetState
#define hipDevicePrimaryCtxRelease       gpuDevicePrimaryCtxRelease
#define hipDevicePrimaryCtxReset         gpuDevicePrimaryCtxReset
#define hipDevicePrimaryCtxRetain        gpuDevicePrimaryCtxRetain
#define hipDevicePrimaryCtxSetFlags      gpuDevicePrimaryCtxSetFlags
#define hipDeviceProp_t                  gpuDeviceProp_t
#define hipDeviceReset                   gpuDeviceReset
#define hipDeviceScheduleAuto            gpuDeviceScheduleAuto
#define hipDeviceScheduleBlockingSync    gpuDeviceScheduleBlockingSync
#define hipDeviceScheduleMask            gpuDeviceScheduleMask
#define hipDeviceScheduleSpin            gpuDeviceScheduleSpin
#define hipDeviceScheduleYield           gpuDeviceScheduleYield
#define hipDeviceSetCacheConfig          gpuDeviceSetCacheConfig
#define hipDeviceSetGraphMemAttribute    gpuDeviceSetGraphMemAttribute
#define hipDeviceSetLimit                gpuDeviceSetLimit
#define hipDeviceSetMemPool              gpuDeviceSetMemPool
#define hipDeviceSetSharedMemConfig      gpuDeviceSetSharedMemConfig
#define hipDeviceSynchronize             gpuDeviceSynchronize
#define hipDeviceTotalMem                gpuDeviceTotalMem
#define hipDevice_t                      gpuDevice_t
#define hipDriverGetVersion              gpuDriverGetVersion
#define hipDrvGetErrorName               gpuDrvGetErrorName
#define hipDrvGetErrorString             gpuDrvGetErrorString
#define hipDrvMemcpy2DUnaligned          gpuDrvMemcpy2DUnaligned
#define hipDrvMemcpy3D                   gpuDrvMemcpy3D
#define hipDrvMemcpy3DAsync              gpuDrvMemcpy3DAsync
#define hipDrvPointerGetAttributes       gpuDrvPointerGetAttributes
#define hipErrorAlreadyAcquired          gpuErrorAlreadyAcquired
#define hipErrorAlreadyMapped            gpuErrorAlreadyMapped
#define hipErrorArrayIsMapped            gpuErrorArrayIsMapped
#define hipErrorAssert                   gpuErrorAssert
#define hipErrorCapturedEvent            gpuErrorCapturedEvent
#define hipErrorContextAlreadyCurrent    gpuErrorContextAlreadyCurrent
#define hipErrorContextAlreadyInUse      gpuErrorContextAlreadyInUse
#define hipErrorContextIsDestroyed       gpuErrorContextIsDestroyed
#define hipErrorCooperativeLaunchTooLarge  \
        gpuErrorCooperativeLaunchTooLarge
#define hipErrorDeinitialized            gpuErrorDeinitialized
#define hipErrorECCNotCorrectable        gpuErrorECCNotCorrectable
#define hipErrorFileNotFound             gpuErrorFileNotFound
#define hipErrorGraphExecUpdateFailure   gpuErrorGraphExecUpdateFailure
#define hipErrorHostMemoryAlreadyRegistered  \
        gpuErrorHostMemoryAlreadyRegistered
#define hipErrorHostMemoryNotRegistered  gpuErrorHostMemoryNotRegistered
#define hipErrorIllegalAddress           gpuErrorIllegalAddress
#define hipErrorIllegalState             gpuErrorIllegalState
#define hipErrorInsufficientDriver       gpuErrorInsufficientDriver
#define hipErrorInvalidConfiguration     gpuErrorInvalidConfiguration
#define hipErrorInvalidContext           gpuErrorInvalidContext
#define hipErrorInvalidDevice            gpuErrorInvalidDevice
#define hipErrorInvalidDeviceFunction    gpuErrorInvalidDeviceFunction
#define hipErrorInvalidDevicePointer     gpuErrorInvalidDevicePointer
#define hipErrorInvalidGraphicsContext   gpuErrorInvalidGraphicsContext
#define hipErrorInvalidHandle            gpuErrorInvalidHandle
#define hipErrorInvalidImage             gpuErrorInvalidImage
#define hipErrorInvalidKernelFile        gpuErrorInvalidKernelFile
#define hipErrorInvalidMemcpyDirection   gpuErrorInvalidMemcpyDirection
#define hipErrorInvalidPitchValue        gpuErrorInvalidPitchValue
#define hipErrorInvalidSource            gpuErrorInvalidSource
#define hipErrorInvalidSymbol            gpuErrorInvalidSymbol
#define hipErrorInvalidValue             gpuErrorInvalidValue
#define hipErrorLaunchFailure            gpuErrorLaunchFailure
#define hipErrorLaunchOutOfResources     gpuErrorLaunchOutOfResources
#define hipErrorLaunchTimeOut            gpuErrorLaunchTimeOut
#define hipErrorMapFailed                gpuErrorMapFailed
#define hipErrorMissingConfiguration     gpuErrorMissingConfiguration
#define hipErrorNoBinaryForGpu           gpuErrorNoBinaryForGpu
#define hipErrorNoDevice                 gpuErrorNoDevice
#define hipErrorNotFound                 gpuErrorNotFound
#define hipErrorNotInitialized           gpuErrorNotInitialized
#define hipErrorNotMapped                gpuErrorNotMapped
#define hipErrorNotMappedAsArray         gpuErrorNotMappedAsArray
#define hipErrorNotMappedAsPointer       gpuErrorNotMappedAsPointer
#define hipErrorNotReady                 gpuErrorNotReady
#define hipErrorNotSupported             gpuErrorNotSupported
#define hipErrorOperatingSystem          gpuErrorOperatingSystem
#define hipErrorOutOfMemory              gpuErrorOutOfMemory
#define hipErrorPeerAccessAlreadyEnabled gpuErrorPeerAccessAlreadyEnabled
#define hipErrorPeerAccessNotEnabled     gpuErrorPeerAccessNotEnabled
#define hipErrorPeerAccessUnsupported    gpuErrorPeerAccessUnsupported
#define hipErrorPriorLaunchFailure       gpuErrorPriorLaunchFailure
#define hipErrorProfilerAlreadyStarted   gpuErrorProfilerAlreadyStarted
#define hipErrorProfilerAlreadyStopped   gpuErrorProfilerAlreadyStopped
#define hipErrorProfilerDisabled         gpuErrorProfilerDisabled
#define hipErrorProfilerNotInitialized   gpuErrorProfilerNotInitialized
#define hipErrorSetOnActiveProcess       gpuErrorSetOnActiveProcess
#define hipErrorSharedObjectInitFailed   gpuErrorSharedObjectInitFailed
#define hipErrorSharedObjectSymbolNotFound  \
        gpuErrorSharedObjectSymbolNotFound
#define hipErrorStreamCaptureImplicit    gpuErrorStreamCaptureImplicit
#define hipErrorStreamCaptureInvalidated gpuErrorStreamCaptureInvalidated
#define hipErrorStreamCaptureIsolation   gpuErrorStreamCaptureIsolation
#define hipErrorStreamCaptureMerge       gpuErrorStreamCaptureMerge
#define hipErrorStreamCaptureUnjoined    gpuErrorStreamCaptureUnjoined
#define hipErrorStreamCaptureUnmatched   gpuErrorStreamCaptureUnmatched
#define hipErrorStreamCaptureUnsupported gpuErrorStreamCaptureUnsupported
#define hipErrorStreamCaptureWrongThread gpuErrorStreamCaptureWrongThread
#define hipErrorUnknown                  gpuErrorUnknown
#define hipErrorUnmapFailed              gpuErrorUnmapFailed
#define hipErrorUnsupportedLimit         gpuErrorUnsupportedLimit
#define hipError_t                       gpuError_t
#define hipEventBlockingSync             gpuEventBlockingSync
#define hipEventCreate                   gpuEventCreate
#define hipEventCreateWithFlags          gpuEventCreateWithFlags
#define hipEventDefault                  gpuEventDefault
#define hipEventDestroy                  gpuEventDestroy
#define hipEventDisableTiming            gpuEventDisableTiming
#define hipEventElapsedTime              gpuEventElapsedTime
#define hipEventInterprocess             gpuEventInterprocess
#define hipEventQuery                    gpuEventQuery
#define hipEventRecord                   gpuEventRecord
#define hipEventSynchronize              gpuEventSynchronize
#define hipEvent_t                       gpuEvent_t
#define hipExternalMemoryBufferDesc      gpuExternalMemoryBufferDesc
#define hipExternalMemoryBufferDesc_st   gpuExternalMemoryBufferDesc_st
#define hipExternalMemoryDedicated       gpuExternalMemoryDedicated
#define hipExternalMemoryGetMappedBuffer gpuExternalMemoryGetMappedBuffer
#define hipExternalMemoryHandleDesc      gpuExternalMemoryHandleDesc
#define hipExternalMemoryHandleDesc_st   gpuExternalMemoryHandleDesc_st
#define hipExternalMemoryHandleType      gpuExternalMemoryHandleType
#define hipExternalMemoryHandleTypeD3D11Resource  \
        gpuExternalMemoryHandleTypeD3D11Resource
#define hipExternalMemoryHandleTypeD3D11ResourceKmt  \
        gpuExternalMemoryHandleTypeD3D11ResourceKmt
#define hipExternalMemoryHandleTypeD3D12Heap  \
        gpuExternalMemoryHandleTypeD3D12Heap
#define hipExternalMemoryHandleTypeD3D12Resource  \
        gpuExternalMemoryHandleTypeD3D12Resource
#define hipExternalMemoryHandleTypeOpaqueFd  \
        gpuExternalMemoryHandleTypeOpaqueFd
#define hipExternalMemoryHandleTypeOpaqueWin32  \
        gpuExternalMemoryHandleTypeOpaqueWin32
#define hipExternalMemoryHandleTypeOpaqueWin32Kmt  \
        gpuExternalMemoryHandleTypeOpaqueWin32Kmt
#define hipExternalMemoryHandleType_enum gpuExternalMemoryHandleType_enum
#define hipExternalMemory_t              gpuExternalMemory_t
#define hipExternalSemaphoreHandleDesc   gpuExternalSemaphoreHandleDesc
#define hipExternalSemaphoreHandleDesc_st  \
        gpuExternalSemaphoreHandleDesc_st
#define hipExternalSemaphoreHandleType   gpuExternalSemaphoreHandleType
#define hipExternalSemaphoreHandleTypeD3D12Fence  \
        gpuExternalSemaphoreHandleTypeD3D12Fence
#define hipExternalSemaphoreHandleTypeOpaqueFd  \
        gpuExternalSemaphoreHandleTypeOpaqueFd
#define hipExternalSemaphoreHandleTypeOpaqueWin32  \
        gpuExternalSemaphoreHandleTypeOpaqueWin32
#define hipExternalSemaphoreHandleTypeOpaqueWin32Kmt  \
        gpuExternalSemaphoreHandleTypeOpaqueWin32Kmt
#define hipExternalSemaphoreHandleType_enum  \
        gpuExternalSemaphoreHandleType_enum
#define hipExternalSemaphoreSignalParams gpuExternalSemaphoreSignalParams
#define hipExternalSemaphoreSignalParams_st  \
        gpuExternalSemaphoreSignalParams_st
#define hipExternalSemaphoreWaitParams   gpuExternalSemaphoreWaitParams
#define hipExternalSemaphoreWaitParams_st  \
        gpuExternalSemaphoreWaitParams_st
#define hipExternalSemaphore_t           gpuExternalSemaphore_t
#define hipFree                          gpuFree
#define hipFreeArray                     gpuFreeArray
#define hipFreeAsync                     gpuFreeAsync
#define hipFreeMipmappedArray            gpuFreeMipmappedArray
#define hipFuncAttribute                 gpuFuncAttribute
#define hipFuncAttributeMax              gpuFuncAttributeMax
#define hipFuncAttributeMaxDynamicSharedMemorySize  \
        gpuFuncAttributeMaxDynamicSharedMemorySize
#define hipFuncAttributePreferredSharedMemoryCarveout  \
        gpuFuncAttributePreferredSharedMemoryCarveout
#define hipFuncAttributes                gpuFuncAttributes
#define hipFuncCachePreferEqual          gpuFuncCachePreferEqual
#define hipFuncCachePreferL1             gpuFuncCachePreferL1
#define hipFuncCachePreferNone           gpuFuncCachePreferNone
#define hipFuncCachePreferShared         gpuFuncCachePreferShared
#define hipFuncCache_t                   gpuFuncCache_t
#define hipFuncGetAttribute              gpuFuncGetAttribute
#define hipFuncGetAttributes             gpuFuncGetAttributes
#define hipFuncSetAttribute              gpuFuncSetAttribute
#define hipFuncSetCacheConfig            gpuFuncSetCacheConfig
#define hipFuncSetSharedMemConfig        gpuFuncSetSharedMemConfig
#define hipFunctionLaunchParams          gpuFunctionLaunchParams
#define hipFunctionLaunchParams_t        gpuFunctionLaunchParams_t
#define hipFunction_t                    gpuFunction_t
#define hipGLDeviceList                  gpuGLDeviceList
#define hipGLDeviceListAll               gpuGLDeviceListAll
#define hipGLDeviceListCurrentFrame      gpuGLDeviceListCurrentFrame
#define hipGLDeviceListNextFrame         gpuGLDeviceListNextFrame
#define hipGLGetDevices                  gpuGLGetDevices
#define hipGetChannelDesc                gpuGetChannelDesc
#define hipGetDevice                     gpuGetDevice
#define hipGetDeviceCount                gpuGetDeviceCount
#define hipGetDeviceFlags                gpuGetDeviceFlags
#define hipGetDeviceProperties           gpuGetDeviceProperties
#define hipGetErrorName                  gpuGetErrorName
#define hipGetErrorString                gpuGetErrorString
#define hipGetLastError                  gpuGetLastError
#define hipGetMipmappedArrayLevel        gpuGetMipmappedArrayLevel
#define hipGetSymbolAddress              gpuGetSymbolAddress
#define hipGetSymbolSize                 gpuGetSymbolSize
#define hipGetTextureAlignmentOffset     gpuGetTextureAlignmentOffset
#define hipGetTextureObjectResourceDesc  gpuGetTextureObjectResourceDesc
#define hipGetTextureObjectResourceViewDesc  \
        gpuGetTextureObjectResourceViewDesc
#define hipGetTextureObjectTextureDesc   gpuGetTextureObjectTextureDesc
#define hipGetTextureReference           gpuGetTextureReference
#define hipGraphAddChildGraphNode        gpuGraphAddChildGraphNode
#define hipGraphAddDependencies          gpuGraphAddDependencies
#define hipGraphAddEmptyNode             gpuGraphAddEmptyNode
#define hipGraphAddEventRecordNode       gpuGraphAddEventRecordNode
#define hipGraphAddEventWaitNode         gpuGraphAddEventWaitNode
#define hipGraphAddHostNode              gpuGraphAddHostNode
#define hipGraphAddKernelNode            gpuGraphAddKernelNode
#define hipGraphAddMemAllocNode          gpuGraphAddMemAllocNode
#define hipGraphAddMemFreeNode           gpuGraphAddMemFreeNode
#define hipGraphAddMemcpyNode            gpuGraphAddMemcpyNode
#define hipGraphAddMemcpyNode1D          gpuGraphAddMemcpyNode1D
#define hipGraphAddMemcpyNodeFromSymbol  gpuGraphAddMemcpyNodeFromSymbol
#define hipGraphAddMemcpyNodeToSymbol    gpuGraphAddMemcpyNodeToSymbol
#define hipGraphAddMemsetNode            gpuGraphAddMemsetNode
#define hipGraphChildGraphNodeGetGraph   gpuGraphChildGraphNodeGetGraph
#define hipGraphClone                    gpuGraphClone
#define hipGraphCreate                   gpuGraphCreate
#define hipGraphDebugDotFlags            gpuGraphDebugDotFlags
#define hipGraphDebugDotFlagsEventNodeParams  \
        gpuGraphDebugDotFlagsEventNodeParams
#define hipGraphDebugDotFlagsExtSemasSignalNodeParams  \
        gpuGraphDebugDotFlagsExtSemasSignalNodeParams
#define hipGraphDebugDotFlagsExtSemasWaitNodeParams  \
        gpuGraphDebugDotFlagsExtSemasWaitNodeParams
#define hipGraphDebugDotFlagsHandles     gpuGraphDebugDotFlagsHandles
#define hipGraphDebugDotFlagsHostNodeParams  \
        gpuGraphDebugDotFlagsHostNodeParams
#define hipGraphDebugDotFlagsKernelNodeAttributes  \
        gpuGraphDebugDotFlagsKernelNodeAttributes
#define hipGraphDebugDotFlagsKernelNodeParams  \
        gpuGraphDebugDotFlagsKernelNodeParams
#define hipGraphDebugDotFlagsMemcpyNodeParams  \
        gpuGraphDebugDotFlagsMemcpyNodeParams
#define hipGraphDebugDotFlagsMemsetNodeParams  \
        gpuGraphDebugDotFlagsMemsetNodeParams
#define hipGraphDebugDotFlagsVerbose     gpuGraphDebugDotFlagsVerbose
#define hipGraphDebugDotPrint            gpuGraphDebugDotPrint
#define hipGraphDestroy                  gpuGraphDestroy
#define hipGraphDestroyNode              gpuGraphDestroyNode
#define hipGraphEventRecordNodeGetEvent  gpuGraphEventRecordNodeGetEvent
#define hipGraphEventRecordNodeSetEvent  gpuGraphEventRecordNodeSetEvent
#define hipGraphEventWaitNodeGetEvent    gpuGraphEventWaitNodeGetEvent
#define hipGraphEventWaitNodeSetEvent    gpuGraphEventWaitNodeSetEvent
#define hipGraphExecChildGraphNodeSetParams  \
        gpuGraphExecChildGraphNodeSetParams
#define hipGraphExecDestroy              gpuGraphExecDestroy
#define hipGraphExecEventRecordNodeSetEvent  \
        gpuGraphExecEventRecordNodeSetEvent
#define hipGraphExecEventWaitNodeSetEvent  \
        gpuGraphExecEventWaitNodeSetEvent
#define hipGraphExecHostNodeSetParams    gpuGraphExecHostNodeSetParams
#define hipGraphExecKernelNodeSetParams  gpuGraphExecKernelNodeSetParams
#define hipGraphExecMemcpyNodeSetParams  gpuGraphExecMemcpyNodeSetParams
#define hipGraphExecMemcpyNodeSetParams1D  \
        gpuGraphExecMemcpyNodeSetParams1D
#define hipGraphExecMemcpyNodeSetParamsFromSymbol  \
        gpuGraphExecMemcpyNodeSetParamsFromSymbol
#define hipGraphExecMemcpyNodeSetParamsToSymbol  \
        gpuGraphExecMemcpyNodeSetParamsToSymbol
#define hipGraphExecMemsetNodeSetParams  gpuGraphExecMemsetNodeSetParams
#define hipGraphExecUpdate               gpuGraphExecUpdate
#define hipGraphExecUpdateError          gpuGraphExecUpdateError
#define hipGraphExecUpdateErrorFunctionChanged  \
        gpuGraphExecUpdateErrorFunctionChanged
#define hipGraphExecUpdateErrorNodeTypeChanged  \
        gpuGraphExecUpdateErrorNodeTypeChanged
#define hipGraphExecUpdateErrorNotSupported  \
        gpuGraphExecUpdateErrorNotSupported
#define hipGraphExecUpdateErrorParametersChanged  \
        gpuGraphExecUpdateErrorParametersChanged
#define hipGraphExecUpdateErrorTopologyChanged  \
        gpuGraphExecUpdateErrorTopologyChanged
#define hipGraphExecUpdateErrorUnsupportedFunctionChange  \
        gpuGraphExecUpdateErrorUnsupportedFunctionChange
#define hipGraphExecUpdateResult         gpuGraphExecUpdateResult
#define hipGraphExecUpdateSuccess        gpuGraphExecUpdateSuccess
#define hipGraphExec_t                   gpuGraphExec_t
#define hipGraphGetEdges                 gpuGraphGetEdges
#define hipGraphGetNodes                 gpuGraphGetNodes
#define hipGraphGetRootNodes             gpuGraphGetRootNodes
#define hipGraphHostNodeGetParams        gpuGraphHostNodeGetParams
#define hipGraphHostNodeSetParams        gpuGraphHostNodeSetParams
#define hipGraphInstantiate              gpuGraphInstantiate
#define hipGraphInstantiateFlagAutoFreeOnLaunch  \
        gpuGraphInstantiateFlagAutoFreeOnLaunch
#define hipGraphInstantiateFlagDeviceLaunch  \
        gpuGraphInstantiateFlagDeviceLaunch
#define hipGraphInstantiateFlagUpload    gpuGraphInstantiateFlagUpload
#define hipGraphInstantiateFlagUseNodePriority  \
        gpuGraphInstantiateFlagUseNodePriority
#define hipGraphInstantiateFlags         gpuGraphInstantiateFlags
#define hipGraphInstantiateWithFlags     gpuGraphInstantiateWithFlags
#define hipGraphKernelNodeCopyAttributes gpuGraphKernelNodeCopyAttributes
#define hipGraphKernelNodeGetAttribute   gpuGraphKernelNodeGetAttribute
#define hipGraphKernelNodeGetParams      gpuGraphKernelNodeGetParams
#define hipGraphKernelNodeSetAttribute   gpuGraphKernelNodeSetAttribute
#define hipGraphKernelNodeSetParams      gpuGraphKernelNodeSetParams
#define hipGraphLaunch                   gpuGraphLaunch
#define hipGraphMemAllocNodeGetParams    gpuGraphMemAllocNodeGetParams
#define hipGraphMemAttrReservedMemCurrent  \
        gpuGraphMemAttrReservedMemCurrent
#define hipGraphMemAttrReservedMemHigh   gpuGraphMemAttrReservedMemHigh
#define hipGraphMemAttrUsedMemCurrent    gpuGraphMemAttrUsedMemCurrent
#define hipGraphMemAttrUsedMemHigh       gpuGraphMemAttrUsedMemHigh
#define hipGraphMemAttributeType         gpuGraphMemAttributeType
#define hipGraphMemFreeNodeGetParams     gpuGraphMemFreeNodeGetParams
#define hipGraphMemcpyNodeGetParams      gpuGraphMemcpyNodeGetParams
#define hipGraphMemcpyNodeSetParams      gpuGraphMemcpyNodeSetParams
#define hipGraphMemcpyNodeSetParams1D    gpuGraphMemcpyNodeSetParams1D
#define hipGraphMemcpyNodeSetParamsFromSymbol  \
        gpuGraphMemcpyNodeSetParamsFromSymbol
#define hipGraphMemcpyNodeSetParamsToSymbol  \
        gpuGraphMemcpyNodeSetParamsToSymbol
#define hipGraphMemsetNodeGetParams      gpuGraphMemsetNodeGetParams
#define hipGraphMemsetNodeSetParams      gpuGraphMemsetNodeSetParams
#define hipGraphNodeFindInClone          gpuGraphNodeFindInClone
#define hipGraphNodeGetDependencies      gpuGraphNodeGetDependencies
#define hipGraphNodeGetDependentNodes    gpuGraphNodeGetDependentNodes
#define hipGraphNodeGetEnabled           gpuGraphNodeGetEnabled
#define hipGraphNodeGetType              gpuGraphNodeGetType
#define hipGraphNodeSetEnabled           gpuGraphNodeSetEnabled
#define hipGraphNodeType                 gpuGraphNodeType
#define hipGraphNodeTypeCount            gpuGraphNodeTypeCount
#define hipGraphNodeTypeEmpty            gpuGraphNodeTypeEmpty
#define hipGraphNodeTypeEventRecord      gpuGraphNodeTypeEventRecord
#define hipGraphNodeTypeExtSemaphoreSignal  \
        gpuGraphNodeTypeExtSemaphoreSignal
#define hipGraphNodeTypeExtSemaphoreWait gpuGraphNodeTypeExtSemaphoreWait
#define hipGraphNodeTypeGraph            gpuGraphNodeTypeGraph
#define hipGraphNodeTypeHost             gpuGraphNodeTypeHost
#define hipGraphNodeTypeKernel           gpuGraphNodeTypeKernel
#define hipGraphNodeTypeMemAlloc         gpuGraphNodeTypeMemAlloc
#define hipGraphNodeTypeMemFree          gpuGraphNodeTypeMemFree
#define hipGraphNodeTypeMemcpy           gpuGraphNodeTypeMemcpy
#define hipGraphNodeTypeMemset           gpuGraphNodeTypeMemset
#define hipGraphNodeTypeWaitEvent        gpuGraphNodeTypeWaitEvent
#define hipGraphNode_t                   gpuGraphNode_t
#define hipGraphReleaseUserObject        gpuGraphReleaseUserObject
#define hipGraphRemoveDependencies       gpuGraphRemoveDependencies
#define hipGraphRetainUserObject         gpuGraphRetainUserObject
#define hipGraphUpload                   gpuGraphUpload
#define hipGraphUserObjectMove           gpuGraphUserObjectMove
#define hipGraph_t                       gpuGraph_t
#define hipGraphicsGLRegisterBuffer      gpuGraphicsGLRegisterBuffer
#define hipGraphicsGLRegisterImage       gpuGraphicsGLRegisterImage
#define hipGraphicsMapResources          gpuGraphicsMapResources
#define hipGraphicsRegisterFlags         gpuGraphicsRegisterFlags
#define hipGraphicsRegisterFlagsNone     gpuGraphicsRegisterFlagsNone
#define hipGraphicsRegisterFlagsReadOnly gpuGraphicsRegisterFlagsReadOnly
#define hipGraphicsRegisterFlagsSurfaceLoadStore  \
        gpuGraphicsRegisterFlagsSurfaceLoadStore
#define hipGraphicsRegisterFlagsTextureGather  \
        gpuGraphicsRegisterFlagsTextureGather
#define hipGraphicsRegisterFlagsWriteDiscard  \
        gpuGraphicsRegisterFlagsWriteDiscard
#define hipGraphicsResource              gpuGraphicsResource
#define hipGraphicsResourceGetMappedPointer  \
        gpuGraphicsResourceGetMappedPointer
#define hipGraphicsResource_t            gpuGraphicsResource_t
#define hipGraphicsSubResourceGetMappedArray  \
        gpuGraphicsSubResourceGetMappedArray
#define hipGraphicsUnmapResources        gpuGraphicsUnmapResources
#define hipGraphicsUnregisterResource    gpuGraphicsUnregisterResource
#define hipHostAlloc                     gpuHostAlloc
#define hipHostFn_t                      gpuHostFn_t
#define hipHostFree                      gpuHostFree
#define hipHostGetDevicePointer          gpuHostGetDevicePointer
#define hipHostGetFlags                  gpuHostGetFlags
#define hipHostMalloc                    gpuHostMalloc
#define hipHostMallocDefault             gpuHostMallocDefault
#define hipHostMallocMapped              gpuHostMallocMapped
#define hipHostMallocPortable            gpuHostMallocPortable
#define hipHostMallocWriteCombined       gpuHostMallocWriteCombined
#define hipHostNodeParams                gpuHostNodeParams
#define hipHostRegister                  gpuHostRegister
#define hipHostRegisterDefault           gpuHostRegisterDefault
#define hipHostRegisterIoMemory          gpuHostRegisterIoMemory
#define hipHostRegisterMapped            gpuHostRegisterMapped
#define hipHostRegisterPortable          gpuHostRegisterPortable
#define hipHostRegisterReadOnly          gpuHostRegisterReadOnly
#define hipHostUnregister                gpuHostUnregister
#define hipImportExternalMemory          gpuImportExternalMemory
#define hipImportExternalSemaphore       gpuImportExternalSemaphore
#define hipInit                          gpuInit
#define hipInvalidDeviceId               gpuInvalidDeviceId
#define hipIpcCloseMemHandle             gpuIpcCloseMemHandle
#define hipIpcEventHandle_st             gpuIpcEventHandle_st
#define hipIpcEventHandle_t              gpuIpcEventHandle_t
#define hipIpcGetEventHandle             gpuIpcGetEventHandle
#define hipIpcGetMemHandle               gpuIpcGetMemHandle
#define hipIpcMemHandle_st               gpuIpcMemHandle_st
#define hipIpcMemHandle_t                gpuIpcMemHandle_t
#define hipIpcMemLazyEnablePeerAccess    gpuIpcMemLazyEnablePeerAccess
#define hipIpcOpenEventHandle            gpuIpcOpenEventHandle
#define hipIpcOpenMemHandle              gpuIpcOpenMemHandle
#define hipJitOption                     gpuJitOption
#define hipKernelNodeAttrID              gpuKernelNodeAttrID
#define hipKernelNodeAttrValue           gpuKernelNodeAttrValue
#define hipKernelNodeAttributeAccessPolicyWindow  \
        gpuKernelNodeAttributeAccessPolicyWindow
#define hipKernelNodeAttributeCooperative  \
        gpuKernelNodeAttributeCooperative
#define hipKernelNodeParams              gpuKernelNodeParams
#define hipLaunchCooperativeKernel       gpuLaunchCooperativeKernel
#define hipLaunchCooperativeKernelMultiDevice  \
        gpuLaunchCooperativeKernelMultiDevice
#define hipLaunchHostFunc                gpuLaunchHostFunc
#define hipLaunchKernel                  gpuLaunchKernel
#define hipLaunchParams                  gpuLaunchParams
#define hipLimitMallocHeapSize           gpuLimitMallocHeapSize
#define hipLimitPrintfFifoSize           gpuLimitPrintfFifoSize
#define hipLimitStackSize                gpuLimitStackSize
#define hipLimit_t                       gpuLimit_t
#define hipMalloc                        gpuMalloc
#define hipMalloc3D                      gpuMalloc3D
#define hipMalloc3DArray                 gpuMalloc3DArray
#define hipMallocArray                   gpuMallocArray
#define hipMallocAsync                   gpuMallocAsync
#define hipMallocFromPoolAsync           gpuMallocFromPoolAsync
#define hipMallocManaged                 gpuMallocManaged
#define hipMallocMipmappedArray          gpuMallocMipmappedArray
#define hipMallocPitch                   gpuMallocPitch
#define hipMemAccessDesc                 gpuMemAccessDesc
#define hipMemAccessFlags                gpuMemAccessFlags
#define hipMemAccessFlagsProtNone        gpuMemAccessFlagsProtNone
#define hipMemAccessFlagsProtRead        gpuMemAccessFlagsProtRead
#define hipMemAccessFlagsProtReadWrite   gpuMemAccessFlagsProtReadWrite
#define hipMemAddressFree                gpuMemAddressFree
#define hipMemAddressReserve             gpuMemAddressReserve
#define hipMemAdvise                     gpuMemAdvise
#define hipMemAdviseSetAccessedBy        gpuMemAdviseSetAccessedBy
#define hipMemAdviseSetPreferredLocation gpuMemAdviseSetPreferredLocation
#define hipMemAdviseSetReadMostly        gpuMemAdviseSetReadMostly
#define hipMemAdviseUnsetAccessedBy      gpuMemAdviseUnsetAccessedBy
#define hipMemAdviseUnsetPreferredLocation  \
        gpuMemAdviseUnsetPreferredLocation
#define hipMemAdviseUnsetReadMostly      gpuMemAdviseUnsetReadMostly
#define hipMemAllocHost                  gpuMemAllocHost
#define hipMemAllocNodeParams            gpuMemAllocNodeParams
#define hipMemAllocPitch                 gpuMemAllocPitch
#define hipMemAllocationGranularityMinimum  \
        gpuMemAllocationGranularityMinimum
#define hipMemAllocationGranularityRecommended  \
        gpuMemAllocationGranularityRecommended
#define hipMemAllocationGranularity_flags  \
        gpuMemAllocationGranularity_flags
#define hipMemAllocationHandleType       gpuMemAllocationHandleType
#define hipMemAllocationProp             gpuMemAllocationProp
#define hipMemAllocationType             gpuMemAllocationType
#define hipMemAllocationTypeInvalid      gpuMemAllocationTypeInvalid
#define hipMemAllocationTypeMax          gpuMemAllocationTypeMax
#define hipMemAllocationTypePinned       gpuMemAllocationTypePinned
#define hipMemAttachGlobal               gpuMemAttachGlobal
#define hipMemAttachHost                 gpuMemAttachHost
#define hipMemAttachSingle               gpuMemAttachSingle
#define hipMemCreate                     gpuMemCreate
#define hipMemExportToShareableHandle    gpuMemExportToShareableHandle
#define hipMemGenericAllocationHandle_t  gpuMemGenericAllocationHandle_t
#define hipMemGetAccess                  gpuMemGetAccess
#define hipMemGetAddressRange            gpuMemGetAddressRange
#define hipMemGetAllocationGranularity   gpuMemGetAllocationGranularity
#define hipMemGetAllocationPropertiesFromHandle  \
        gpuMemGetAllocationPropertiesFromHandle
#define hipMemGetInfo                    gpuMemGetInfo
#define hipMemHandleType                 gpuMemHandleType
#define hipMemHandleTypeGeneric          gpuMemHandleTypeGeneric
#define hipMemHandleTypeNone             gpuMemHandleTypeNone
#define hipMemHandleTypePosixFileDescriptor  \
        gpuMemHandleTypePosixFileDescriptor
#define hipMemHandleTypeWin32            gpuMemHandleTypeWin32
#define hipMemHandleTypeWin32Kmt         gpuMemHandleTypeWin32Kmt
#define hipMemImportFromShareableHandle  gpuMemImportFromShareableHandle
#define hipMemLocation                   gpuMemLocation
#define hipMemLocationType               gpuMemLocationType
#define hipMemLocationTypeDevice         gpuMemLocationTypeDevice
#define hipMemLocationTypeInvalid        gpuMemLocationTypeInvalid
#define hipMemMap                        gpuMemMap
#define hipMemMapArrayAsync              gpuMemMapArrayAsync
#define hipMemOperationType              gpuMemOperationType
#define hipMemOperationTypeMap           gpuMemOperationTypeMap
#define hipMemOperationTypeUnmap         gpuMemOperationTypeUnmap
#define hipMemPoolAttr                   gpuMemPoolAttr
#define hipMemPoolAttrReleaseThreshold   gpuMemPoolAttrReleaseThreshold
#define hipMemPoolAttrReservedMemCurrent gpuMemPoolAttrReservedMemCurrent
#define hipMemPoolAttrReservedMemHigh    gpuMemPoolAttrReservedMemHigh
#define hipMemPoolAttrUsedMemCurrent     gpuMemPoolAttrUsedMemCurrent
#define hipMemPoolAttrUsedMemHigh        gpuMemPoolAttrUsedMemHigh
#define hipMemPoolCreate                 gpuMemPoolCreate
#define hipMemPoolDestroy                gpuMemPoolDestroy
#define hipMemPoolExportPointer          gpuMemPoolExportPointer
#define hipMemPoolExportToShareableHandle  \
        gpuMemPoolExportToShareableHandle
#define hipMemPoolGetAccess              gpuMemPoolGetAccess
#define hipMemPoolGetAttribute           gpuMemPoolGetAttribute
#define hipMemPoolImportFromShareableHandle  \
        gpuMemPoolImportFromShareableHandle
#define hipMemPoolImportPointer          gpuMemPoolImportPointer
#define hipMemPoolProps                  gpuMemPoolProps
#define hipMemPoolPtrExportData          gpuMemPoolPtrExportData
#define hipMemPoolReuseAllowInternalDependencies  \
        gpuMemPoolReuseAllowInternalDependencies
#define hipMemPoolReuseAllowOpportunistic  \
        gpuMemPoolReuseAllowOpportunistic
#define hipMemPoolReuseFollowEventDependencies  \
        gpuMemPoolReuseFollowEventDependencies
#define hipMemPoolSetAccess              gpuMemPoolSetAccess
#define hipMemPoolSetAttribute           gpuMemPoolSetAttribute
#define hipMemPoolTrimTo                 gpuMemPoolTrimTo
#define hipMemPool_t                     gpuMemPool_t
#define hipMemPrefetchAsync              gpuMemPrefetchAsync
#define hipMemRangeAttribute             gpuMemRangeAttribute
#define hipMemRangeAttributeAccessedBy   gpuMemRangeAttributeAccessedBy
#define hipMemRangeAttributeLastPrefetchLocation  \
        gpuMemRangeAttributeLastPrefetchLocation
#define hipMemRangeAttributePreferredLocation  \
        gpuMemRangeAttributePreferredLocation
#define hipMemRangeAttributeReadMostly   gpuMemRangeAttributeReadMostly
#define hipMemRangeGetAttribute          gpuMemRangeGetAttribute
#define hipMemRangeGetAttributes         gpuMemRangeGetAttributes
#define hipMemRelease                    gpuMemRelease
#define hipMemRetainAllocationHandle     gpuMemRetainAllocationHandle
#define hipMemSetAccess                  gpuMemSetAccess
#define hipMemUnmap                      gpuMemUnmap
#define hipMemcpy                        gpuMemcpy
#define hipMemcpy2D                      gpuMemcpy2D
#define hipMemcpy2DAsync                 gpuMemcpy2DAsync
#define hipMemcpy2DFromArray             gpuMemcpy2DFromArray
#define hipMemcpy2DFromArrayAsync        gpuMemcpy2DFromArrayAsync
#define hipMemcpy2DToArray               gpuMemcpy2DToArray
#define hipMemcpy2DToArrayAsync          gpuMemcpy2DToArrayAsync
#define hipMemcpy3D                      gpuMemcpy3D
#define hipMemcpy3DAsync                 gpuMemcpy3DAsync
#define hipMemcpyAsync                   gpuMemcpyAsync
#define hipMemcpyAtoH                    gpuMemcpyAtoH
#define hipMemcpyDtoD                    gpuMemcpyDtoD
#define hipMemcpyDtoDAsync               gpuMemcpyDtoDAsync
#define hipMemcpyDtoH                    gpuMemcpyDtoH
#define hipMemcpyDtoHAsync               gpuMemcpyDtoHAsync
#define hipMemcpyFromArray               gpuMemcpyFromArray
#define hipMemcpyFromSymbol              gpuMemcpyFromSymbol
#define hipMemcpyFromSymbolAsync         gpuMemcpyFromSymbolAsync
#define hipMemcpyHtoA                    gpuMemcpyHtoA
#define hipMemcpyHtoD                    gpuMemcpyHtoD
#define hipMemcpyHtoDAsync               gpuMemcpyHtoDAsync
#define hipMemcpyParam2D                 gpuMemcpyParam2D
#define hipMemcpyParam2DAsync            gpuMemcpyParam2DAsync
#define hipMemcpyPeer                    gpuMemcpyPeer
#define hipMemcpyPeerAsync               gpuMemcpyPeerAsync
#define hipMemcpyToArray                 gpuMemcpyToArray
#define hipMemcpyToSymbol                gpuMemcpyToSymbol
#define hipMemcpyToSymbolAsync           gpuMemcpyToSymbolAsync
#define hipMemoryAdvise                  gpuMemoryAdvise
#define hipMemoryType                    gpuMemoryType
#define hipMemoryTypeArray               gpuMemoryTypeArray
#define hipMemoryTypeDevice              gpuMemoryTypeDevice
#define hipMemoryTypeHost                gpuMemoryTypeHost
#define hipMemoryTypeManaged             gpuMemoryTypeManaged
#define hipMemoryTypeUnified             gpuMemoryTypeUnified
#define hipMemset                        gpuMemset
#define hipMemset2D                      gpuMemset2D
#define hipMemset2DAsync                 gpuMemset2DAsync
#define hipMemset3D                      gpuMemset3D
#define hipMemset3DAsync                 gpuMemset3DAsync
#define hipMemsetAsync                   gpuMemsetAsync
#define hipMemsetD16                     gpuMemsetD16
#define hipMemsetD16Async                gpuMemsetD16Async
#define hipMemsetD32                     gpuMemsetD32
#define hipMemsetD32Async                gpuMemsetD32Async
#define hipMemsetD8                      gpuMemsetD8
#define hipMemsetD8Async                 gpuMemsetD8Async
#define hipMemsetParams                  gpuMemsetParams
#define hipMipmappedArrayCreate          gpuMipmappedArrayCreate
#define hipMipmappedArrayDestroy         gpuMipmappedArrayDestroy
#define hipMipmappedArrayGetLevel        gpuMipmappedArrayGetLevel
#define hipModuleGetFunction             gpuModuleGetFunction
#define hipModuleGetGlobal               gpuModuleGetGlobal
#define hipModuleGetTexRef               gpuModuleGetTexRef
#define hipModuleLaunchCooperativeKernel gpuModuleLaunchCooperativeKernel
#define hipModuleLaunchCooperativeKernelMultiDevice  \
        gpuModuleLaunchCooperativeKernelMultiDevice
#define hipModuleLaunchKernel            gpuModuleLaunchKernel
#define hipModuleLoad                    gpuModuleLoad
#define hipModuleLoadData                gpuModuleLoadData
#define hipModuleLoadDataEx              gpuModuleLoadDataEx
#define hipModuleOccupancyMaxActiveBlocksPerMultiprocessor  \
        gpuModuleOccupancyMaxActiveBlocksPerMultiprocessor
#define hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags  \
        gpuModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define hipModuleOccupancyMaxPotentialBlockSize  \
        gpuModuleOccupancyMaxPotentialBlockSize
#define hipModuleOccupancyMaxPotentialBlockSizeWithFlags  \
        gpuModuleOccupancyMaxPotentialBlockSizeWithFlags
#define hipModuleUnload                  gpuModuleUnload
#define hipModule_t                      gpuModule_t
#define hipOccupancyDefault              gpuOccupancyDefault
#define hipOccupancyDisableCachingOverride  \
        gpuOccupancyDisableCachingOverride
#define hipOccupancyMaxActiveBlocksPerMultiprocessor  \
        gpuOccupancyMaxActiveBlocksPerMultiprocessor
#define hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags  \
        gpuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define hipOccupancyMaxPotentialBlockSize  \
        gpuOccupancyMaxPotentialBlockSize
#define hipOccupancyMaxPotentialBlockSizeVariableSMem  \
        gpuOccupancyMaxPotentialBlockSizeVariableSMem
#define hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags  \
        gpuOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
#define hipOccupancyMaxPotentialBlockSizeWithFlags  \
        gpuOccupancyMaxPotentialBlockSizeWithFlags
#define hipPeekAtLastError               gpuPeekAtLastError
#define hipPointerAttribute_t            gpuPointerAttribute_t
#define hipPointerGetAttribute           gpuPointerGetAttribute
#define hipPointerGetAttributes          gpuPointerGetAttributes
#define hipPointerSetAttribute           gpuPointerSetAttribute
#define hipProfilerStart                 gpuProfilerStart
#define hipProfilerStop                  gpuProfilerStop
#define hipRuntimeGetVersion             gpuRuntimeGetVersion
#define hipSetDevice                     gpuSetDevice
#define hipSetDeviceFlags                gpuSetDeviceFlags
#define hipSharedMemBankSizeDefault      gpuSharedMemBankSizeDefault
#define hipSharedMemBankSizeEightByte    gpuSharedMemBankSizeEightByte
#define hipSharedMemBankSizeFourByte     gpuSharedMemBankSizeFourByte
#define hipSharedMemConfig               gpuSharedMemConfig
#define hipSignalExternalSemaphoresAsync gpuSignalExternalSemaphoresAsync
#define hipStreamAddCallback             gpuStreamAddCallback
#define hipStreamAddCaptureDependencies  gpuStreamAddCaptureDependencies
#define hipStreamAttachMemAsync          gpuStreamAttachMemAsync
#define hipStreamBeginCapture            gpuStreamBeginCapture
#define hipStreamCallback_t              gpuStreamCallback_t
#define hipStreamCaptureMode             gpuStreamCaptureMode
#define hipStreamCaptureModeGlobal       gpuStreamCaptureModeGlobal
#define hipStreamCaptureModeRelaxed      gpuStreamCaptureModeRelaxed
#define hipStreamCaptureModeThreadLocal  gpuStreamCaptureModeThreadLocal
#define hipStreamCaptureStatus           gpuStreamCaptureStatus
#define hipStreamCaptureStatusActive     gpuStreamCaptureStatusActive
#define hipStreamCaptureStatusInvalidated  \
        gpuStreamCaptureStatusInvalidated
#define hipStreamCaptureStatusNone       gpuStreamCaptureStatusNone
#define hipStreamCreate                  gpuStreamCreate
#define hipStreamCreateWithFlags         gpuStreamCreateWithFlags
#define hipStreamCreateWithPriority      gpuStreamCreateWithPriority
#define hipStreamDefault                 gpuStreamDefault
#define hipStreamDestroy                 gpuStreamDestroy
#define hipStreamEndCapture              gpuStreamEndCapture
#define hipStreamGetCaptureInfo          gpuStreamGetCaptureInfo
#define hipStreamGetCaptureInfo_v2       gpuStreamGetCaptureInfo_v2
#define hipStreamGetFlags                gpuStreamGetFlags
#define hipStreamGetPriority             gpuStreamGetPriority
#define hipStreamIsCapturing             gpuStreamIsCapturing
#define hipStreamNonBlocking             gpuStreamNonBlocking
#define hipStreamPerThread               gpuStreamPerThread
#define hipStreamQuery                   gpuStreamQuery
#define hipStreamSetCaptureDependencies  gpuStreamSetCaptureDependencies
#define hipStreamSynchronize             gpuStreamSynchronize
#define hipStreamUpdateCaptureDependencies  \
        gpuStreamUpdateCaptureDependencies
#define hipStreamUpdateCaptureDependenciesFlags  \
        gpuStreamUpdateCaptureDependenciesFlags
#define hipStreamWaitEvent               gpuStreamWaitEvent
#define hipStreamWaitValue32             gpuStreamWaitValue32
#define hipStreamWaitValue64             gpuStreamWaitValue64
#define hipStreamWaitValueAnd            gpuStreamWaitValueAnd
#define hipStreamWaitValueEq             gpuStreamWaitValueEq
#define hipStreamWaitValueGte            gpuStreamWaitValueGte
#define hipStreamWaitValueNor            gpuStreamWaitValueNor
#define hipStreamWriteValue32            gpuStreamWriteValue32
#define hipStreamWriteValue64            gpuStreamWriteValue64
#define hipStream_t                      gpuStream_t
#define hipSuccess                       gpuSuccess
#define hipTexObjectCreate               gpuTexObjectCreate
#define hipTexObjectDestroy              gpuTexObjectDestroy
#define hipTexObjectGetResourceDesc      gpuTexObjectGetResourceDesc
#define hipTexObjectGetResourceViewDesc  gpuTexObjectGetResourceViewDesc
#define hipTexObjectGetTextureDesc       gpuTexObjectGetTextureDesc
#define hipTexRefGetAddress              gpuTexRefGetAddress
#define hipTexRefGetAddressMode          gpuTexRefGetAddressMode
#define hipTexRefGetFilterMode           gpuTexRefGetFilterMode
#define hipTexRefGetFlags                gpuTexRefGetFlags
#define hipTexRefGetFormat               gpuTexRefGetFormat
#define hipTexRefGetMaxAnisotropy        gpuTexRefGetMaxAnisotropy
#define hipTexRefGetMipMappedArray       gpuTexRefGetMipMappedArray
#define hipTexRefGetMipmapFilterMode     gpuTexRefGetMipmapFilterMode
#define hipTexRefGetMipmapLevelBias      gpuTexRefGetMipmapLevelBias
#define hipTexRefGetMipmapLevelClamp     gpuTexRefGetMipmapLevelClamp
#define hipTexRefSetAddress              gpuTexRefSetAddress
#define hipTexRefSetAddress2D            gpuTexRefSetAddress2D
#define hipTexRefSetAddressMode          gpuTexRefSetAddressMode
#define hipTexRefSetArray                gpuTexRefSetArray
#define hipTexRefSetBorderColor          gpuTexRefSetBorderColor
#define hipTexRefSetFilterMode           gpuTexRefSetFilterMode
#define hipTexRefSetFlags                gpuTexRefSetFlags
#define hipTexRefSetFormat               gpuTexRefSetFormat
#define hipTexRefSetMaxAnisotropy        gpuTexRefSetMaxAnisotropy
#define hipTexRefSetMipmapFilterMode     gpuTexRefSetMipmapFilterMode
#define hipTexRefSetMipmapLevelBias      gpuTexRefSetMipmapLevelBias
#define hipTexRefSetMipmapLevelClamp     gpuTexRefSetMipmapLevelClamp
#define hipTexRefSetMipmappedArray       gpuTexRefSetMipmappedArray
#define hipThreadExchangeStreamCaptureMode  \
        gpuThreadExchangeStreamCaptureMode
#define hipUUID                          gpuUUID
#define hipUUID_t                        gpuUUID_t
#define hipUnbindTexture                 gpuUnbindTexture
#define hipUserObjectCreate              gpuUserObjectCreate
#define hipUserObjectFlags               gpuUserObjectFlags
#define hipUserObjectNoDestructorSync    gpuUserObjectNoDestructorSync
#define hipUserObjectRelease             gpuUserObjectRelease
#define hipUserObjectRetain              gpuUserObjectRetain
#define hipUserObjectRetainFlags         gpuUserObjectRetainFlags
#define hipUserObject_t                  gpuUserObject_t
#define hipWaitExternalSemaphoresAsync   gpuWaitExternalSemaphoresAsync

/* hip/channel_descriptor.h */
#define hipCreateChannelDesc             gpuCreateChannelDesc

/* hip/driver_types.h */
#define HIP_AD_FORMAT_FLOAT              GPU_AD_FORMAT_FLOAT
#define HIP_AD_FORMAT_HALF               GPU_AD_FORMAT_HALF
#define HIP_AD_FORMAT_SIGNED_INT16       GPU_AD_FORMAT_SIGNED_INT16
#define HIP_AD_FORMAT_SIGNED_INT32       GPU_AD_FORMAT_SIGNED_INT32
#define HIP_AD_FORMAT_SIGNED_INT8        GPU_AD_FORMAT_SIGNED_INT8
#define HIP_AD_FORMAT_UNSIGNED_INT16     GPU_AD_FORMAT_UNSIGNED_INT16
#define HIP_AD_FORMAT_UNSIGNED_INT32     GPU_AD_FORMAT_UNSIGNED_INT32
#define HIP_AD_FORMAT_UNSIGNED_INT8      GPU_AD_FORMAT_UNSIGNED_INT8
#define HIP_ARRAY3D_DESCRIPTOR           GPU_ARRAY3D_DESCRIPTOR
#define HIP_ARRAY_DESCRIPTOR             GPU_ARRAY_DESCRIPTOR
#define HIP_FUNC_ATTRIBUTE_BINARY_VERSION  \
        GPU_FUNC_ATTRIBUTE_BINARY_VERSION
#define HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA GPU_FUNC_ATTRIBUTE_CACHE_MODE_CA
#define HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES  \
        GPU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES  \
        GPU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_MAX           GPU_FUNC_ATTRIBUTE_MAX
#define HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES  \
        GPU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK  \
        GPU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define HIP_FUNC_ATTRIBUTE_NUM_REGS      GPU_FUNC_ATTRIBUTE_NUM_REGS
#define HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT  \
        GPU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
#define HIP_FUNC_ATTRIBUTE_PTX_VERSION   GPU_FUNC_ATTRIBUTE_PTX_VERSION
#define HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES  \
        GPU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define HIP_MEMCPY3D                     GPU_MEMCPY3D
#define HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS  \
        GPU_POINTER_ATTRIBUTE_ACCESS_FLAGS
#define HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES  \
        GPU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
#define HIP_POINTER_ATTRIBUTE_BUFFER_ID  GPU_POINTER_ATTRIBUTE_BUFFER_ID
#define HIP_POINTER_ATTRIBUTE_CONTEXT    GPU_POINTER_ATTRIBUTE_CONTEXT
#define HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL  \
        GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define HIP_POINTER_ATTRIBUTE_DEVICE_POINTER  \
        GPU_POINTER_ATTRIBUTE_DEVICE_POINTER
#define HIP_POINTER_ATTRIBUTE_HOST_POINTER  \
        GPU_POINTER_ATTRIBUTE_HOST_POINTER
#define HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE  \
        GPU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
#define HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE  \
        GPU_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
#define HIP_POINTER_ATTRIBUTE_IS_MANAGED GPU_POINTER_ATTRIBUTE_IS_MANAGED
#define HIP_POINTER_ATTRIBUTE_MAPPED     GPU_POINTER_ATTRIBUTE_MAPPED
#define HIP_POINTER_ATTRIBUTE_MEMORY_TYPE  \
        GPU_POINTER_ATTRIBUTE_MEMORY_TYPE
#define HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE  \
        GPU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#define HIP_POINTER_ATTRIBUTE_P2P_TOKENS GPU_POINTER_ATTRIBUTE_P2P_TOKENS
#define HIP_POINTER_ATTRIBUTE_RANGE_SIZE GPU_POINTER_ATTRIBUTE_RANGE_SIZE
#define HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR  \
        GPU_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS  \
        GPU_POINTER_ATTRIBUTE_SYNC_MEMOPS
#define HIP_RESOURCE_DESC                GPU_RESOURCE_DESC
#define HIP_RESOURCE_DESC_st             GPU_RESOURCE_DESC_st
#define HIP_RESOURCE_TYPE_ARRAY          GPU_RESOURCE_TYPE_ARRAY
#define HIP_RESOURCE_TYPE_LINEAR         GPU_RESOURCE_TYPE_LINEAR
#define HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY  \
        GPU_RESOURCE_TYPE_MIPMAPPED_ARRAY
#define HIP_RESOURCE_TYPE_PITCH2D        GPU_RESOURCE_TYPE_PITCH2D
#define HIP_RESOURCE_VIEW_DESC           GPU_RESOURCE_VIEW_DESC
#define HIP_RESOURCE_VIEW_DESC_st        GPU_RESOURCE_VIEW_DESC_st
#define HIP_RES_VIEW_FORMAT_FLOAT_1X16   GPU_RES_VIEW_FORMAT_FLOAT_1X16
#define HIP_RES_VIEW_FORMAT_FLOAT_1X32   GPU_RES_VIEW_FORMAT_FLOAT_1X32
#define HIP_RES_VIEW_FORMAT_FLOAT_2X16   GPU_RES_VIEW_FORMAT_FLOAT_2X16
#define HIP_RES_VIEW_FORMAT_FLOAT_2X32   GPU_RES_VIEW_FORMAT_FLOAT_2X32
#define HIP_RES_VIEW_FORMAT_FLOAT_4X16   GPU_RES_VIEW_FORMAT_FLOAT_4X16
#define HIP_RES_VIEW_FORMAT_FLOAT_4X32   GPU_RES_VIEW_FORMAT_FLOAT_4X32
#define HIP_RES_VIEW_FORMAT_NONE         GPU_RES_VIEW_FORMAT_NONE
#define HIP_RES_VIEW_FORMAT_SIGNED_BC4   GPU_RES_VIEW_FORMAT_SIGNED_BC4
#define HIP_RES_VIEW_FORMAT_SIGNED_BC5   GPU_RES_VIEW_FORMAT_SIGNED_BC5
#define HIP_RES_VIEW_FORMAT_SIGNED_BC6H  GPU_RES_VIEW_FORMAT_SIGNED_BC6H
#define HIP_RES_VIEW_FORMAT_SINT_1X16    GPU_RES_VIEW_FORMAT_SINT_1X16
#define HIP_RES_VIEW_FORMAT_SINT_1X32    GPU_RES_VIEW_FORMAT_SINT_1X32
#define HIP_RES_VIEW_FORMAT_SINT_1X8     GPU_RES_VIEW_FORMAT_SINT_1X8
#define HIP_RES_VIEW_FORMAT_SINT_2X16    GPU_RES_VIEW_FORMAT_SINT_2X16
#define HIP_RES_VIEW_FORMAT_SINT_2X32    GPU_RES_VIEW_FORMAT_SINT_2X32
#define HIP_RES_VIEW_FORMAT_SINT_2X8     GPU_RES_VIEW_FORMAT_SINT_2X8
#define HIP_RES_VIEW_FORMAT_SINT_4X16    GPU_RES_VIEW_FORMAT_SINT_4X16
#define HIP_RES_VIEW_FORMAT_SINT_4X32    GPU_RES_VIEW_FORMAT_SINT_4X32
#define HIP_RES_VIEW_FORMAT_SINT_4X8     GPU_RES_VIEW_FORMAT_SINT_4X8
#define HIP_RES_VIEW_FORMAT_UINT_1X16    GPU_RES_VIEW_FORMAT_UINT_1X16
#define HIP_RES_VIEW_FORMAT_UINT_1X32    GPU_RES_VIEW_FORMAT_UINT_1X32
#define HIP_RES_VIEW_FORMAT_UINT_1X8     GPU_RES_VIEW_FORMAT_UINT_1X8
#define HIP_RES_VIEW_FORMAT_UINT_2X16    GPU_RES_VIEW_FORMAT_UINT_2X16
#define HIP_RES_VIEW_FORMAT_UINT_2X32    GPU_RES_VIEW_FORMAT_UINT_2X32
#define HIP_RES_VIEW_FORMAT_UINT_2X8     GPU_RES_VIEW_FORMAT_UINT_2X8
#define HIP_RES_VIEW_FORMAT_UINT_4X16    GPU_RES_VIEW_FORMAT_UINT_4X16
#define HIP_RES_VIEW_FORMAT_UINT_4X32    GPU_RES_VIEW_FORMAT_UINT_4X32
#define HIP_RES_VIEW_FORMAT_UINT_4X8     GPU_RES_VIEW_FORMAT_UINT_4X8
#define HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 GPU_RES_VIEW_FORMAT_UNSIGNED_BC1
#define HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 GPU_RES_VIEW_FORMAT_UNSIGNED_BC2
#define HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 GPU_RES_VIEW_FORMAT_UNSIGNED_BC3
#define HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 GPU_RES_VIEW_FORMAT_UNSIGNED_BC4
#define HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 GPU_RES_VIEW_FORMAT_UNSIGNED_BC5
#define HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H  \
        GPU_RES_VIEW_FORMAT_UNSIGNED_BC6H
#define HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 GPU_RES_VIEW_FORMAT_UNSIGNED_BC7
#define HIP_TEXTURE_DESC                 GPU_TEXTURE_DESC
#define HIP_TEXTURE_DESC_st              GPU_TEXTURE_DESC_st
#define HIP_TRSA_OVERRIDE_FORMAT         GPU_TRSA_OVERRIDE_FORMAT
#define HIP_TRSF_NORMALIZED_COORDINATES  GPU_TRSF_NORMALIZED_COORDINATES
#define HIP_TRSF_READ_AS_INTEGER         GPU_TRSF_READ_AS_INTEGER
#define HIP_TRSF_SRGB                    GPU_TRSF_SRGB
#define HIP_TR_ADDRESS_MODE_BORDER       GPU_TR_ADDRESS_MODE_BORDER
#define HIP_TR_ADDRESS_MODE_CLAMP        GPU_TR_ADDRESS_MODE_CLAMP
#define HIP_TR_ADDRESS_MODE_MIRROR       GPU_TR_ADDRESS_MODE_MIRROR
#define HIP_TR_ADDRESS_MODE_WRAP         GPU_TR_ADDRESS_MODE_WRAP
#define HIP_TR_FILTER_MODE_LINEAR        GPU_TR_FILTER_MODE_LINEAR
#define HIP_TR_FILTER_MODE_POINT         GPU_TR_FILTER_MODE_POINT
#define HIPaddress_mode                  GPUaddress_mode
#define HIPaddress_mode_enum             GPUaddress_mode_enum
#define HIPfilter_mode                   GPUfilter_mode
#define HIPfilter_mode_enum              GPUfilter_mode_enum
#define HIPresourceViewFormat            GPUresourceViewFormat
#define HIPresourceViewFormat_enum       GPUresourceViewFormat_enum
#define HIPresourcetype                  GPUresourcetype
#define HIPresourcetype_enum             GPUresourcetype_enum
#define hipArray                         gpuArray
#define hipArray_Format                  gpuArray_Format
#define hipArray_const_t                 gpuArray_const_t
#define hipArray_t                       gpuArray_t
#define hipChannelFormatDesc             gpuChannelFormatDesc
#define hipChannelFormatKind             gpuChannelFormatKind
#define hipChannelFormatKindFloat        gpuChannelFormatKindFloat
#define hipChannelFormatKindNone         gpuChannelFormatKindNone
#define hipChannelFormatKindSigned       gpuChannelFormatKindSigned
#define hipChannelFormatKindUnsigned     gpuChannelFormatKindUnsigned
#define hipDeviceptr_t                   gpuDeviceptr_t
#define hipExtent                        gpuExtent
#define hipFunction_attribute            gpuFunction_attribute
#define hipMemcpy3DParms                 gpuMemcpy3DParms
#define hipMemcpyDefault                 gpuMemcpyDefault
#define hipMemcpyDeviceToDevice          gpuMemcpyDeviceToDevice
#define hipMemcpyDeviceToHost            gpuMemcpyDeviceToHost
#define hipMemcpyHostToDevice            gpuMemcpyHostToDevice
#define hipMemcpyHostToHost              gpuMemcpyHostToHost
#define hipMemcpyKind                    gpuMemcpyKind
#define hipMipmappedArray                gpuMipmappedArray
#define hipMipmappedArray_const_t        gpuMipmappedArray_const_t
#define hipMipmappedArray_t              gpuMipmappedArray_t
#define hipPitchedPtr                    gpuPitchedPtr
#define hipPointer_attribute             gpuPointer_attribute
#define hipPos                           gpuPos
#define hipResViewFormatFloat1           gpuResViewFormatFloat1
#define hipResViewFormatFloat2           gpuResViewFormatFloat2
#define hipResViewFormatFloat4           gpuResViewFormatFloat4
#define hipResViewFormatHalf1            gpuResViewFormatHalf1
#define hipResViewFormatHalf2            gpuResViewFormatHalf2
#define hipResViewFormatHalf4            gpuResViewFormatHalf4
#define hipResViewFormatNone             gpuResViewFormatNone
#define hipResViewFormatSignedBlockCompressed4  \
        gpuResViewFormatSignedBlockCompressed4
#define hipResViewFormatSignedBlockCompressed5  \
        gpuResViewFormatSignedBlockCompressed5
#define hipResViewFormatSignedBlockCompressed6H  \
        gpuResViewFormatSignedBlockCompressed6H
#define hipResViewFormatSignedChar1      gpuResViewFormatSignedChar1
#define hipResViewFormatSignedChar2      gpuResViewFormatSignedChar2
#define hipResViewFormatSignedChar4      gpuResViewFormatSignedChar4
#define hipResViewFormatSignedInt1       gpuResViewFormatSignedInt1
#define hipResViewFormatSignedInt2       gpuResViewFormatSignedInt2
#define hipResViewFormatSignedInt4       gpuResViewFormatSignedInt4
#define hipResViewFormatSignedShort1     gpuResViewFormatSignedShort1
#define hipResViewFormatSignedShort2     gpuResViewFormatSignedShort2
#define hipResViewFormatSignedShort4     gpuResViewFormatSignedShort4
#define hipResViewFormatUnsignedBlockCompressed1  \
        gpuResViewFormatUnsignedBlockCompressed1
#define hipResViewFormatUnsignedBlockCompressed2  \
        gpuResViewFormatUnsignedBlockCompressed2
#define hipResViewFormatUnsignedBlockCompressed3  \
        gpuResViewFormatUnsignedBlockCompressed3
#define hipResViewFormatUnsignedBlockCompressed4  \
        gpuResViewFormatUnsignedBlockCompressed4
#define hipResViewFormatUnsignedBlockCompressed5  \
        gpuResViewFormatUnsignedBlockCompressed5
#define hipResViewFormatUnsignedBlockCompressed6H  \
        gpuResViewFormatUnsignedBlockCompressed6H
#define hipResViewFormatUnsignedBlockCompressed7  \
        gpuResViewFormatUnsignedBlockCompressed7
#define hipResViewFormatUnsignedChar1    gpuResViewFormatUnsignedChar1
#define hipResViewFormatUnsignedChar2    gpuResViewFormatUnsignedChar2
#define hipResViewFormatUnsignedChar4    gpuResViewFormatUnsignedChar4
#define hipResViewFormatUnsignedInt1     gpuResViewFormatUnsignedInt1
#define hipResViewFormatUnsignedInt2     gpuResViewFormatUnsignedInt2
#define hipResViewFormatUnsignedInt4     gpuResViewFormatUnsignedInt4
#define hipResViewFormatUnsignedShort1   gpuResViewFormatUnsignedShort1
#define hipResViewFormatUnsignedShort2   gpuResViewFormatUnsignedShort2
#define hipResViewFormatUnsignedShort4   gpuResViewFormatUnsignedShort4
#define hipResourceDesc                  gpuResourceDesc
#define hipResourceType                  gpuResourceType
#define hipResourceTypeArray             gpuResourceTypeArray
#define hipResourceTypeLinear            gpuResourceTypeLinear
#define hipResourceTypeMipmappedArray    gpuResourceTypeMipmappedArray
#define hipResourceTypePitch2D           gpuResourceTypePitch2D
#define hipResourceViewDesc              gpuResourceViewDesc
#define hipResourceViewFormat            gpuResourceViewFormat
#define hip_Memcpy2D                     gpu_Memcpy2D
#define make_hipExtent                   make_gpuExtent
#define make_hipPitchedPtr               make_gpuPitchedPtr
#define make_hipPos                      make_gpuPos

/* hip/surface_types.h */
#define hipBoundaryModeClamp             gpuBoundaryModeClamp
#define hipBoundaryModeTrap              gpuBoundaryModeTrap
#define hipBoundaryModeZero              gpuBoundaryModeZero
#define hipSurfaceBoundaryMode           gpuSurfaceBoundaryMode
#define hipSurfaceObject_t               gpuSurfaceObject_t

/* hip/texture_types.h */
#define hipAddressModeBorder             gpuAddressModeBorder
#define hipAddressModeClamp              gpuAddressModeClamp
#define hipAddressModeMirror             gpuAddressModeMirror
#define hipAddressModeWrap               gpuAddressModeWrap
#define hipFilterModeLinear              gpuFilterModeLinear
#define hipFilterModePoint               gpuFilterModePoint
#define hipReadModeElementType           gpuReadModeElementType
#define hipReadModeNormalizedFloat       gpuReadModeNormalizedFloat
#define hipTexRef                        gpuTexRef
#define hipTextureAddressMode            gpuTextureAddressMode
#define hipTextureDesc                   gpuTextureDesc
#define hipTextureFilterMode             gpuTextureFilterMode
#define hipTextureObject_t               gpuTextureObject_t
#define hipTextureReadMode               gpuTextureReadMode
#define hipTextureType1D                 gpuTextureType1D
#define hipTextureType1DLayered          gpuTextureType1DLayered
#define hipTextureType2D                 gpuTextureType2D
#define hipTextureType2DLayered          gpuTextureType2DLayered
#define hipTextureType3D                 gpuTextureType3D
#define hipTextureTypeCubemap            gpuTextureTypeCubemap
#define hipTextureTypeCubemapLayered     gpuTextureTypeCubemapLayered

#include <hop/hop_runtime_api.h>

#endif
