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

#ifndef __HOP_SOURCE_CUDA_CUDA_H__
#define __HOP_SOURCE_CUDA_CUDA_H__

#define HOP_SOURCE_CUDA

#define CUDA_ARRAY3D_CUBEMAP             gpuArrayCubemap
#define CUDA_ARRAY3D_DESCRIPTOR          GPU_ARRAY3D_DESCRIPTOR
#define CUDA_ARRAY3D_DESCRIPTOR_st       GPU_ARRAY3D_DESCRIPTOR
#define CUDA_ARRAY3D_DESCRIPTOR_v2       GPU_ARRAY3D_DESCRIPTOR
#define CUDA_ARRAY3D_LAYERED             gpuArrayLayered
#define CUDA_ARRAY3D_SURFACE_LDST        gpuArraySurfaceLoadStore
#define CUDA_ARRAY3D_TEXTURE_GATHER      gpuArrayTextureGather
#define CUDA_ARRAY_DESCRIPTOR            GPU_ARRAY_DESCRIPTOR
#define CUDA_ARRAY_DESCRIPTOR_st         GPU_ARRAY_DESCRIPTOR
#define CUDA_ARRAY_DESCRIPTOR_v1         GPU_ARRAY_DESCRIPTOR
#define CUDA_ARRAY_DESCRIPTOR_v1_st      GPU_ARRAY_DESCRIPTOR
#define CUDA_ARRAY_DESCRIPTOR_v2         GPU_ARRAY_DESCRIPTOR
#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC  \
        gpuCooperativeLaunchMultiDeviceNoPostSync
#define CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC  \
        gpuCooperativeLaunchMultiDeviceNoPreSync
#define CUDA_ERROR_ALREADY_ACQUIRED      gpuErrorAlreadyAcquired
#define CUDA_ERROR_ALREADY_MAPPED        gpuErrorAlreadyMapped
#define CUDA_ERROR_ARRAY_IS_MAPPED       gpuErrorArrayIsMapped
#define CUDA_ERROR_ASSERT                gpuErrorAssert
#define CUDA_ERROR_CAPTURED_EVENT        gpuErrorCapturedEvent
#define CUDA_ERROR_CONTEXT_ALREADY_CURRENT  \
        gpuErrorContextAlreadyCurrent
#define CUDA_ERROR_CONTEXT_ALREADY_IN_USE  \
        gpuErrorContextAlreadyInUse
#define CUDA_ERROR_CONTEXT_IS_DESTROYED  gpuErrorContextIsDestroyed
#define CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE  \
        gpuErrorCooperativeLaunchTooLarge
#define CUDA_ERROR_DEINITIALIZED         gpuErrorDeinitialized
#define CUDA_ERROR_ECC_UNCORRECTABLE     gpuErrorECCNotCorrectable
#define CUDA_ERROR_FILE_NOT_FOUND        gpuErrorFileNotFound
#define CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE  \
        gpuErrorGraphExecUpdateFailure
#define CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED  \
        gpuErrorHostMemoryAlreadyRegistered
#define CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED  \
        gpuErrorHostMemoryNotRegistered
#define CUDA_ERROR_ILLEGAL_ADDRESS       gpuErrorIllegalAddress
#define CUDA_ERROR_ILLEGAL_STATE         gpuErrorIllegalState
#define CUDA_ERROR_INVALID_CONTEXT       gpuErrorInvalidContext
#define CUDA_ERROR_INVALID_DEVICE        gpuErrorInvalidDevice
#define CUDA_ERROR_INVALID_GRAPHICS_CONTEXT  \
        gpuErrorInvalidGraphicsContext
#define CUDA_ERROR_INVALID_HANDLE        gpuErrorInvalidHandle
#define CUDA_ERROR_INVALID_IMAGE         gpuErrorInvalidImage
#define CUDA_ERROR_INVALID_PTX           gpuErrorInvalidKernelFile
#define CUDA_ERROR_INVALID_SOURCE        gpuErrorInvalidSource
#define CUDA_ERROR_INVALID_VALUE         gpuErrorInvalidValue
#define CUDA_ERROR_LAUNCH_FAILED         gpuErrorLaunchFailure
#define CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES  \
        gpuErrorLaunchOutOfResources
#define CUDA_ERROR_LAUNCH_TIMEOUT        gpuErrorLaunchTimeOut
#define CUDA_ERROR_MAP_FAILED            gpuErrorMapFailed
#define CUDA_ERROR_NOT_FOUND             gpuErrorNotFound
#define CUDA_ERROR_NOT_INITIALIZED       gpuErrorNotInitialized
#define CUDA_ERROR_NOT_MAPPED            gpuErrorNotMapped
#define CUDA_ERROR_NOT_MAPPED_AS_ARRAY   gpuErrorNotMappedAsArray
#define CUDA_ERROR_NOT_MAPPED_AS_POINTER gpuErrorNotMappedAsPointer
#define CUDA_ERROR_NOT_READY             gpuErrorNotReady
#define CUDA_ERROR_NOT_SUPPORTED         gpuErrorNotSupported
#define CUDA_ERROR_NO_BINARY_FOR_GPU     gpuErrorNoBinaryForGpu
#define CUDA_ERROR_NO_DEVICE             gpuErrorNoDevice
#define CUDA_ERROR_OPERATING_SYSTEM      gpuErrorOperatingSystem
#define CUDA_ERROR_OUT_OF_MEMORY         gpuErrorOutOfMemory
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED  \
        gpuErrorPeerAccessAlreadyEnabled
#define CUDA_ERROR_PEER_ACCESS_NOT_ENABLED  \
        gpuErrorPeerAccessNotEnabled
#define CUDA_ERROR_PEER_ACCESS_UNSUPPORTED  \
        gpuErrorPeerAccessUnsupported
#define CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE  \
        gpuErrorSetOnActiveProcess
#define CUDA_ERROR_PROFILER_ALREADY_STARTED  \
        gpuErrorProfilerAlreadyStarted
#define CUDA_ERROR_PROFILER_ALREADY_STOPPED  \
        gpuErrorProfilerAlreadyStopped
#define CUDA_ERROR_PROFILER_DISABLED     gpuErrorProfilerDisabled
#define CUDA_ERROR_PROFILER_NOT_INITIALIZED  \
        gpuErrorProfilerNotInitialized
#define CUDA_ERROR_SHARED_OBJECT_INIT_FAILED  \
        gpuErrorSharedObjectInitFailed
#define CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND  \
        gpuErrorSharedObjectSymbolNotFound
#define CUDA_ERROR_STREAM_CAPTURE_IMPLICIT  \
        gpuErrorStreamCaptureImplicit
#define CUDA_ERROR_STREAM_CAPTURE_INVALIDATED  \
        gpuErrorStreamCaptureInvalidated
#define CUDA_ERROR_STREAM_CAPTURE_ISOLATION  \
        gpuErrorStreamCaptureIsolation
#define CUDA_ERROR_STREAM_CAPTURE_MERGE  gpuErrorStreamCaptureMerge
#define CUDA_ERROR_STREAM_CAPTURE_UNJOINED  \
        gpuErrorStreamCaptureUnjoined
#define CUDA_ERROR_STREAM_CAPTURE_UNMATCHED  \
        gpuErrorStreamCaptureUnmatched
#define CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED  \
        gpuErrorStreamCaptureUnsupported
#define CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD  \
        gpuErrorStreamCaptureWrongThread
#define CUDA_ERROR_UNKNOWN               gpuErrorUnknown
#define CUDA_ERROR_UNMAP_FAILED          gpuErrorUnmapFailed
#define CUDA_ERROR_UNSUPPORTED_LIMIT     gpuErrorUnsupportedLimit
#define CUDA_EXTERNAL_MEMORY_BUFFER_DESC gpuExternalMemoryBufferDesc
#define CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st  \
        gpuExternalMemoryBufferDesc_st
#define CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1  \
        gpuExternalMemoryBufferDesc
#define CUDA_EXTERNAL_MEMORY_DEDICATED   gpuExternalMemoryDedicated
#define CUDA_EXTERNAL_MEMORY_HANDLE_DESC gpuExternalMemoryHandleDesc
#define CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st  \
        gpuExternalMemoryHandleDesc_st
#define CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1  \
        gpuExternalMemoryHandleDesc
#define CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC  \
        gpuExternalSemaphoreHandleDesc
#define CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st  \
        gpuExternalSemaphoreHandleDesc_st
#define CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1  \
        gpuExternalSemaphoreHandleDesc
#define CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS  \
        gpuExternalSemaphoreSignalParams
#define CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st  \
        gpuExternalSemaphoreSignalParams_st
#define CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1  \
        gpuExternalSemaphoreSignalParams
#define CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS  \
        gpuExternalSemaphoreWaitParams
#define CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st  \
        gpuExternalSemaphoreWaitParams_st
#define CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1  \
        gpuExternalSemaphoreWaitParams
#define CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH  \
        gpuGraphInstantiateFlagAutoFreeOnLaunch
#define CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH  \
        gpuGraphInstantiateFlagDeviceLaunch
#define CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD  \
        gpuGraphInstantiateFlagUpload
#define CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY  \
        gpuGraphInstantiateFlagUseNodePriority
#define CUDA_HOST_NODE_PARAMS            gpuHostNodeParams
#define CUDA_HOST_NODE_PARAMS_st         gpuHostNodeParams
#define CUDA_HOST_NODE_PARAMS_v1         gpuHostNodeParams
#define CUDA_KERNEL_NODE_PARAMS          gpuKernelNodeParams
#define CUDA_KERNEL_NODE_PARAMS_st       gpuKernelNodeParams
#define CUDA_KERNEL_NODE_PARAMS_v1       gpuKernelNodeParams
#define CUDA_LAUNCH_PARAMS               gpuFunctionLaunchParams
#define CUDA_LAUNCH_PARAMS_st            gpuFunctionLaunchParams_t
#define CUDA_LAUNCH_PARAMS_v1            gpuFunctionLaunchParams
#define CUDA_MEMCPY2D                    gpu_Memcpy2D
#define CUDA_MEMCPY2D_st                 gpu_Memcpy2D
#define CUDA_MEMCPY2D_v1                 gpu_Memcpy2D
#define CUDA_MEMCPY2D_v1_st              gpu_Memcpy2D
#define CUDA_MEMCPY2D_v2                 gpu_Memcpy2D
#define CUDA_MEMCPY3D                    GPU_MEMCPY3D
#define CUDA_MEMCPY3D_st                 GPU_MEMCPY3D
#define CUDA_MEMCPY3D_v1                 GPU_MEMCPY3D
#define CUDA_MEMCPY3D_v1_st              GPU_MEMCPY3D
#define CUDA_MEMCPY3D_v2                 GPU_MEMCPY3D
#define CUDA_MEMSET_NODE_PARAMS          gpuMemsetParams
#define CUDA_MEMSET_NODE_PARAMS_st       gpuMemsetParams
#define CUDA_MEMSET_NODE_PARAMS_v1       gpuMemsetParams
#define CUDA_MEM_ALLOC_NODE_PARAMS       gpuMemAllocNodeParams
#define CUDA_MEM_ALLOC_NODE_PARAMS_st    gpuMemAllocNodeParams
#define CUDA_RESOURCE_DESC               GPU_RESOURCE_DESC
#define CUDA_RESOURCE_DESC_st            GPU_RESOURCE_DESC_st
#define CUDA_RESOURCE_DESC_v1            GPU_RESOURCE_DESC
#define CUDA_RESOURCE_VIEW_DESC          GPU_RESOURCE_VIEW_DESC
#define CUDA_RESOURCE_VIEW_DESC_st       GPU_RESOURCE_VIEW_DESC_st
#define CUDA_RESOURCE_VIEW_DESC_v1       GPU_RESOURCE_VIEW_DESC
#define CUDA_SUCCESS                     gpuSuccess
#define CUDA_TEXTURE_DESC                GPU_TEXTURE_DESC
#define CUDA_TEXTURE_DESC_st             GPU_TEXTURE_DESC_st
#define CUDA_TEXTURE_DESC_v1             GPU_TEXTURE_DESC
#define CU_ACCESS_PROPERTY_NORMAL        gpuAccessPropertyNormal
#define CU_ACCESS_PROPERTY_PERSISTING    gpuAccessPropertyPersisting
#define CU_ACCESS_PROPERTY_STREAMING     gpuAccessPropertyStreaming
#define CU_AD_FORMAT_FLOAT               GPU_AD_FORMAT_FLOAT
#define CU_AD_FORMAT_HALF                GPU_AD_FORMAT_HALF
#define CU_AD_FORMAT_SIGNED_INT16        GPU_AD_FORMAT_SIGNED_INT16
#define CU_AD_FORMAT_SIGNED_INT32        GPU_AD_FORMAT_SIGNED_INT32
#define CU_AD_FORMAT_SIGNED_INT8         GPU_AD_FORMAT_SIGNED_INT8
#define CU_AD_FORMAT_UNSIGNED_INT16      GPU_AD_FORMAT_UNSIGNED_INT16
#define CU_AD_FORMAT_UNSIGNED_INT32      GPU_AD_FORMAT_UNSIGNED_INT32
#define CU_AD_FORMAT_UNSIGNED_INT8       GPU_AD_FORMAT_UNSIGNED_INT8
#define CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL  \
        gpuArraySparseSubresourceTypeMiptail
#define CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL  \
        gpuArraySparseSubresourceTypeSparseLevel
#define CU_COMPUTEMODE_DEFAULT           gpuComputeModeDefault
#define CU_COMPUTEMODE_EXCLUSIVE_PROCESS gpuComputeModeExclusiveProcess
#define CU_COMPUTEMODE_PROHIBITED        gpuComputeModeProhibited
#define CU_CTX_BLOCKING_SYNC             gpuDeviceScheduleBlockingSync
#define CU_CTX_LMEM_RESIZE_TO_MAX        gpuDeviceLmemResizeToMax
#define CU_CTX_MAP_HOST                  gpuDeviceMapHost
#define CU_CTX_SCHED_AUTO                gpuDeviceScheduleAuto
#define CU_CTX_SCHED_BLOCKING_SYNC       gpuDeviceScheduleBlockingSync
#define CU_CTX_SCHED_MASK                gpuDeviceScheduleMask
#define CU_CTX_SCHED_SPIN                gpuDeviceScheduleSpin
#define CU_CTX_SCHED_YIELD               gpuDeviceScheduleYield
#define CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT  \
        gpuDeviceAttributeAsyncEngineCount
#define CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY  \
        gpuDeviceAttributeCanMapHostMemory
#define CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM  \
        gpuDeviceAttributeCanUseHostPointerForRegisteredMem
#define CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR  \
        gpuDeviceAttributeCanUseStreamWaitValue
#define CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1  \
        gpuDeviceAttributeCanUseStreamWaitValue
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE   gpuDeviceAttributeClockRate
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR  \
        gpuDeviceAttributeComputeCapabilityMajor
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR  \
        gpuDeviceAttributeComputeCapabilityMinor
#define CU_DEVICE_ATTRIBUTE_COMPUTE_MODE gpuDeviceAttributeComputeMode
#define CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED  \
        gpuDeviceAttributeComputePreemptionSupported
#define CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS  \
        gpuDeviceAttributeConcurrentKernels
#define CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS  \
        gpuDeviceAttributeConcurrentManagedAccess
#define CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH  \
        gpuDeviceAttributeCooperativeLaunch
#define CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH  \
        gpuDeviceAttributeCooperativeMultiDeviceLaunch
#define CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST  \
        gpuDeviceAttributeDirectManagedMemAccessFromHost
#define CU_DEVICE_ATTRIBUTE_ECC_ENABLED  gpuDeviceAttributeEccEnabled
#define CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED  \
        gpuDeviceAttributeGlobalL1CacheSupported
#define CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH  \
        gpuDeviceAttributeMemoryBusWidth
#define CU_DEVICE_ATTRIBUTE_GPU_OVERLAP  gpuDeviceAttributeAsyncEngineCount
#define CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED  \
        gpuDeviceAttributeHostNativeAtomicSupported
#define CU_DEVICE_ATTRIBUTE_INTEGRATED   gpuDeviceAttributeIntegrated
#define CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT  \
        gpuDeviceAttributeKernelExecTimeout
#define CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE  \
        gpuDeviceAttributeL2CacheSize
#define CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED  \
        gpuDeviceAttributeLocalL1CacheSupported
#define CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY  \
        gpuDeviceAttributeManagedMemory
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH  \
        gpuDeviceAttributeMaxSurface1DLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH  \
        gpuDeviceAttributeMaxSurface1D
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT  \
        gpuDeviceAttributeMaxSurface2D
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT  \
        gpuDeviceAttributeMaxSurface2DLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH  \
        gpuDeviceAttributeMaxSurface2DLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH  \
        gpuDeviceAttributeMaxSurface2D
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH  \
        gpuDeviceAttributeMaxSurface3D
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT  \
        gpuDeviceAttributeMaxSurface3D
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH  \
        gpuDeviceAttributeMaxSurface3D
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH  \
        gpuDeviceAttributeMaxSurfaceCubemapLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH  \
        gpuDeviceAttributeMaxSurfaceCubemap
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH  \
        gpuDeviceAttributeMaxTexture1DLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH  \
        gpuDeviceAttributeMaxTexture1DLinear
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH  \
        gpuDeviceAttributeMaxTexture1DMipmap
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH  \
        gpuDeviceAttributeMaxTexture1DWidth
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT  \
        gpuDeviceAttributeMaxTexture2DLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH  \
        gpuDeviceAttributeMaxTexture2DLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT  \
        gpuDeviceAttributeMaxTexture2DGather
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH  \
        gpuDeviceAttributeMaxTexture2DGather
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT  \
        gpuDeviceAttributeMaxTexture2DHeight
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT  \
        gpuDeviceAttributeMaxTexture2DLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH  \
        gpuDeviceAttributeMaxTexture2DLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT  \
        gpuDeviceAttributeMaxTexture2DLinear
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH  \
        gpuDeviceAttributeMaxTexture2DLinear
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH  \
        gpuDeviceAttributeMaxTexture2DLinear
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT  \
        gpuDeviceAttributeMaxTexture2DMipmap
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH  \
        gpuDeviceAttributeMaxTexture2DMipmap
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH  \
        gpuDeviceAttributeMaxTexture2DWidth
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH  \
        gpuDeviceAttributeMaxTexture3DDepth
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE  \
        gpuDeviceAttributeMaxTexture3DAlt
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT  \
        gpuDeviceAttributeMaxTexture3DHeight
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE  \
        gpuDeviceAttributeMaxTexture3DAlt
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH  \
        gpuDeviceAttributeMaxTexture3DWidth
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE  \
        gpuDeviceAttributeMaxTexture3DAlt
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH  \
        gpuDeviceAttributeMaxTextureCubemapLayered
#define CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH  \
        gpuDeviceAttributeMaxTextureCubemap
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR  \
        gpuDeviceAttributeMaxBlocksPerMultiProcessor
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X  \
        gpuDeviceAttributeMaxBlockDimX
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y  \
        gpuDeviceAttributeMaxBlockDimY
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z  \
        gpuDeviceAttributeMaxBlockDimZ
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X  \
        gpuDeviceAttributeMaxGridDimX
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y  \
        gpuDeviceAttributeMaxGridDimY
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z  \
        gpuDeviceAttributeMaxGridDimZ
#define CU_DEVICE_ATTRIBUTE_MAX_PITCH    gpuDeviceAttributeMaxPitch
#define CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK  \
        gpuDeviceAttributeMaxRegistersPerBlock
#define CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR  \
        gpuDeviceAttributeMaxRegistersPerMultiprocessor
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK  \
        gpuDeviceAttributeMaxSharedMemoryPerBlock
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN  \
        gpuDeviceAttributeSharedMemPerBlockOptin
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR  \
        gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK  \
        gpuDeviceAttributeMaxThreadsPerBlock
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR  \
        gpuDeviceAttributeMaxThreadsPerMultiProcessor
#define CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE  \
        gpuDeviceAttributeMemoryClockRate
#define CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED  \
        gpuDeviceAttributeMemoryPoolsSupported
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT  \
        gpuDeviceAttributeMultiprocessorCount
#define CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD  \
        gpuDeviceAttributeIsMultiGpuBoard
#define CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID  \
        gpuDeviceAttributeMultiGpuBoardGroupID
#define CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS  \
        gpuDeviceAttributePageableMemoryAccess
#define CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES  \
        gpuDeviceAttributePageableMemoryAccessUsesHostPageTables
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID   gpuDeviceAttributePciBusId
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID  \
        gpuDeviceAttributePciDeviceId
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID  \
        gpuDeviceAttributePciDomainID
#define CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK  \
        gpuDeviceAttributeMaxRegistersPerBlock
#define CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK  \
        gpuDeviceAttributeMaxSharedMemoryPerBlock
#define CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO  \
        gpuDeviceAttributeSingleToDoublePrecisionPerfRatio
#define CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED  \
        gpuDeviceAttributeStreamPrioritiesSupported
#define CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT  \
        gpuDeviceAttributeSurfaceAlignment
#define CU_DEVICE_ATTRIBUTE_TCC_DRIVER   gpuDeviceAttributeTccDriver
#define CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT  \
        gpuDeviceAttributeTextureAlignment
#define CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT  \
        gpuDeviceAttributeTexturePitchAlignment
#define CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY  \
        gpuDeviceAttributeTotalConstantMemory
#define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING  \
        gpuDeviceAttributeUnifiedAddressing
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED  \
        gpuDeviceAttributeVirtualMemoryManagementSupported
#define CU_DEVICE_ATTRIBUTE_WARP_SIZE    gpuDeviceAttributeWarpSize
#define CU_DEVICE_CPU                    gpuCpuDeviceId
#define CU_DEVICE_INVALID                gpuInvalidDeviceId
#define CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED  \
        gpuDevP2PAttrHipArrayAccessSupported
#define CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED  \
        gpuDevP2PAttrAccessSupported
#define CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED  \
        gpuDevP2PAttrHipArrayAccessSupported
#define CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED  \
        gpuDevP2PAttrNativeAtomicSupported
#define CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK  \
        gpuDevP2PAttrPerformanceRank
#define CU_EVENT_BLOCKING_SYNC           gpuEventBlockingSync
#define CU_EVENT_DEFAULT                 gpuEventDefault
#define CU_EVENT_DISABLE_TIMING          gpuEventDisableTiming
#define CU_EVENT_INTERPROCESS            gpuEventInterprocess
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE  \
        gpuExternalMemoryHandleTypeD3D11Resource
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT  \
        gpuExternalMemoryHandleTypeD3D11ResourceKmt
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP  \
        gpuExternalMemoryHandleTypeD3D12Heap
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE  \
        gpuExternalMemoryHandleTypeD3D12Resource
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD  \
        gpuExternalMemoryHandleTypeOpaqueFd
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32  \
        gpuExternalMemoryHandleTypeOpaqueWin32
#define CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT  \
        gpuExternalMemoryHandleTypeOpaqueWin32Kmt
#define CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE  \
        gpuExternalSemaphoreHandleTypeD3D12Fence
#define CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD  \
        gpuExternalSemaphoreHandleTypeOpaqueFd
#define CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32  \
        gpuExternalSemaphoreHandleTypeOpaqueWin32
#define CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT  \
        gpuExternalSemaphoreHandleTypeOpaqueWin32Kmt
#define CU_FUNC_ATTRIBUTE_BINARY_VERSION GPU_FUNC_ATTRIBUTE_BINARY_VERSION
#define CU_FUNC_ATTRIBUTE_CACHE_MODE_CA  GPU_FUNC_ATTRIBUTE_CACHE_MODE_CA
#define CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES  \
        GPU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES  \
        GPU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_MAX            GPU_FUNC_ATTRIBUTE_MAX
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES  \
        GPU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK  \
        GPU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define CU_FUNC_ATTRIBUTE_NUM_REGS       GPU_FUNC_ATTRIBUTE_NUM_REGS
#define CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT  \
        GPU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
#define CU_FUNC_ATTRIBUTE_PTX_VERSION    GPU_FUNC_ATTRIBUTE_PTX_VERSION
#define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES  \
        GPU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define CU_FUNC_CACHE_PREFER_EQUAL       gpuFuncCachePreferEqual
#define CU_FUNC_CACHE_PREFER_L1          gpuFuncCachePreferL1
#define CU_FUNC_CACHE_PREFER_NONE        gpuFuncCachePreferNone
#define CU_FUNC_CACHE_PREFER_SHARED      gpuFuncCachePreferShared
#define CU_GRAPHICS_REGISTER_FLAGS_NONE  gpuGraphicsRegisterFlagsNone
#define CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY  \
        gpuGraphicsRegisterFlagsReadOnly
#define CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST  \
        gpuGraphicsRegisterFlagsSurfaceLoadStore
#define CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER  \
        gpuGraphicsRegisterFlagsTextureGather
#define CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD  \
        gpuGraphicsRegisterFlagsWriteDiscard
#define CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS  \
        gpuGraphDebugDotFlagsEventNodeParams
#define CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS  \
        gpuGraphDebugDotFlagsExtSemasSignalNodeParams
#define CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS  \
        gpuGraphDebugDotFlagsExtSemasWaitNodeParams
#define CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES gpuGraphDebugDotFlagsHandles
#define CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS  \
        gpuGraphDebugDotFlagsHostNodeParams
#define CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES  \
        gpuGraphDebugDotFlagsKernelNodeAttributes
#define CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS  \
        gpuGraphDebugDotFlagsKernelNodeParams
#define CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS  \
        gpuGraphDebugDotFlagsMemcpyNodeParams
#define CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS  \
        gpuGraphDebugDotFlagsMemsetNodeParams
#define CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE gpuGraphDebugDotFlagsVerbose
#define CU_GRAPH_EXEC_UPDATE_ERROR       gpuGraphExecUpdateError
#define CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED  \
        gpuGraphExecUpdateErrorFunctionChanged
#define CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED  \
        gpuGraphExecUpdateErrorNodeTypeChanged
#define CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED  \
        gpuGraphExecUpdateErrorNotSupported
#define CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED  \
        gpuGraphExecUpdateErrorParametersChanged
#define CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED  \
        gpuGraphExecUpdateErrorTopologyChanged
#define CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE  \
        gpuGraphExecUpdateErrorUnsupportedFunctionChange
#define CU_GRAPH_EXEC_UPDATE_SUCCESS     gpuGraphExecUpdateSuccess
#define CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT  \
        gpuGraphMemAttrReservedMemCurrent
#define CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH  \
        gpuGraphMemAttrReservedMemHigh
#define CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT  \
        gpuGraphMemAttrUsedMemCurrent
#define CU_GRAPH_MEM_ATTR_USED_MEM_HIGH  gpuGraphMemAttrUsedMemHigh
#define CU_GRAPH_NODE_TYPE_EMPTY         gpuGraphNodeTypeEmpty
#define CU_GRAPH_NODE_TYPE_EVENT_RECORD  gpuGraphNodeTypeEventRecord
#define CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL  \
        gpuGraphNodeTypeExtSemaphoreSignal
#define CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT  \
        gpuGraphNodeTypeExtSemaphoreWait
#define CU_GRAPH_NODE_TYPE_GRAPH         gpuGraphNodeTypeGraph
#define CU_GRAPH_NODE_TYPE_HOST          gpuGraphNodeTypeHost
#define CU_GRAPH_NODE_TYPE_KERNEL        gpuGraphNodeTypeKernel
#define CU_GRAPH_NODE_TYPE_MEMCPY        gpuGraphNodeTypeMemcpy
#define CU_GRAPH_NODE_TYPE_MEMSET        gpuGraphNodeTypeMemset
#define CU_GRAPH_NODE_TYPE_MEM_ALLOC     gpuGraphNodeTypeMemAlloc
#define CU_GRAPH_NODE_TYPE_MEM_FREE      gpuGraphNodeTypeMemFree
#define CU_GRAPH_NODE_TYPE_WAIT_EVENT    gpuGraphNodeTypeWaitEvent
#define CU_GRAPH_USER_OBJECT_MOVE        gpuGraphUserObjectMove
#define CU_IPC_HANDLE_SIZE               GPU_IPC_HANDLE_SIZE
#define CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS  \
        gpuIpcMemLazyEnablePeerAccess
#define CU_JIT_CACHE_MODE                GPURTC_JIT_CACHE_MODE
#define CU_JIT_ERROR_LOG_BUFFER          GPURTC_JIT_ERROR_LOG_BUFFER
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES  \
        GPURTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#define CU_JIT_FALLBACK_STRATEGY         GPURTC_JIT_FALLBACK_STRATEGY
#define CU_JIT_FAST_COMPILE              GPURTC_JIT_FAST_COMPILE
#define CU_JIT_GENERATE_DEBUG_INFO       GPURTC_JIT_GENERATE_DEBUG_INFO
#define CU_JIT_GENERATE_LINE_INFO        GPURTC_JIT_GENERATE_LINE_INFO
#define CU_JIT_INFO_LOG_BUFFER           GPURTC_JIT_INFO_LOG_BUFFER
#define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES  \
        GPURTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES
#define CU_JIT_INPUT_CUBIN               GPURTC_JIT_INPUT_CUBIN
#define CU_JIT_INPUT_FATBINARY           GPURTC_JIT_INPUT_FATBINARY
#define CU_JIT_INPUT_LIBRARY             GPURTC_JIT_INPUT_LIBRARY
#define CU_JIT_INPUT_NVVM                GPURTC_JIT_INPUT_NVVM
#define CU_JIT_INPUT_OBJECT              GPURTC_JIT_INPUT_OBJECT
#define CU_JIT_INPUT_PTX                 GPURTC_JIT_INPUT_PTX
#define CU_JIT_LOG_VERBOSE               GPURTC_JIT_LOG_VERBOSE
#define CU_JIT_MAX_REGISTERS             GPURTC_JIT_MAX_REGISTERS
#define CU_JIT_NEW_SM3X_OPT              GPURTC_JIT_NEW_SM3X_OPT
#define CU_JIT_NUM_INPUT_TYPES           GPURTC_JIT_NUM_LEGACY_INPUT_TYPES
#define CU_JIT_NUM_OPTIONS               GPURTC_JIT_NUM_OPTIONS
#define CU_JIT_OPTIMIZATION_LEVEL        GPURTC_JIT_OPTIMIZATION_LEVEL
#define CU_JIT_TARGET                    GPURTC_JIT_TARGET
#define CU_JIT_TARGET_FROM_CUCONTEXT     GPURTC_JIT_TARGET_FROM_HIPCONTEXT
#define CU_JIT_THREADS_PER_BLOCK         GPURTC_JIT_THREADS_PER_BLOCK
#define CU_JIT_WALL_TIME                 GPURTC_JIT_WALL_TIME
#define CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW  \
        gpuKernelNodeAttributeAccessPolicyWindow
#define CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE  \
        gpuKernelNodeAttributeCooperative
#define CU_LAUNCH_PARAM_BUFFER_POINTER   GPU_LAUNCH_PARAM_BUFFER_POINTER
#define CU_LAUNCH_PARAM_BUFFER_SIZE      GPU_LAUNCH_PARAM_BUFFER_SIZE
#define CU_LAUNCH_PARAM_END              GPU_LAUNCH_PARAM_END
#define CU_LIMIT_MALLOC_HEAP_SIZE        gpuLimitMallocHeapSize
#define CU_LIMIT_PRINTF_FIFO_SIZE        gpuLimitPrintfFifoSize
#define CU_LIMIT_STACK_SIZE              gpuLimitStackSize
#define CU_MEMHOSTALLOC_DEVICEMAP        gpuHostMallocMapped
#define CU_MEMHOSTALLOC_PORTABLE         gpuHostMallocPortable
#define CU_MEMHOSTALLOC_WRITECOMBINED    gpuHostMallocWriteCombined
#define CU_MEMHOSTREGISTER_DEVICEMAP     gpuHostRegisterMapped
#define CU_MEMHOSTREGISTER_IOMEMORY      gpuHostRegisterIoMemory
#define CU_MEMHOSTREGISTER_PORTABLE      gpuHostRegisterPortable
#define CU_MEMHOSTREGISTER_READ_ONLY     gpuHostRegisterReadOnly
#define CU_MEMORYTYPE_ARRAY              gpuMemoryTypeArray
#define CU_MEMORYTYPE_DEVICE             gpuMemoryTypeDevice
#define CU_MEMORYTYPE_HOST               gpuMemoryTypeHost
#define CU_MEMORYTYPE_UNIFIED            gpuMemoryTypeUnified
#define CU_MEMPOOL_ATTR_RELEASE_THRESHOLD  \
        gpuMemPoolAttrReleaseThreshold
#define CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT  \
        gpuMemPoolAttrReservedMemCurrent
#define CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH  \
        gpuMemPoolAttrReservedMemHigh
#define CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES  \
        gpuMemPoolReuseAllowInternalDependencies
#define CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC  \
        gpuMemPoolReuseAllowOpportunistic
#define CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES  \
        gpuMemPoolReuseFollowEventDependencies
#define CU_MEMPOOL_ATTR_USED_MEM_CURRENT gpuMemPoolAttrUsedMemCurrent
#define CU_MEMPOOL_ATTR_USED_MEM_HIGH    gpuMemPoolAttrUsedMemHigh
#define CU_MEM_ACCESS_FLAGS_PROT_NONE    gpuMemAccessFlagsProtNone
#define CU_MEM_ACCESS_FLAGS_PROT_READ    gpuMemAccessFlagsProtRead
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE  \
        gpuMemAccessFlagsProtReadWrite
#define CU_MEM_ADVISE_SET_ACCESSED_BY    gpuMemAdviseSetAccessedBy
#define CU_MEM_ADVISE_SET_PREFERRED_LOCATION  \
        gpuMemAdviseSetPreferredLocation
#define CU_MEM_ADVISE_SET_READ_MOSTLY    gpuMemAdviseSetReadMostly
#define CU_MEM_ADVISE_UNSET_ACCESSED_BY  gpuMemAdviseUnsetAccessedBy
#define CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION  \
        gpuMemAdviseUnsetPreferredLocation
#define CU_MEM_ADVISE_UNSET_READ_MOSTLY  gpuMemAdviseUnsetReadMostly
#define CU_MEM_ALLOCATION_TYPE_INVALID   gpuMemAllocationTypeInvalid
#define CU_MEM_ALLOCATION_TYPE_MAX       gpuMemAllocationTypeMax
#define CU_MEM_ALLOCATION_TYPE_PINNED    gpuMemAllocationTypePinned
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM gpuMemAllocationGranularityMinimum
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED  \
        gpuMemAllocationGranularityRecommended
#define CU_MEM_ATTACH_GLOBAL             gpuMemAttachGlobal
#define CU_MEM_ATTACH_HOST               gpuMemAttachHost
#define CU_MEM_ATTACH_SINGLE             gpuMemAttachSingle
#define CU_MEM_HANDLE_TYPE_GENERIC       gpuMemHandleTypeGeneric
#define CU_MEM_HANDLE_TYPE_NONE          gpuMemHandleTypeNone
#define CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR  \
        gpuMemHandleTypePosixFileDescriptor
#define CU_MEM_HANDLE_TYPE_WIN32         gpuMemHandleTypeWin32
#define CU_MEM_HANDLE_TYPE_WIN32_KMT     gpuMemHandleTypeWin32Kmt
#define CU_MEM_LOCATION_TYPE_DEVICE      gpuMemLocationTypeDevice
#define CU_MEM_LOCATION_TYPE_INVALID     gpuMemLocationTypeInvalid
#define CU_MEM_OPERATION_TYPE_MAP        gpuMemOperationTypeMap
#define CU_MEM_OPERATION_TYPE_UNMAP      gpuMemOperationTypeUnmap
#define CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY  \
        gpuMemRangeAttributeAccessedBy
#define CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION  \
        gpuMemRangeAttributeLastPrefetchLocation
#define CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION  \
        gpuMemRangeAttributePreferredLocation
#define CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY  \
        gpuMemRangeAttributeReadMostly
#define CU_OCCUPANCY_DEFAULT             gpuOccupancyDefault
#define CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE  \
        gpuOccupancyDisableCachingOverride
#define CU_POINTER_ATTRIBUTE_ACCESS_FLAGS  \
        GPU_POINTER_ATTRIBUTE_ACCESS_FLAGS
#define CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES  \
        GPU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
#define CU_POINTER_ATTRIBUTE_BUFFER_ID   GPU_POINTER_ATTRIBUTE_BUFFER_ID
#define CU_POINTER_ATTRIBUTE_CONTEXT     GPU_POINTER_ATTRIBUTE_CONTEXT
#define CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL  \
        GPU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
#define CU_POINTER_ATTRIBUTE_DEVICE_POINTER  \
        GPU_POINTER_ATTRIBUTE_DEVICE_POINTER
#define CU_POINTER_ATTRIBUTE_HOST_POINTER  \
        GPU_POINTER_ATTRIBUTE_HOST_POINTER
#define CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE  \
        GPU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
#define CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE  \
        GPU_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
#define CU_POINTER_ATTRIBUTE_IS_MANAGED  GPU_POINTER_ATTRIBUTE_IS_MANAGED
#define CU_POINTER_ATTRIBUTE_MAPPED      GPU_POINTER_ATTRIBUTE_MAPPED
#define CU_POINTER_ATTRIBUTE_MEMORY_TYPE GPU_POINTER_ATTRIBUTE_MEMORY_TYPE
#define CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE  \
        GPU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
#define CU_POINTER_ATTRIBUTE_P2P_TOKENS  GPU_POINTER_ATTRIBUTE_P2P_TOKENS
#define CU_POINTER_ATTRIBUTE_RANGE_SIZE  GPU_POINTER_ATTRIBUTE_RANGE_SIZE
#define CU_POINTER_ATTRIBUTE_RANGE_START_ADDR  \
        GPU_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define CU_POINTER_ATTRIBUTE_SYNC_MEMOPS GPU_POINTER_ATTRIBUTE_SYNC_MEMOPS
#define CU_RESOURCE_TYPE_ARRAY           GPU_RESOURCE_TYPE_ARRAY
#define CU_RESOURCE_TYPE_LINEAR          GPU_RESOURCE_TYPE_LINEAR
#define CU_RESOURCE_TYPE_MIPMAPPED_ARRAY GPU_RESOURCE_TYPE_MIPMAPPED_ARRAY
#define CU_RESOURCE_TYPE_PITCH2D         GPU_RESOURCE_TYPE_PITCH2D
#define CU_RES_VIEW_FORMAT_FLOAT_1X16    GPU_RES_VIEW_FORMAT_FLOAT_1X16
#define CU_RES_VIEW_FORMAT_FLOAT_1X32    GPU_RES_VIEW_FORMAT_FLOAT_1X32
#define CU_RES_VIEW_FORMAT_FLOAT_2X16    GPU_RES_VIEW_FORMAT_FLOAT_2X16
#define CU_RES_VIEW_FORMAT_FLOAT_2X32    GPU_RES_VIEW_FORMAT_FLOAT_2X32
#define CU_RES_VIEW_FORMAT_FLOAT_4X16    GPU_RES_VIEW_FORMAT_FLOAT_4X16
#define CU_RES_VIEW_FORMAT_FLOAT_4X32    GPU_RES_VIEW_FORMAT_FLOAT_4X32
#define CU_RES_VIEW_FORMAT_NONE          GPU_RES_VIEW_FORMAT_NONE
#define CU_RES_VIEW_FORMAT_SIGNED_BC4    GPU_RES_VIEW_FORMAT_SIGNED_BC4
#define CU_RES_VIEW_FORMAT_SIGNED_BC5    GPU_RES_VIEW_FORMAT_SIGNED_BC5
#define CU_RES_VIEW_FORMAT_SIGNED_BC6H   GPU_RES_VIEW_FORMAT_SIGNED_BC6H
#define CU_RES_VIEW_FORMAT_SINT_1X16     GPU_RES_VIEW_FORMAT_SINT_1X16
#define CU_RES_VIEW_FORMAT_SINT_1X32     GPU_RES_VIEW_FORMAT_SINT_1X32
#define CU_RES_VIEW_FORMAT_SINT_1X8      GPU_RES_VIEW_FORMAT_SINT_1X8
#define CU_RES_VIEW_FORMAT_SINT_2X16     GPU_RES_VIEW_FORMAT_SINT_2X16
#define CU_RES_VIEW_FORMAT_SINT_2X32     GPU_RES_VIEW_FORMAT_SINT_2X32
#define CU_RES_VIEW_FORMAT_SINT_2X8      GPU_RES_VIEW_FORMAT_SINT_2X8
#define CU_RES_VIEW_FORMAT_SINT_4X16     GPU_RES_VIEW_FORMAT_SINT_4X16
#define CU_RES_VIEW_FORMAT_SINT_4X32     GPU_RES_VIEW_FORMAT_SINT_4X32
#define CU_RES_VIEW_FORMAT_SINT_4X8      GPU_RES_VIEW_FORMAT_SINT_4X8
#define CU_RES_VIEW_FORMAT_UINT_1X16     GPU_RES_VIEW_FORMAT_UINT_1X16
#define CU_RES_VIEW_FORMAT_UINT_1X32     GPU_RES_VIEW_FORMAT_UINT_1X32
#define CU_RES_VIEW_FORMAT_UINT_1X8      GPU_RES_VIEW_FORMAT_UINT_1X8
#define CU_RES_VIEW_FORMAT_UINT_2X16     GPU_RES_VIEW_FORMAT_UINT_2X16
#define CU_RES_VIEW_FORMAT_UINT_2X32     GPU_RES_VIEW_FORMAT_UINT_2X32
#define CU_RES_VIEW_FORMAT_UINT_2X8      GPU_RES_VIEW_FORMAT_UINT_2X8
#define CU_RES_VIEW_FORMAT_UINT_4X16     GPU_RES_VIEW_FORMAT_UINT_4X16
#define CU_RES_VIEW_FORMAT_UINT_4X32     GPU_RES_VIEW_FORMAT_UINT_4X32
#define CU_RES_VIEW_FORMAT_UINT_4X8      GPU_RES_VIEW_FORMAT_UINT_4X8
#define CU_RES_VIEW_FORMAT_UNSIGNED_BC1  GPU_RES_VIEW_FORMAT_UNSIGNED_BC1
#define CU_RES_VIEW_FORMAT_UNSIGNED_BC2  GPU_RES_VIEW_FORMAT_UNSIGNED_BC2
#define CU_RES_VIEW_FORMAT_UNSIGNED_BC3  GPU_RES_VIEW_FORMAT_UNSIGNED_BC3
#define CU_RES_VIEW_FORMAT_UNSIGNED_BC4  GPU_RES_VIEW_FORMAT_UNSIGNED_BC4
#define CU_RES_VIEW_FORMAT_UNSIGNED_BC5  GPU_RES_VIEW_FORMAT_UNSIGNED_BC5
#define CU_RES_VIEW_FORMAT_UNSIGNED_BC6H GPU_RES_VIEW_FORMAT_UNSIGNED_BC6H
#define CU_RES_VIEW_FORMAT_UNSIGNED_BC7  GPU_RES_VIEW_FORMAT_UNSIGNED_BC7
#define CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE  \
        gpuSharedMemBankSizeDefault
#define CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE  \
        gpuSharedMemBankSizeEightByte
#define CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE  \
        gpuSharedMemBankSizeFourByte
#define CU_STREAM_ADD_CAPTURE_DEPENDENCIES  \
        gpuStreamAddCaptureDependencies
#define CU_STREAM_CAPTURE_MODE_GLOBAL    gpuStreamCaptureModeGlobal
#define CU_STREAM_CAPTURE_MODE_RELAXED   gpuStreamCaptureModeRelaxed
#define CU_STREAM_CAPTURE_MODE_THREAD_LOCAL  \
        gpuStreamCaptureModeThreadLocal
#define CU_STREAM_CAPTURE_STATUS_ACTIVE  gpuStreamCaptureStatusActive
#define CU_STREAM_CAPTURE_STATUS_INVALIDATED  \
        gpuStreamCaptureStatusInvalidated
#define CU_STREAM_CAPTURE_STATUS_NONE    gpuStreamCaptureStatusNone
#define CU_STREAM_DEFAULT                gpuStreamDefault
#define CU_STREAM_NON_BLOCKING           gpuStreamNonBlocking
#define CU_STREAM_PER_THREAD             gpuStreamPerThread
#define CU_STREAM_SET_CAPTURE_DEPENDENCIES  \
        gpuStreamSetCaptureDependencies
#define CU_STREAM_WAIT_VALUE_AND         gpuStreamWaitValueAnd
#define CU_STREAM_WAIT_VALUE_EQ          gpuStreamWaitValueEq
#define CU_STREAM_WAIT_VALUE_GEQ         gpuStreamWaitValueGte
#define CU_STREAM_WAIT_VALUE_NOR         gpuStreamWaitValueNor
#define CU_TRSA_OVERRIDE_FORMAT          GPU_TRSA_OVERRIDE_FORMAT
#define CU_TRSF_NORMALIZED_COORDINATES   GPU_TRSF_NORMALIZED_COORDINATES
#define CU_TRSF_READ_AS_INTEGER          GPU_TRSF_READ_AS_INTEGER
#define CU_TRSF_SRGB                     GPU_TRSF_SRGB
#define CU_TR_ADDRESS_MODE_BORDER        GPU_TR_ADDRESS_MODE_BORDER
#define CU_TR_ADDRESS_MODE_CLAMP         GPU_TR_ADDRESS_MODE_CLAMP
#define CU_TR_ADDRESS_MODE_MIRROR        GPU_TR_ADDRESS_MODE_MIRROR
#define CU_TR_ADDRESS_MODE_WRAP          GPU_TR_ADDRESS_MODE_WRAP
#define CU_TR_FILTER_MODE_LINEAR         GPU_TR_FILTER_MODE_LINEAR
#define CU_TR_FILTER_MODE_POINT          GPU_TR_FILTER_MODE_POINT
#define CU_USER_OBJECT_NO_DESTRUCTOR_SYNC  \
        gpuUserObjectNoDestructorSync
#define CUaccessPolicyWindow             gpuAccessPolicyWindow
#define CUaccessPolicyWindow_st          gpuAccessPolicyWindow
#define CUaccessProperty                 gpuAccessProperty
#define CUaccessProperty_enum            gpuAccessProperty
#define CUaddress_mode                   GPUaddress_mode
#define CUaddress_mode_enum              GPUaddress_mode_enum
#define CUarray                          gpuArray_t
#define CUarrayMapInfo                   gpuArrayMapInfo
#define CUarrayMapInfo_st                gpuArrayMapInfo
#define CUarrayMapInfo_v1                gpuArrayMapInfo
#define CUarraySparseSubresourceType     gpuArraySparseSubresourceType
#define CUarraySparseSubresourceType_enum  \
        gpuArraySparseSubresourceType
#define CUarray_format                   gpuArray_Format
#define CUarray_format_enum              gpuArray_Format
#define CUarray_st                       gpuArray
#define CUcomputemode                    gpuComputeMode
#define CUcomputemode_enum               gpuComputeMode
#define CUcontext                        gpuCtx_t
#define CUdevice                         gpuDevice_t
#define CUdevice_P2PAttribute            gpuDeviceP2PAttr
#define CUdevice_P2PAttribute_enum       gpuDeviceP2PAttr
#define CUdevice_attribute               gpuDeviceAttribute_t
#define CUdevice_attribute_enum          gpuDeviceAttribute_t
#define CUdevice_v1                      gpuDevice_t
#define CUdeviceptr                      gpuDeviceptr_t
#define CUdeviceptr_v1                   gpuDeviceptr_t
#define CUdeviceptr_v2                   gpuDeviceptr_t
#define CUevent                          gpuEvent_t
#define CUexternalMemory                 gpuExternalMemory_t
#define CUexternalMemoryHandleType       gpuExternalMemoryHandleType
#define CUexternalMemoryHandleType_enum  gpuExternalMemoryHandleType_enum
#define CUexternalSemaphore              gpuExternalSemaphore_t
#define CUexternalSemaphoreHandleType    gpuExternalSemaphoreHandleType
#define CUexternalSemaphoreHandleType_enum  \
        gpuExternalSemaphoreHandleType_enum
#define CUfilter_mode                    GPUfilter_mode
#define CUfilter_mode_enum               GPUfilter_mode_enum
#define CUfunc_cache                     gpuFuncCache_t
#define CUfunc_cache_enum                gpuFuncCache_t
#define CUfunction                       gpuFunction_t
#define CUfunction_attribute             gpuFunction_attribute
#define CUfunction_attribute_enum        gpuFunction_attribute
#define CUgraph                          gpuGraph_t
#define CUgraphDebugDot_flags            gpuGraphDebugDotFlags
#define CUgraphDebugDot_flags_enum       gpuGraphDebugDotFlags
#define CUgraphExec                      gpuGraphExec_t
#define CUgraphExecUpdateResult          gpuGraphExecUpdateResult
#define CUgraphExecUpdateResult_enum     gpuGraphExecUpdateResult
#define CUgraphInstantiate_flags         gpuGraphInstantiateFlags
#define CUgraphInstantiate_flags_enum    gpuGraphInstantiateFlags
#define CUgraphMem_attribute             gpuGraphMemAttributeType
#define CUgraphMem_attribute_enum        gpuGraphMemAttributeType
#define CUgraphNode                      gpuGraphNode_t
#define CUgraphNodeType                  gpuGraphNodeType
#define CUgraphNodeType_enum             gpuGraphNodeType
#define CUgraphicsRegisterFlags          gpuGraphicsRegisterFlags
#define CUgraphicsRegisterFlags_enum     gpuGraphicsRegisterFlags
#define CUgraphicsResource               gpuGraphicsResource_t
#define CUgraphicsResource_st            gpuGraphicsResource
#define CUhostFn                         gpuHostFn_t
#define CUipcEventHandle                 gpuIpcEventHandle_t
#define CUipcEventHandle_st              gpuIpcEventHandle_st
#define CUipcEventHandle_v1              gpuIpcEventHandle_t
#define CUipcMemHandle                   gpuIpcMemHandle_t
#define CUipcMemHandle_st                gpuIpcMemHandle_st
#define CUipcMemHandle_v1                gpuIpcMemHandle_t
#define CUjitInputType                   gpurtcJITInputType
#define CUjitInputType_enum              gpurtcJITInputType
#define CUjit_option                     gpuJitOption
#define CUjit_option_enum                gpuJitOption
#define CUkernelNodeAttrID               gpuKernelNodeAttrID
#define CUkernelNodeAttrID_enum          gpuKernelNodeAttrID
#define CUkernelNodeAttrValue            gpuKernelNodeAttrValue
#define CUkernelNodeAttrValue_union      gpuKernelNodeAttrValue
#define CUkernelNodeAttrValue_v1         gpuKernelNodeAttrValue
#define CUlimit                          gpuLimit_t
#define CUlimit_enum                     gpuLimit_t
#define CUlinkState                      gpurtcLinkState
#define CUmemAccessDesc                  gpuMemAccessDesc
#define CUmemAccessDesc_st               gpuMemAccessDesc
#define CUmemAccessDesc_v1               gpuMemAccessDesc
#define CUmemAccess_flags                gpuMemAccessFlags
#define CUmemAccess_flags_enum           gpuMemAccessFlags
#define CUmemAllocationGranularity_flags gpuMemAllocationGranularity_flags
#define CUmemAllocationGranularity_flags_enum  \
        gpuMemAllocationGranularity_flags
#define CUmemAllocationHandleType        gpuMemAllocationHandleType
#define CUmemAllocationHandleType_enum   gpuMemAllocationHandleType
#define CUmemAllocationProp              gpuMemAllocationProp
#define CUmemAllocationProp_st           gpuMemAllocationProp
#define CUmemAllocationProp_v1           gpuMemAllocationProp
#define CUmemAllocationType              gpuMemAllocationType
#define CUmemAllocationType_enum         gpuMemAllocationType
#define CUmemGenericAllocationHandle     gpuMemGenericAllocationHandle_t
#define CUmemGenericAllocationHandle_v1  gpuMemGenericAllocationHandle_t
#define CUmemHandleType                  gpuMemHandleType
#define CUmemHandleType_enum             gpuMemHandleType
#define CUmemLocation                    gpuMemLocation
#define CUmemLocationType                gpuMemLocationType
#define CUmemLocationType_enum           gpuMemLocationType
#define CUmemLocation_st                 gpuMemLocation
#define CUmemLocation_v1                 gpuMemLocation
#define CUmemOperationType               gpuMemOperationType
#define CUmemOperationType_enum          gpuMemOperationType
#define CUmemPoolProps                   gpuMemPoolProps
#define CUmemPoolProps_st                gpuMemPoolProps
#define CUmemPoolProps_v1                gpuMemPoolProps
#define CUmemPoolPtrExportData           gpuMemPoolPtrExportData
#define CUmemPoolPtrExportData_st        gpuMemPoolPtrExportData
#define CUmemPoolPtrExportData_v1        gpuMemPoolPtrExportData
#define CUmemPool_attribute              gpuMemPoolAttr
#define CUmemPool_attribute_enum         gpuMemPoolAttr
#define CUmem_advise                     gpuMemoryAdvise
#define CUmem_advise_enum                gpuMemoryAdvise
#define CUmem_range_attribute            gpuMemRangeAttribute
#define CUmem_range_attribute_enum       gpuMemRangeAttribute
#define CUmemoryPool                     gpuMemPool_t
#define CUmemorytype                     gpuMemoryType
#define CUmemorytype_enum                gpuMemoryType
#define CUmipmappedArray                 gpuMipmappedArray_t
#define CUmipmappedArray_st              gpuMipmappedArray
#define CUmodule                         gpuModule_t
#define CUpointer_attribute              gpuPointer_attribute
#define CUpointer_attribute_enum         gpuPointer_attribute
#define CUresourceViewFormat             GPUresourceViewFormat
#define CUresourceViewFormat_enum        GPUresourceViewFormat_enum
#define CUresourcetype                   GPUresourcetype
#define CUresourcetype_enum              GPUresourcetype_enum
#define CUresult                         gpuError_t
#define CUsharedconfig                   gpuSharedMemConfig
#define CUsharedconfig_enum              gpuSharedMemConfig
#define CUstream                         gpuStream_t
#define CUstreamCallback                 gpuStreamCallback_t
#define CUstreamCaptureMode              gpuStreamCaptureMode
#define CUstreamCaptureMode_enum         gpuStreamCaptureMode
#define CUstreamCaptureStatus            gpuStreamCaptureStatus
#define CUstreamCaptureStatus_enum       gpuStreamCaptureStatus
#define CUstreamUpdateCaptureDependencies_flags  \
        gpuStreamUpdateCaptureDependenciesFlags
#define CUstreamUpdateCaptureDependencies_flags_enum  \
        gpuStreamUpdateCaptureDependenciesFlags
#define CUsurfObject                     gpuSurfaceObject_t
#define CUsurfObject_v1                  gpuSurfaceObject_t
#define CUtexObject                      gpuTextureObject_t
#define CUtexObject_v1                   gpuTextureObject_t
#define CUtexref                         gpuTexRef
#define CUuserObject                     gpuUserObject_t
#define CUuserObjectRetain_flags         gpuUserObjectRetainFlags
#define CUuserObjectRetain_flags_enum    gpuUserObjectRetainFlags
#define CUuserObject_flags               gpuUserObjectFlags
#define CUuserObject_flags_enum          gpuUserObjectFlags
#define cuArray3DCreate                  gpuArray3DCreate
#define cuArray3DCreate_v2               gpuArray3DCreate
#define cuArray3DGetDescriptor           gpuArray3DGetDescriptor
#define cuArray3DGetDescriptor_v2        gpuArray3DGetDescriptor
#define cuArrayCreate                    gpuArrayCreate
#define cuArrayCreate_v2                 gpuArrayCreate
#define cuArrayDestroy                   gpuArrayDestroy
#define cuArrayGetDescriptor             gpuArrayGetDescriptor
#define cuArrayGetDescriptor_v2          gpuArrayGetDescriptor
#define cuCtxCreate                      gpuCtxCreate
#define cuCtxCreate_v2                   gpuCtxCreate
#define cuCtxDestroy                     gpuCtxDestroy
#define cuCtxDestroy_v2                  gpuCtxDestroy
#define cuCtxDisablePeerAccess           gpuCtxDisablePeerAccess
#define cuCtxEnablePeerAccess            gpuCtxEnablePeerAccess
#define cuCtxGetApiVersion               gpuCtxGetApiVersion
#define cuCtxGetCacheConfig              gpuCtxGetCacheConfig
#define cuCtxGetCurrent                  gpuCtxGetCurrent
#define cuCtxGetDevice                   gpuCtxGetDevice
#define cuCtxGetFlags                    gpuCtxGetFlags
#define cuCtxGetLimit                    gpuDeviceGetLimit
#define cuCtxGetSharedMemConfig          gpuCtxGetSharedMemConfig
#define cuCtxGetStreamPriorityRange      gpuDeviceGetStreamPriorityRange
#define cuCtxPopCurrent                  gpuCtxPopCurrent
#define cuCtxPopCurrent_v2               gpuCtxPopCurrent
#define cuCtxPushCurrent                 gpuCtxPushCurrent
#define cuCtxPushCurrent_v2              gpuCtxPushCurrent
#define cuCtxSetCacheConfig              gpuCtxSetCacheConfig
#define cuCtxSetCurrent                  gpuCtxSetCurrent
#define cuCtxSetLimit                    gpuDeviceSetLimit
#define cuCtxSetSharedMemConfig          gpuCtxSetSharedMemConfig
#define cuCtxSynchronize                 gpuCtxSynchronize
#define cuDestroyExternalMemory          gpuDestroyExternalMemory
#define cuDestroyExternalSemaphore       gpuDestroyExternalSemaphore
#define cuDeviceCanAccessPeer            gpuDeviceCanAccessPeer
#define cuDeviceComputeCapability        gpuDeviceComputeCapability
#define cuDeviceGet                      gpuDeviceGet
#define cuDeviceGetAttribute             gpuDeviceGetAttribute
#define cuDeviceGetByPCIBusId            gpuDeviceGetByPCIBusId
#define cuDeviceGetCount                 gpuGetDeviceCount
#define cuDeviceGetDefaultMemPool        gpuDeviceGetDefaultMemPool
#define cuDeviceGetGraphMemAttribute     gpuDeviceGetGraphMemAttribute
#define cuDeviceGetMemPool               gpuDeviceGetMemPool
#define cuDeviceGetName                  gpuDeviceGetName
#define cuDeviceGetP2PAttribute          gpuDeviceGetP2PAttribute
#define cuDeviceGetPCIBusId              gpuDeviceGetPCIBusId
#define cuDeviceGetUuid                  gpuDeviceGetUuid
#define cuDeviceGetUuid_v2               gpuDeviceGetUuid
#define cuDeviceGraphMemTrim             gpuDeviceGraphMemTrim
#define cuDevicePrimaryCtxGetState       gpuDevicePrimaryCtxGetState
#define cuDevicePrimaryCtxRelease        gpuDevicePrimaryCtxRelease
#define cuDevicePrimaryCtxRelease_v2     gpuDevicePrimaryCtxRelease
#define cuDevicePrimaryCtxReset          gpuDevicePrimaryCtxReset
#define cuDevicePrimaryCtxReset_v2       gpuDevicePrimaryCtxReset
#define cuDevicePrimaryCtxRetain         gpuDevicePrimaryCtxRetain
#define cuDevicePrimaryCtxSetFlags       gpuDevicePrimaryCtxSetFlags
#define cuDevicePrimaryCtxSetFlags_v2    gpuDevicePrimaryCtxSetFlags
#define cuDeviceSetGraphMemAttribute     gpuDeviceSetGraphMemAttribute
#define cuDeviceSetMemPool               gpuDeviceSetMemPool
#define cuDeviceTotalMem                 gpuDeviceTotalMem
#define cuDeviceTotalMem_v2              gpuDeviceTotalMem
#define cuDriverGetVersion               gpuDriverGetVersion
#define cuEventCreate                    gpuEventCreateWithFlags
#define cuEventDestroy                   gpuEventDestroy
#define cuEventDestroy_v2                gpuEventDestroy
#define cuEventElapsedTime               gpuEventElapsedTime
#define cuEventQuery                     gpuEventQuery
#define cuEventRecord                    gpuEventRecord
#define cuEventSynchronize               gpuEventSynchronize
#define cuExternalMemoryGetMappedBuffer  gpuExternalMemoryGetMappedBuffer
#define cuFuncGetAttribute               gpuFuncGetAttribute
#define cuGetErrorName                   gpuDrvGetErrorName
#define cuGetErrorString                 gpuDrvGetErrorString
#define cuGraphAddChildGraphNode         gpuGraphAddChildGraphNode
#define cuGraphAddDependencies           gpuGraphAddDependencies
#define cuGraphAddEmptyNode              gpuGraphAddEmptyNode
#define cuGraphAddEventRecordNode        gpuGraphAddEventRecordNode
#define cuGraphAddEventWaitNode          gpuGraphAddEventWaitNode
#define cuGraphAddHostNode               gpuGraphAddHostNode
#define cuGraphAddKernelNode             gpuGraphAddKernelNode
#define cuGraphAddMemAllocNode           gpuGraphAddMemAllocNode
#define cuGraphAddMemFreeNode            gpuGraphAddMemFreeNode
#define cuGraphChildGraphNodeGetGraph    gpuGraphChildGraphNodeGetGraph
#define cuGraphClone                     gpuGraphClone
#define cuGraphCreate                    gpuGraphCreate
#define cuGraphDebugDotPrint             gpuGraphDebugDotPrint
#define cuGraphDestroy                   gpuGraphDestroy
#define cuGraphDestroyNode               gpuGraphDestroyNode
#define cuGraphEventRecordNodeGetEvent   gpuGraphEventRecordNodeGetEvent
#define cuGraphEventRecordNodeSetEvent   gpuGraphEventRecordNodeSetEvent
#define cuGraphEventWaitNodeGetEvent     gpuGraphEventWaitNodeGetEvent
#define cuGraphEventWaitNodeSetEvent     gpuGraphEventWaitNodeSetEvent
#define cuGraphExecChildGraphNodeSetParams  \
        gpuGraphExecChildGraphNodeSetParams
#define cuGraphExecDestroy               gpuGraphExecDestroy
#define cuGraphExecEventRecordNodeSetEvent  \
        gpuGraphExecEventRecordNodeSetEvent
#define cuGraphExecEventWaitNodeSetEvent gpuGraphExecEventWaitNodeSetEvent
#define cuGraphExecHostNodeSetParams     gpuGraphExecHostNodeSetParams
#define cuGraphExecKernelNodeSetParams   gpuGraphExecKernelNodeSetParams
#define cuGraphExecUpdate                gpuGraphExecUpdate
#define cuGraphGetEdges                  gpuGraphGetEdges
#define cuGraphGetNodes                  gpuGraphGetNodes
#define cuGraphGetRootNodes              gpuGraphGetRootNodes
#define cuGraphHostNodeGetParams         gpuGraphHostNodeGetParams
#define cuGraphHostNodeSetParams         gpuGraphHostNodeSetParams
#define cuGraphInstantiate               gpuGraphInstantiate
#define cuGraphInstantiateWithFlags      gpuGraphInstantiateWithFlags
#define cuGraphInstantiate_v2            gpuGraphInstantiate
#define cuGraphKernelNodeCopyAttributes  gpuGraphKernelNodeCopyAttributes
#define cuGraphKernelNodeGetAttribute    gpuGraphKernelNodeGetAttribute
#define cuGraphKernelNodeGetParams       gpuGraphKernelNodeGetParams
#define cuGraphKernelNodeSetAttribute    gpuGraphKernelNodeSetAttribute
#define cuGraphKernelNodeSetParams       gpuGraphKernelNodeSetParams
#define cuGraphLaunch                    gpuGraphLaunch
#define cuGraphMemAllocNodeGetParams     gpuGraphMemAllocNodeGetParams
#define cuGraphMemFreeNodeGetParams      gpuGraphMemFreeNodeGetParams
#define cuGraphMemcpyNodeGetParams       gpuGraphMemcpyNodeGetParams
#define cuGraphMemcpyNodeSetParams       gpuGraphMemcpyNodeSetParams
#define cuGraphMemsetNodeGetParams       gpuGraphMemsetNodeGetParams
#define cuGraphMemsetNodeSetParams       gpuGraphMemsetNodeSetParams
#define cuGraphNodeFindInClone           gpuGraphNodeFindInClone
#define cuGraphNodeGetDependencies       gpuGraphNodeGetDependencies
#define cuGraphNodeGetDependentNodes     gpuGraphNodeGetDependentNodes
#define cuGraphNodeGetEnabled            gpuGraphNodeGetEnabled
#define cuGraphNodeGetType               gpuGraphNodeGetType
#define cuGraphNodeSetEnabled            gpuGraphNodeSetEnabled
#define cuGraphReleaseUserObject         gpuGraphReleaseUserObject
#define cuGraphRemoveDependencies        gpuGraphRemoveDependencies
#define cuGraphRetainUserObject          gpuGraphRetainUserObject
#define cuGraphUpload                    gpuGraphUpload
#define cuGraphicsMapResources           gpuGraphicsMapResources
#define cuGraphicsResourceGetMappedPointer  \
        gpuGraphicsResourceGetMappedPointer
#define cuGraphicsResourceGetMappedPointer_v2  \
        gpuGraphicsResourceGetMappedPointer
#define cuGraphicsSubResourceGetMappedArray  \
        gpuGraphicsSubResourceGetMappedArray
#define cuGraphicsUnmapResources         gpuGraphicsUnmapResources
#define cuGraphicsUnregisterResource     gpuGraphicsUnregisterResource
#define cuImportExternalMemory           gpuImportExternalMemory
#define cuImportExternalSemaphore        gpuImportExternalSemaphore
#define cuInit                           gpuInit
#define cuIpcCloseMemHandle              gpuIpcCloseMemHandle
#define cuIpcGetEventHandle              gpuIpcGetEventHandle
#define cuIpcGetMemHandle                gpuIpcGetMemHandle
#define cuIpcOpenEventHandle             gpuIpcOpenEventHandle
#define cuIpcOpenMemHandle               gpuIpcOpenMemHandle
#define cuLaunchCooperativeKernel        gpuModuleLaunchCooperativeKernel
#define cuLaunchCooperativeKernelMultiDevice  \
        gpuModuleLaunchCooperativeKernelMultiDevice
#define cuLaunchHostFunc                 gpuLaunchHostFunc
#define cuLaunchKernel                   gpuModuleLaunchKernel
#define cuLinkAddData                    gpurtcLinkAddData
#define cuLinkAddData_v2                 gpurtcLinkAddData
#define cuLinkAddFile                    gpurtcLinkAddFile
#define cuLinkAddFile_v2                 gpurtcLinkAddFile
#define cuLinkComplete                   gpurtcLinkComplete
#define cuLinkCreate                     gpurtcLinkCreate
#define cuLinkCreate_v2                  gpurtcLinkCreate
#define cuLinkDestroy                    gpurtcLinkDestroy
#define cuMemAddressFree                 gpuMemAddressFree
#define cuMemAddressReserve              gpuMemAddressReserve
#define cuMemAdvise                      gpuMemAdvise
#define cuMemAlloc                       gpuMalloc
#define cuMemAllocAsync                  gpuMallocAsync
#define cuMemAllocFromPoolAsync          gpuMallocFromPoolAsync
#define cuMemAllocHost                   gpuMemAllocHost
#define cuMemAllocHost_v2                gpuMemAllocHost
#define cuMemAllocManaged                gpuMallocManaged
#define cuMemAllocPitch                  gpuMemAllocPitch
#define cuMemAllocPitch_v2               gpuMemAllocPitch
#define cuMemAlloc_v2                    gpuMalloc
#define cuMemCreate                      gpuMemCreate
#define cuMemExportToShareableHandle     gpuMemExportToShareableHandle
#define cuMemFree                        gpuFree
#define cuMemFreeAsync                   gpuFreeAsync
#define cuMemFreeHost                    gpuHostFree
#define cuMemFree_v2                     gpuFree
#define cuMemGetAccess                   gpuMemGetAccess
#define cuMemGetAddressRange             gpuMemGetAddressRange
#define cuMemGetAddressRange_v2          gpuMemGetAddressRange
#define cuMemGetAllocationGranularity    gpuMemGetAllocationGranularity
#define cuMemGetAllocationPropertiesFromHandle  \
        gpuMemGetAllocationPropertiesFromHandle
#define cuMemGetInfo                     gpuMemGetInfo
#define cuMemGetInfo_v2                  gpuMemGetInfo
#define cuMemHostAlloc                   gpuHostAlloc
#define cuMemHostGetDevicePointer        gpuHostGetDevicePointer
#define cuMemHostGetDevicePointer_v2     gpuHostGetDevicePointer
#define cuMemHostGetFlags                gpuHostGetFlags
#define cuMemHostRegister                gpuHostRegister
#define cuMemHostRegister_v2             gpuHostRegister
#define cuMemHostUnregister              gpuHostUnregister
#define cuMemImportFromShareableHandle   gpuMemImportFromShareableHandle
#define cuMemMap                         gpuMemMap
#define cuMemMapArrayAsync               gpuMemMapArrayAsync
#define cuMemPoolCreate                  gpuMemPoolCreate
#define cuMemPoolDestroy                 gpuMemPoolDestroy
#define cuMemPoolExportPointer           gpuMemPoolExportPointer
#define cuMemPoolExportToShareableHandle gpuMemPoolExportToShareableHandle
#define cuMemPoolGetAccess               gpuMemPoolGetAccess
#define cuMemPoolGetAttribute            gpuMemPoolGetAttribute
#define cuMemPoolImportFromShareableHandle  \
        gpuMemPoolImportFromShareableHandle
#define cuMemPoolImportPointer           gpuMemPoolImportPointer
#define cuMemPoolSetAccess               gpuMemPoolSetAccess
#define cuMemPoolSetAttribute            gpuMemPoolSetAttribute
#define cuMemPoolTrimTo                  gpuMemPoolTrimTo
#define cuMemPrefetchAsync               gpuMemPrefetchAsync
#define cuMemRangeGetAttribute           gpuMemRangeGetAttribute
#define cuMemRangeGetAttributes          gpuMemRangeGetAttributes
#define cuMemRelease                     gpuMemRelease
#define cuMemRetainAllocationHandle      gpuMemRetainAllocationHandle
#define cuMemSetAccess                   gpuMemSetAccess
#define cuMemUnmap                       gpuMemUnmap
#define cuMemcpy2D                       gpuMemcpyParam2D
#define cuMemcpy2DAsync                  gpuMemcpyParam2DAsync
#define cuMemcpy2DAsync_v2               gpuMemcpyParam2DAsync
#define cuMemcpy2DUnaligned              gpuDrvMemcpy2DUnaligned
#define cuMemcpy2DUnaligned_v2           gpuDrvMemcpy2DUnaligned
#define cuMemcpy2D_v2                    gpuMemcpyParam2D
#define cuMemcpy3D                       gpuDrvMemcpy3D
#define cuMemcpy3DAsync                  gpuDrvMemcpy3DAsync
#define cuMemcpy3DAsync_v2               gpuDrvMemcpy3DAsync
#define cuMemcpy3D_v2                    gpuDrvMemcpy3D
#define cuMemcpyAtoH                     gpuMemcpyAtoH
#define cuMemcpyAtoH_v2                  gpuMemcpyAtoH
#define cuMemcpyDtoD                     gpuMemcpyDtoD
#define cuMemcpyDtoDAsync                gpuMemcpyDtoDAsync
#define cuMemcpyDtoDAsync_v2             gpuMemcpyDtoDAsync
#define cuMemcpyDtoD_v2                  gpuMemcpyDtoD
#define cuMemcpyDtoH                     gpuMemcpyDtoH
#define cuMemcpyDtoHAsync                gpuMemcpyDtoHAsync
#define cuMemcpyDtoHAsync_v2             gpuMemcpyDtoHAsync
#define cuMemcpyDtoH_v2                  gpuMemcpyDtoH
#define cuMemcpyHtoA                     gpuMemcpyHtoA
#define cuMemcpyHtoA_v2                  gpuMemcpyHtoA
#define cuMemcpyHtoD                     gpuMemcpyHtoD
#define cuMemcpyHtoDAsync                gpuMemcpyHtoDAsync
#define cuMemcpyHtoDAsync_v2             gpuMemcpyHtoDAsync
#define cuMemcpyHtoD_v2                  gpuMemcpyHtoD
#define cuMemsetD16                      gpuMemsetD16
#define cuMemsetD16Async                 gpuMemsetD16Async
#define cuMemsetD16_v2                   gpuMemsetD16
#define cuMemsetD32                      gpuMemsetD32
#define cuMemsetD32Async                 gpuMemsetD32Async
#define cuMemsetD32_v2                   gpuMemsetD32
#define cuMemsetD8                       gpuMemsetD8
#define cuMemsetD8Async                  gpuMemsetD8Async
#define cuMemsetD8_v2                    gpuMemsetD8
#define cuMipmappedArrayCreate           gpuMipmappedArrayCreate
#define cuMipmappedArrayDestroy          gpuMipmappedArrayDestroy
#define cuMipmappedArrayGetLevel         gpuMipmappedArrayGetLevel
#define cuModuleGetFunction              gpuModuleGetFunction
#define cuModuleGetGlobal                gpuModuleGetGlobal
#define cuModuleGetGlobal_v2             gpuModuleGetGlobal
#define cuModuleGetTexRef                gpuModuleGetTexRef
#define cuModuleLoad                     gpuModuleLoad
#define cuModuleLoadData                 gpuModuleLoadData
#define cuModuleLoadDataEx               gpuModuleLoadDataEx
#define cuModuleUnload                   gpuModuleUnload
#define cuOccupancyMaxActiveBlocksPerMultiprocessor  \
        gpuModuleOccupancyMaxActiveBlocksPerMultiprocessor
#define cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags  \
        gpuModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define cuOccupancyMaxPotentialBlockSize gpuModuleOccupancyMaxPotentialBlockSize
#define cuOccupancyMaxPotentialBlockSizeWithFlags  \
        gpuModuleOccupancyMaxPotentialBlockSizeWithFlags
#define cuPointerGetAttribute            gpuPointerGetAttribute
#define cuPointerGetAttributes           gpuDrvPointerGetAttributes
#define cuPointerSetAttribute            gpuPointerSetAttribute
#define cuSignalExternalSemaphoresAsync  gpuSignalExternalSemaphoresAsync
#define cuStreamAddCallback              gpuStreamAddCallback
#define cuStreamAttachMemAsync           gpuStreamAttachMemAsync
#define cuStreamBeginCapture             gpuStreamBeginCapture
#define cuStreamBeginCapture_v2          gpuStreamBeginCapture
#define cuStreamCreate                   gpuStreamCreateWithFlags
#define cuStreamCreateWithPriority       gpuStreamCreateWithPriority
#define cuStreamDestroy                  gpuStreamDestroy
#define cuStreamDestroy_v2               gpuStreamDestroy
#define cuStreamEndCapture               gpuStreamEndCapture
#define cuStreamGetCaptureInfo           gpuStreamGetCaptureInfo
#define cuStreamGetCaptureInfo_v2        gpuStreamGetCaptureInfo_v2
#define cuStreamGetFlags                 gpuStreamGetFlags
#define cuStreamGetPriority              gpuStreamGetPriority
#define cuStreamIsCapturing              gpuStreamIsCapturing
#define cuStreamQuery                    gpuStreamQuery
#define cuStreamSynchronize              gpuStreamSynchronize
#define cuStreamUpdateCaptureDependencies  \
        gpuStreamUpdateCaptureDependencies
#define cuStreamWaitEvent                gpuStreamWaitEvent
#define cuStreamWaitValue32              gpuStreamWaitValue32
#define cuStreamWaitValue32_v2           gpuStreamWaitValue32
#define cuStreamWaitValue64              gpuStreamWaitValue64
#define cuStreamWaitValue64_v2           gpuStreamWaitValue64
#define cuStreamWriteValue32             gpuStreamWriteValue32
#define cuStreamWriteValue32_v2          gpuStreamWriteValue32
#define cuStreamWriteValue64             gpuStreamWriteValue64
#define cuStreamWriteValue64_v2          gpuStreamWriteValue64
#define cuTexObjectCreate                gpuTexObjectCreate
#define cuTexObjectDestroy               gpuTexObjectDestroy
#define cuTexObjectGetResourceDesc       gpuTexObjectGetResourceDesc
#define cuTexObjectGetResourceViewDesc   gpuTexObjectGetResourceViewDesc
#define cuTexObjectGetTextureDesc        gpuTexObjectGetTextureDesc
#define cuTexRefGetAddress               gpuTexRefGetAddress
#define cuTexRefGetAddressMode           gpuTexRefGetAddressMode
#define cuTexRefGetAddress_v2            gpuTexRefGetAddress
#define cuTexRefGetFilterMode            gpuTexRefGetFilterMode
#define cuTexRefGetFlags                 gpuTexRefGetFlags
#define cuTexRefGetFormat                gpuTexRefGetFormat
#define cuTexRefGetMaxAnisotropy         gpuTexRefGetMaxAnisotropy
#define cuTexRefGetMipmapFilterMode      gpuTexRefGetMipmapFilterMode
#define cuTexRefGetMipmapLevelBias       gpuTexRefGetMipmapLevelBias
#define cuTexRefGetMipmapLevelClamp      gpuTexRefGetMipmapLevelClamp
#define cuTexRefGetMipmappedArray        gpuTexRefGetMipMappedArray
#define cuTexRefSetAddress               gpuTexRefSetAddress
#define cuTexRefSetAddress2D             gpuTexRefSetAddress2D
#define cuTexRefSetAddress2D_v2          gpuTexRefSetAddress2D
#define cuTexRefSetAddress2D_v3          gpuTexRefSetAddress2D
#define cuTexRefSetAddressMode           gpuTexRefSetAddressMode
#define cuTexRefSetAddress_v2            gpuTexRefSetAddress
#define cuTexRefSetArray                 gpuTexRefSetArray
#define cuTexRefSetBorderColor           gpuTexRefSetBorderColor
#define cuTexRefSetFilterMode            gpuTexRefSetFilterMode
#define cuTexRefSetFlags                 gpuTexRefSetFlags
#define cuTexRefSetFormat                gpuTexRefSetFormat
#define cuTexRefSetMaxAnisotropy         gpuTexRefSetMaxAnisotropy
#define cuTexRefSetMipmapFilterMode      gpuTexRefSetMipmapFilterMode
#define cuTexRefSetMipmapLevelBias       gpuTexRefSetMipmapLevelBias
#define cuTexRefSetMipmapLevelClamp      gpuTexRefSetMipmapLevelClamp
#define cuTexRefSetMipmappedArray        gpuTexRefSetMipmappedArray
#define cuThreadExchangeStreamCaptureMode  \
        gpuThreadExchangeStreamCaptureMode
#define cuUserObjectCreate               gpuUserObjectCreate
#define cuUserObjectRelease              gpuUserObjectRelease
#define cuUserObjectRetain               gpuUserObjectRetain
#define cuWaitExternalSemaphoresAsync    gpuWaitExternalSemaphoresAsync
#define cudaError_enum                   gpuError_t

#include <hop/hop_runtime_api.h>
#include <hop/hoprtc.h>

#endif
