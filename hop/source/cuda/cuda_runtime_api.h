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

#ifndef __HOP_SOURCE_CUDA_CUDA_RUNTIME_API_H__
#define __HOP_SOURCE_CUDA_CUDA_RUNTIME_API_H__

#define HOP_SOURCE_CUDA

#define cudaArrayGetInfo                 gpuArrayGetInfo
#define cudaBindTexture                  gpuBindTexture
#define cudaBindTexture2D                gpuBindTexture2D
#define cudaBindTextureToArray           gpuBindTextureToArray
#define cudaBindTextureToMipmappedArray  gpuBindTextureToMipmappedArray
#define cudaChooseDevice                 gpuChooseDevice
#define cudaCreateChannelDesc            gpuCreateChannelDesc
#define cudaCreateSurfaceObject          gpuCreateSurfaceObject
#define cudaCreateTextureObject          gpuCreateTextureObject
#define cudaDestroyExternalMemory        gpuDestroyExternalMemory
#define cudaDestroyExternalSemaphore     gpuDestroyExternalSemaphore
#define cudaDestroySurfaceObject         gpuDestroySurfaceObject
#define cudaDestroyTextureObject         gpuDestroyTextureObject
#define cudaDeviceCanAccessPeer          gpuDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess      gpuDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess       gpuDeviceEnablePeerAccess
#define cudaDeviceGetAttribute           gpuDeviceGetAttribute
#define cudaDeviceGetByPCIBusId          gpuDeviceGetByPCIBusId
#define cudaDeviceGetCacheConfig         gpuDeviceGetCacheConfig
#define cudaDeviceGetDefaultMemPool      gpuDeviceGetDefaultMemPool
#define cudaDeviceGetGraphMemAttribute   gpuDeviceGetGraphMemAttribute
#define cudaDeviceGetLimit               gpuDeviceGetLimit
#define cudaDeviceGetMemPool             gpuDeviceGetMemPool
#define cudaDeviceGetP2PAttribute        gpuDeviceGetP2PAttribute
#define cudaDeviceGetPCIBusId            gpuDeviceGetPCIBusId
#define cudaDeviceGetSharedMemConfig     gpuDeviceGetSharedMemConfig
#define cudaDeviceGetStreamPriorityRange gpuDeviceGetStreamPriorityRange
#define cudaDeviceGraphMemTrim           gpuDeviceGraphMemTrim
#define cudaDeviceReset                  gpuDeviceReset
#define cudaDeviceSetCacheConfig         gpuDeviceSetCacheConfig
#define cudaDeviceSetGraphMemAttribute   gpuDeviceSetGraphMemAttribute
#define cudaDeviceSetLimit               gpuDeviceSetLimit
#define cudaDeviceSetMemPool             gpuDeviceSetMemPool
#define cudaDeviceSetSharedMemConfig     gpuDeviceSetSharedMemConfig
#define cudaDeviceSynchronize            gpuDeviceSynchronize
#define cudaDriverGetVersion             gpuDriverGetVersion
#define cudaEventCreate                  gpuEventCreate
#define cudaEventCreateWithFlags         gpuEventCreateWithFlags
#define cudaEventDestroy                 gpuEventDestroy
#define cudaEventElapsedTime             gpuEventElapsedTime
#define cudaEventQuery                   gpuEventQuery
#define cudaEventRecord                  gpuEventRecord
#define cudaEventSynchronize             gpuEventSynchronize
#define cudaExternalMemoryGetMappedBuffer  \
        gpuExternalMemoryGetMappedBuffer
#define cudaFree                         gpuFree
#define cudaFreeArray                    gpuFreeArray
#define cudaFreeAsync                    gpuFreeAsync
#define cudaFreeHost                     gpuHostFree
#define cudaFreeMipmappedArray           gpuFreeMipmappedArray
#define cudaFuncGetAttributes            gpuFuncGetAttributes
#define cudaFuncSetAttribute             gpuFuncSetAttribute
#define cudaFuncSetCacheConfig           gpuFuncSetCacheConfig
#define cudaFuncSetSharedMemConfig       gpuFuncSetSharedMemConfig
#define cudaGetChannelDesc               gpuGetChannelDesc
#define cudaGetDevice                    gpuGetDevice
#define cudaGetDeviceCount               gpuGetDeviceCount
#define cudaGetDeviceFlags               gpuGetDeviceFlags
#define cudaGetDeviceProperties          gpuGetDeviceProperties
#define cudaGetErrorName                 gpuGetErrorName
#define cudaGetErrorString               gpuGetErrorString
#define cudaGetLastError                 gpuGetLastError
#define cudaGetMipmappedArrayLevel       gpuGetMipmappedArrayLevel
#define cudaGetSymbolAddress             gpuGetSymbolAddress
#define cudaGetSymbolSize                gpuGetSymbolSize
#define cudaGetTextureAlignmentOffset    gpuGetTextureAlignmentOffset
#define cudaGetTextureObjectResourceDesc gpuGetTextureObjectResourceDesc
#define cudaGetTextureObjectResourceViewDesc  \
        gpuGetTextureObjectResourceViewDesc
#define cudaGetTextureObjectTextureDesc  gpuGetTextureObjectTextureDesc
#define cudaGetTextureReference          gpuGetTextureReference
#define cudaGraphAddChildGraphNode       gpuGraphAddChildGraphNode
#define cudaGraphAddDependencies         gpuGraphAddDependencies
#define cudaGraphAddEmptyNode            gpuGraphAddEmptyNode
#define cudaGraphAddEventRecordNode      gpuGraphAddEventRecordNode
#define cudaGraphAddEventWaitNode        gpuGraphAddEventWaitNode
#define cudaGraphAddHostNode             gpuGraphAddHostNode
#define cudaGraphAddKernelNode           gpuGraphAddKernelNode
#define cudaGraphAddMemAllocNode         gpuGraphAddMemAllocNode
#define cudaGraphAddMemFreeNode          gpuGraphAddMemFreeNode
#define cudaGraphAddMemcpyNode           gpuGraphAddMemcpyNode
#define cudaGraphAddMemcpyNode1D         gpuGraphAddMemcpyNode1D
#define cudaGraphAddMemcpyNodeFromSymbol gpuGraphAddMemcpyNodeFromSymbol
#define cudaGraphAddMemcpyNodeToSymbol   gpuGraphAddMemcpyNodeToSymbol
#define cudaGraphAddMemsetNode           gpuGraphAddMemsetNode
#define cudaGraphChildGraphNodeGetGraph  gpuGraphChildGraphNodeGetGraph
#define cudaGraphClone                   gpuGraphClone
#define cudaGraphCreate                  gpuGraphCreate
#define cudaGraphDebugDotPrint           gpuGraphDebugDotPrint
#define cudaGraphDestroy                 gpuGraphDestroy
#define cudaGraphDestroyNode             gpuGraphDestroyNode
#define cudaGraphEventRecordNodeGetEvent gpuGraphEventRecordNodeGetEvent
#define cudaGraphEventRecordNodeSetEvent gpuGraphEventRecordNodeSetEvent
#define cudaGraphEventWaitNodeGetEvent   gpuGraphEventWaitNodeGetEvent
#define cudaGraphEventWaitNodeSetEvent   gpuGraphEventWaitNodeSetEvent
#define cudaGraphExecChildGraphNodeSetParams  \
        gpuGraphExecChildGraphNodeSetParams
#define cudaGraphExecDestroy             gpuGraphExecDestroy
#define cudaGraphExecEventRecordNodeSetEvent  \
        gpuGraphExecEventRecordNodeSetEvent
#define cudaGraphExecEventWaitNodeSetEvent  \
        gpuGraphExecEventWaitNodeSetEvent
#define cudaGraphExecHostNodeSetParams   gpuGraphExecHostNodeSetParams
#define cudaGraphExecKernelNodeSetParams gpuGraphExecKernelNodeSetParams
#define cudaGraphExecMemcpyNodeSetParams gpuGraphExecMemcpyNodeSetParams
#define cudaGraphExecMemcpyNodeSetParams1D  \
        gpuGraphExecMemcpyNodeSetParams1D
#define cudaGraphExecMemcpyNodeSetParamsFromSymbol  \
        gpuGraphExecMemcpyNodeSetParamsFromSymbol
#define cudaGraphExecMemcpyNodeSetParamsToSymbol  \
        gpuGraphExecMemcpyNodeSetParamsToSymbol
#define cudaGraphExecMemsetNodeSetParams gpuGraphExecMemsetNodeSetParams
#define cudaGraphExecUpdate              gpuGraphExecUpdate
#define cudaGraphGetEdges                gpuGraphGetEdges
#define cudaGraphGetNodes                gpuGraphGetNodes
#define cudaGraphGetRootNodes            gpuGraphGetRootNodes
#define cudaGraphHostNodeGetParams       gpuGraphHostNodeGetParams
#define cudaGraphHostNodeSetParams       gpuGraphHostNodeSetParams
#define cudaGraphInstantiate             gpuGraphInstantiate
#define cudaGraphInstantiateWithFlags    gpuGraphInstantiateWithFlags
#define cudaGraphKernelNodeCopyAttributes  \
        gpuGraphKernelNodeCopyAttributes
#define cudaGraphKernelNodeGetAttribute  gpuGraphKernelNodeGetAttribute
#define cudaGraphKernelNodeGetParams     gpuGraphKernelNodeGetParams
#define cudaGraphKernelNodeSetAttribute  gpuGraphKernelNodeSetAttribute
#define cudaGraphKernelNodeSetParams     gpuGraphKernelNodeSetParams
#define cudaGraphLaunch                  gpuGraphLaunch
#define cudaGraphMemAllocNodeGetParams   gpuGraphMemAllocNodeGetParams
#define cudaGraphMemFreeNodeGetParams    gpuGraphMemFreeNodeGetParams
#define cudaGraphMemcpyNodeGetParams     gpuGraphMemcpyNodeGetParams
#define cudaGraphMemcpyNodeSetParams     gpuGraphMemcpyNodeSetParams
#define cudaGraphMemcpyNodeSetParams1D   gpuGraphMemcpyNodeSetParams1D
#define cudaGraphMemcpyNodeSetParamsFromSymbol  \
        gpuGraphMemcpyNodeSetParamsFromSymbol
#define cudaGraphMemcpyNodeSetParamsToSymbol  \
        gpuGraphMemcpyNodeSetParamsToSymbol
#define cudaGraphMemsetNodeGetParams     gpuGraphMemsetNodeGetParams
#define cudaGraphMemsetNodeSetParams     gpuGraphMemsetNodeSetParams
#define cudaGraphNodeFindInClone         gpuGraphNodeFindInClone
#define cudaGraphNodeGetDependencies     gpuGraphNodeGetDependencies
#define cudaGraphNodeGetDependentNodes   gpuGraphNodeGetDependentNodes
#define cudaGraphNodeGetEnabled          gpuGraphNodeGetEnabled
#define cudaGraphNodeGetType             gpuGraphNodeGetType
#define cudaGraphNodeSetEnabled          gpuGraphNodeSetEnabled
#define cudaGraphReleaseUserObject       gpuGraphReleaseUserObject
#define cudaGraphRemoveDependencies      gpuGraphRemoveDependencies
#define cudaGraphRetainUserObject        gpuGraphRetainUserObject
#define cudaGraphUpload                  gpuGraphUpload
#define cudaGraphicsMapResources         gpuGraphicsMapResources
#define cudaGraphicsResourceGetMappedPointer  \
        gpuGraphicsResourceGetMappedPointer
#define cudaGraphicsSubResourceGetMappedArray  \
        gpuGraphicsSubResourceGetMappedArray
#define cudaGraphicsUnmapResources       gpuGraphicsUnmapResources
#define cudaGraphicsUnregisterResource   gpuGraphicsUnregisterResource
#define cudaHostAlloc                    gpuHostAlloc
#define cudaHostGetDevicePointer         gpuHostGetDevicePointer
#define cudaHostGetFlags                 gpuHostGetFlags
#define cudaHostRegister                 gpuHostRegister
#define cudaHostUnregister               gpuHostUnregister
#define cudaImportExternalMemory         gpuImportExternalMemory
#define cudaImportExternalSemaphore      gpuImportExternalSemaphore
#define cudaIpcCloseMemHandle            gpuIpcCloseMemHandle
#define cudaIpcGetEventHandle            gpuIpcGetEventHandle
#define cudaIpcGetMemHandle              gpuIpcGetMemHandle
#define cudaIpcOpenEventHandle           gpuIpcOpenEventHandle
#define cudaIpcOpenMemHandle             gpuIpcOpenMemHandle
#define cudaLaunchCooperativeKernel      gpuLaunchCooperativeKernel
#define cudaLaunchCooperativeKernelMultiDevice  \
        gpuLaunchCooperativeKernelMultiDevice
#define cudaLaunchHostFunc               gpuLaunchHostFunc
#define cudaLaunchKernel                 gpuLaunchKernel
#define cudaMalloc                       gpuMalloc
#define cudaMalloc3D                     gpuMalloc3D
#define cudaMalloc3DArray                gpuMalloc3DArray
#define cudaMallocArray                  gpuMallocArray
#define cudaMallocAsync                  gpuMallocAsync
#define cudaMallocFromPoolAsync          gpuMallocFromPoolAsync
#define cudaMallocHost                   gpuHostMalloc
#define cudaMallocManaged                gpuMallocManaged
#define cudaMallocMipmappedArray         gpuMallocMipmappedArray
#define cudaMallocPitch                  gpuMallocPitch
#define cudaMemAdvise                    gpuMemAdvise
#define cudaMemGetInfo                   gpuMemGetInfo
#define cudaMemPoolCreate                gpuMemPoolCreate
#define cudaMemPoolDestroy               gpuMemPoolDestroy
#define cudaMemPoolExportPointer         gpuMemPoolExportPointer
#define cudaMemPoolExportToShareableHandle  \
        gpuMemPoolExportToShareableHandle
#define cudaMemPoolGetAccess             gpuMemPoolGetAccess
#define cudaMemPoolGetAttribute          gpuMemPoolGetAttribute
#define cudaMemPoolImportFromShareableHandle  \
        gpuMemPoolImportFromShareableHandle
#define cudaMemPoolImportPointer         gpuMemPoolImportPointer
#define cudaMemPoolSetAccess             gpuMemPoolSetAccess
#define cudaMemPoolSetAttribute          gpuMemPoolSetAttribute
#define cudaMemPoolTrimTo                gpuMemPoolTrimTo
#define cudaMemPrefetchAsync             gpuMemPrefetchAsync
#define cudaMemRangeGetAttribute         gpuMemRangeGetAttribute
#define cudaMemRangeGetAttributes        gpuMemRangeGetAttributes
#define cudaMemcpy                       gpuMemcpy
#define cudaMemcpy2D                     gpuMemcpy2D
#define cudaMemcpy2DAsync                gpuMemcpy2DAsync
#define cudaMemcpy2DFromArray            gpuMemcpy2DFromArray
#define cudaMemcpy2DFromArrayAsync       gpuMemcpy2DFromArrayAsync
#define cudaMemcpy2DToArray              gpuMemcpy2DToArray
#define cudaMemcpy2DToArrayAsync         gpuMemcpy2DToArrayAsync
#define cudaMemcpy3D                     gpuMemcpy3D
#define cudaMemcpy3DAsync                gpuMemcpy3DAsync
#define cudaMemcpyAsync                  gpuMemcpyAsync
#define cudaMemcpyFromArray              gpuMemcpyFromArray
#define cudaMemcpyFromSymbol             gpuMemcpyFromSymbol
#define cudaMemcpyFromSymbolAsync        gpuMemcpyFromSymbolAsync
#define cudaMemcpyPeer                   gpuMemcpyPeer
#define cudaMemcpyPeerAsync              gpuMemcpyPeerAsync
#define cudaMemcpyToArray                gpuMemcpyToArray
#define cudaMemcpyToSymbol               gpuMemcpyToSymbol
#define cudaMemcpyToSymbolAsync          gpuMemcpyToSymbolAsync
#define cudaMemset                       gpuMemset
#define cudaMemset2D                     gpuMemset2D
#define cudaMemset2DAsync                gpuMemset2DAsync
#define cudaMemset3D                     gpuMemset3D
#define cudaMemset3DAsync                gpuMemset3DAsync
#define cudaMemsetAsync                  gpuMemsetAsync
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor  \
        gpuOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags  \
        gpuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#define cudaPeekAtLastError              gpuPeekAtLastError
#define cudaPointerGetAttributes         gpuPointerGetAttributes
#define cudaRuntimeGetVersion            gpuRuntimeGetVersion
#define cudaSetDevice                    gpuSetDevice
#define cudaSetDeviceFlags               gpuSetDeviceFlags
#define cudaSignalExternalSemaphoresAsync  \
        gpuSignalExternalSemaphoresAsync
#define cudaStreamAddCallback            gpuStreamAddCallback
#define cudaStreamAttachMemAsync         gpuStreamAttachMemAsync
#define cudaStreamBeginCapture           gpuStreamBeginCapture
#define cudaStreamCallback_t             gpuStreamCallback_t
#define cudaStreamCreate                 gpuStreamCreate
#define cudaStreamCreateWithFlags        gpuStreamCreateWithFlags
#define cudaStreamCreateWithPriority     gpuStreamCreateWithPriority
#define cudaStreamDestroy                gpuStreamDestroy
#define cudaStreamEndCapture             gpuStreamEndCapture
#define cudaStreamGetCaptureInfo         gpuStreamGetCaptureInfo
#define cudaStreamGetFlags               gpuStreamGetFlags
#define cudaStreamGetPriority            gpuStreamGetPriority
#define cudaStreamIsCapturing            gpuStreamIsCapturing
#define cudaStreamQuery                  gpuStreamQuery
#define cudaStreamSynchronize            gpuStreamSynchronize
#define cudaStreamWaitEvent              gpuStreamWaitEvent
#define cudaThreadExchangeStreamCaptureMode  \
        gpuThreadExchangeStreamCaptureMode
#define cudaThreadExit                   gpuDeviceReset
#define cudaThreadGetCacheConfig         gpuDeviceGetCacheConfig
#define cudaThreadSetCacheConfig         gpuDeviceSetCacheConfig
#define cudaThreadSynchronize            gpuDeviceSynchronize
#define cudaUnbindTexture                gpuUnbindTexture
#define cudaUserObjectCreate             gpuUserObjectCreate
#define cudaUserObjectRelease            gpuUserObjectRelease
#define cudaUserObjectRetain             gpuUserObjectRetain
#define cudaWaitExternalSemaphoresAsync  gpuWaitExternalSemaphoresAsync

/* cuda_profiler_api.h */
#define cudaProfilerStart                gpuProfilerStart
#define cudaProfilerStop                 gpuProfilerStop

/* cudaGL.h */
#define CUGLDeviceList                   gpuGLDeviceList
#define CUGLDeviceList_enum              gpuGLDeviceList
#define CU_GL_DEVICE_LIST_ALL            gpuGLDeviceListAll
#define CU_GL_DEVICE_LIST_CURRENT_FRAME  gpuGLDeviceListCurrentFrame
#define CU_GL_DEVICE_LIST_NEXT_FRAME     gpuGLDeviceListNextFrame
#define cuGLGetDevices                   gpuGLGetDevices
#define cuGraphicsGLRegisterBuffer       gpuGraphicsGLRegisterBuffer
#define cuGraphicsGLRegisterImage        gpuGraphicsGLRegisterImage

/* driver_functions.h */
#define make_cudaExtent                  make_gpuExtent
#define make_cudaPitchedPtr              make_gpuPitchedPtr
#define make_cudaPos                     make_gpuPos

/* driver_types.h */
#define CUDA_IPC_HANDLE_SIZE             GPU_IPC_HANDLE_SIZE
#define CUuuid                           gpuUUID
#define CUuuid_st                        gpuUUID_t
#define cudaAccessPolicyWindow           gpuAccessPolicyWindow
#define cudaAccessProperty               gpuAccessProperty
#define cudaAccessPropertyNormal         gpuAccessPropertyNormal
#define cudaAccessPropertyPersisting     gpuAccessPropertyPersisting
#define cudaAccessPropertyStreaming      gpuAccessPropertyStreaming
#define cudaArray                        gpuArray
#define cudaArrayCubemap                 gpuArrayCubemap
#define cudaArrayDefault                 gpuArrayDefault
#define cudaArrayLayered                 gpuArrayLayered
#define cudaArraySurfaceLoadStore        gpuArraySurfaceLoadStore
#define cudaArrayTextureGather           gpuArrayTextureGather
#define cudaArray_const_t                gpuArray_const_t
#define cudaArray_t                      gpuArray_t
#define cudaChannelFormatDesc            gpuChannelFormatDesc
#define cudaChannelFormatKind            gpuChannelFormatKind
#define cudaChannelFormatKindFloat       gpuChannelFormatKindFloat
#define cudaChannelFormatKindNone        gpuChannelFormatKindNone
#define cudaChannelFormatKindSigned      gpuChannelFormatKindSigned
#define cudaChannelFormatKindUnsigned    gpuChannelFormatKindUnsigned
#define cudaComputeMode                  gpuComputeMode
#define cudaComputeModeDefault           gpuComputeModeDefault
#define cudaComputeModeExclusive         gpuComputeModeExclusive
#define cudaComputeModeExclusiveProcess  gpuComputeModeExclusiveProcess
#define cudaComputeModeProhibited        gpuComputeModeProhibited
#define cudaCooperativeLaunchMultiDeviceNoPostSync  \
        gpuCooperativeLaunchMultiDeviceNoPostSync
#define cudaCooperativeLaunchMultiDeviceNoPreSync  \
        gpuCooperativeLaunchMultiDeviceNoPreSync
#define cudaCpuDeviceId                  gpuCpuDeviceId
#define cudaDevAttrAsyncEngineCount      gpuDeviceAttributeAsyncEngineCount
#define cudaDevAttrCanMapHostMemory      gpuDeviceAttributeCanMapHostMemory
#define cudaDevAttrCanUseHostPointerForRegisteredMem  \
        gpuDeviceAttributeCanUseHostPointerForRegisteredMem
#define cudaDevAttrClockRate             gpuDeviceAttributeClockRate
#define cudaDevAttrComputeCapabilityMajor  \
        gpuDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor  \
        gpuDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrComputeMode           gpuDeviceAttributeComputeMode
#define cudaDevAttrComputePreemptionSupported  \
        gpuDeviceAttributeComputePreemptionSupported
#define cudaDevAttrConcurrentKernels     gpuDeviceAttributeConcurrentKernels
#define cudaDevAttrConcurrentManagedAccess  \
        gpuDeviceAttributeConcurrentManagedAccess
#define cudaDevAttrCooperativeLaunch     gpuDeviceAttributeCooperativeLaunch
#define cudaDevAttrCooperativeMultiDeviceLaunch  \
        gpuDeviceAttributeCooperativeMultiDeviceLaunch
#define cudaDevAttrDirectManagedMemAccessFromHost  \
        gpuDeviceAttributeDirectManagedMemAccessFromHost
#define cudaDevAttrEccEnabled            gpuDeviceAttributeEccEnabled
#define cudaDevAttrGlobalL1CacheSupported  \
        gpuDeviceAttributeGlobalL1CacheSupported
#define cudaDevAttrGlobalMemoryBusWidth  gpuDeviceAttributeMemoryBusWidth
#define cudaDevAttrGpuOverlap            gpuDeviceAttributeAsyncEngineCount
#define cudaDevAttrHostNativeAtomicSupported  \
        gpuDeviceAttributeHostNativeAtomicSupported
#define cudaDevAttrIntegrated            gpuDeviceAttributeIntegrated
#define cudaDevAttrIsMultiGpuBoard       gpuDeviceAttributeIsMultiGpuBoard
#define cudaDevAttrKernelExecTimeout     gpuDeviceAttributeKernelExecTimeout
#define cudaDevAttrL2CacheSize           gpuDeviceAttributeL2CacheSize
#define cudaDevAttrLocalL1CacheSupported gpuDeviceAttributeLocalL1CacheSupported
#define cudaDevAttrManagedMemory         gpuDeviceAttributeManagedMemory
#define cudaDevAttrMaxBlockDimX          gpuDeviceAttributeMaxBlockDimX
#define cudaDevAttrMaxBlockDimY          gpuDeviceAttributeMaxBlockDimY
#define cudaDevAttrMaxBlockDimZ          gpuDeviceAttributeMaxBlockDimZ
#define cudaDevAttrMaxBlocksPerMultiprocessor  \
        gpuDeviceAttributeMaxBlocksPerMultiProcessor
#define cudaDevAttrMaxGridDimX           gpuDeviceAttributeMaxGridDimX
#define cudaDevAttrMaxGridDimY           gpuDeviceAttributeMaxGridDimY
#define cudaDevAttrMaxGridDimZ           gpuDeviceAttributeMaxGridDimZ
#define cudaDevAttrMaxPitch              gpuDeviceAttributeMaxPitch
#define cudaDevAttrMaxRegistersPerBlock  gpuDeviceAttributeMaxRegistersPerBlock
#define cudaDevAttrMaxRegistersPerMultiprocessor  \
        gpuDeviceAttributeMaxRegistersPerMultiprocessor
#define cudaDevAttrMaxSharedMemoryPerBlock  \
        gpuDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlockOptin  \
        gpuDeviceAttributeSharedMemPerBlockOptin
#define cudaDevAttrMaxSharedMemoryPerMultiprocessor  \
        gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define cudaDevAttrMaxSurface1DLayeredWidth  \
        gpuDeviceAttributeMaxSurface1DLayered
#define cudaDevAttrMaxSurface1DWidth     gpuDeviceAttributeMaxSurface1D
#define cudaDevAttrMaxSurface2DHeight    gpuDeviceAttributeMaxSurface2D
#define cudaDevAttrMaxSurface2DLayeredHeight  \
        gpuDeviceAttributeMaxSurface2DLayered
#define cudaDevAttrMaxSurface2DLayeredWidth  \
        gpuDeviceAttributeMaxSurface2DLayered
#define cudaDevAttrMaxSurface2DWidth     gpuDeviceAttributeMaxSurface2D
#define cudaDevAttrMaxSurface3DDepth     gpuDeviceAttributeMaxSurface3D
#define cudaDevAttrMaxSurface3DHeight    gpuDeviceAttributeMaxSurface3D
#define cudaDevAttrMaxSurface3DWidth     gpuDeviceAttributeMaxSurface3D
#define cudaDevAttrMaxSurfaceCubemapLayeredWidth  \
        gpuDeviceAttributeMaxSurfaceCubemapLayered
#define cudaDevAttrMaxSurfaceCubemapWidth  \
        gpuDeviceAttributeMaxSurfaceCubemap
#define cudaDevAttrMaxTexture1DLayeredWidth  \
        gpuDeviceAttributeMaxTexture1DLayered
#define cudaDevAttrMaxTexture1DLinearWidth  \
        gpuDeviceAttributeMaxTexture1DLinear
#define cudaDevAttrMaxTexture1DMipmappedWidth  \
        gpuDeviceAttributeMaxTexture1DMipmap
#define cudaDevAttrMaxTexture1DWidth     gpuDeviceAttributeMaxTexture1DWidth
#define cudaDevAttrMaxTexture2DGatherHeight  \
        gpuDeviceAttributeMaxTexture2DGather
#define cudaDevAttrMaxTexture2DGatherWidth  \
        gpuDeviceAttributeMaxTexture2DGather
#define cudaDevAttrMaxTexture2DHeight    gpuDeviceAttributeMaxTexture2DHeight
#define cudaDevAttrMaxTexture2DLayeredHeight  \
        gpuDeviceAttributeMaxTexture2DLayered
#define cudaDevAttrMaxTexture2DLayeredWidth  \
        gpuDeviceAttributeMaxTexture2DLayered
#define cudaDevAttrMaxTexture2DLinearHeight  \
        gpuDeviceAttributeMaxTexture2DLinear
#define cudaDevAttrMaxTexture2DLinearPitch  \
        gpuDeviceAttributeMaxTexture2DLinear
#define cudaDevAttrMaxTexture2DLinearWidth  \
        gpuDeviceAttributeMaxTexture2DLinear
#define cudaDevAttrMaxTexture2DMipmappedHeight  \
        gpuDeviceAttributeMaxTexture2DMipmap
#define cudaDevAttrMaxTexture2DMipmappedWidth  \
        gpuDeviceAttributeMaxTexture2DMipmap
#define cudaDevAttrMaxTexture2DWidth     gpuDeviceAttributeMaxTexture2DWidth
#define cudaDevAttrMaxTexture3DDepth     gpuDeviceAttributeMaxTexture3DDepth
#define cudaDevAttrMaxTexture3DDepthAlt  gpuDeviceAttributeMaxTexture3DAlt
#define cudaDevAttrMaxTexture3DHeight    gpuDeviceAttributeMaxTexture3DHeight
#define cudaDevAttrMaxTexture3DHeightAlt gpuDeviceAttributeMaxTexture3DAlt
#define cudaDevAttrMaxTexture3DWidth     gpuDeviceAttributeMaxTexture3DWidth
#define cudaDevAttrMaxTexture3DWidthAlt  gpuDeviceAttributeMaxTexture3DAlt
#define cudaDevAttrMaxTextureCubemapLayeredWidth  \
        gpuDeviceAttributeMaxTextureCubemapLayered
#define cudaDevAttrMaxTextureCubemapWidth  \
        gpuDeviceAttributeMaxTextureCubemap
#define cudaDevAttrMaxThreadsPerBlock    gpuDeviceAttributeMaxThreadsPerBlock
#define cudaDevAttrMaxThreadsPerMultiProcessor  \
        gpuDeviceAttributeMaxThreadsPerMultiProcessor
#define cudaDevAttrMemoryClockRate       gpuDeviceAttributeMemoryClockRate
#define cudaDevAttrMemoryPoolsSupported  gpuDeviceAttributeMemoryPoolsSupported
#define cudaDevAttrMultiGpuBoardGroupID  gpuDeviceAttributeMultiGpuBoardGroupID
#define cudaDevAttrMultiProcessorCount   gpuDeviceAttributeMultiprocessorCount
#define cudaDevAttrPageableMemoryAccess  gpuDeviceAttributePageableMemoryAccess
#define cudaDevAttrPageableMemoryAccessUsesHostPageTables  \
        gpuDeviceAttributePageableMemoryAccessUsesHostPageTables
#define cudaDevAttrPciBusId              gpuDeviceAttributePciBusId
#define cudaDevAttrPciDeviceId           gpuDeviceAttributePciDeviceId
#define cudaDevAttrPciDomainId           gpuDeviceAttributePciDomainID
#define cudaDevAttrReserved94            gpuDeviceAttributeCanUseStreamWaitValue
#define cudaDevAttrSingleToDoublePrecisionPerfRatio  \
        gpuDeviceAttributeSingleToDoublePrecisionPerfRatio
#define cudaDevAttrStreamPrioritiesSupported  \
        gpuDeviceAttributeStreamPrioritiesSupported
#define cudaDevAttrSurfaceAlignment      gpuDeviceAttributeSurfaceAlignment
#define cudaDevAttrTccDriver             gpuDeviceAttributeTccDriver
#define cudaDevAttrTextureAlignment      gpuDeviceAttributeTextureAlignment
#define cudaDevAttrTexturePitchAlignment gpuDeviceAttributeTexturePitchAlignment
#define cudaDevAttrTotalConstantMemory   gpuDeviceAttributeTotalConstantMemory
#define cudaDevAttrUnifiedAddressing     gpuDeviceAttributeUnifiedAddressing
#define cudaDevAttrWarpSize              gpuDeviceAttributeWarpSize
#define cudaDevP2PAttrAccessSupported    gpuDevP2PAttrAccessSupported
#define cudaDevP2PAttrCudaArrayAccessSupported  \
        gpuDevP2PAttrHipArrayAccessSupported
#define cudaDevP2PAttrNativeAtomicSupported  \
        gpuDevP2PAttrNativeAtomicSupported
#define cudaDevP2PAttrPerformanceRank    gpuDevP2PAttrPerformanceRank
#define cudaDeviceAttr                   gpuDeviceAttribute_t
#define cudaDeviceBlockingSync           gpuDeviceScheduleBlockingSync
#define cudaDeviceLmemResizeToMax        gpuDeviceLmemResizeToMax
#define cudaDeviceMapHost                gpuDeviceMapHost
#define cudaDeviceP2PAttr                gpuDeviceP2PAttr
#define cudaDeviceProp                   gpuDeviceProp_t
#define cudaDeviceScheduleAuto           gpuDeviceScheduleAuto
#define cudaDeviceScheduleBlockingSync   gpuDeviceScheduleBlockingSync
#define cudaDeviceScheduleMask           gpuDeviceScheduleMask
#define cudaDeviceScheduleSpin           gpuDeviceScheduleSpin
#define cudaDeviceScheduleYield          gpuDeviceScheduleYield
#define cudaError                        gpuError_t
#define cudaErrorAlreadyAcquired         gpuErrorAlreadyAcquired
#define cudaErrorAlreadyMapped           gpuErrorAlreadyMapped
#define cudaErrorArrayIsMapped           gpuErrorArrayIsMapped
#define cudaErrorAssert                  gpuErrorAssert
#define cudaErrorCapturedEvent           gpuErrorCapturedEvent
#define cudaErrorContextIsDestroyed      gpuErrorContextIsDestroyed
#define cudaErrorCooperativeLaunchTooLarge  \
        gpuErrorCooperativeLaunchTooLarge
#define cudaErrorCudartUnloading         gpuErrorDeinitialized
#define cudaErrorDeviceAlreadyInUse      gpuErrorContextAlreadyInUse
#define cudaErrorDeviceUninitialized     gpuErrorInvalidContext
#define cudaErrorECCUncorrectable        gpuErrorECCNotCorrectable
#define cudaErrorFileNotFound            gpuErrorFileNotFound
#define cudaErrorGraphExecUpdateFailure  gpuErrorGraphExecUpdateFailure
#define cudaErrorHostMemoryAlreadyRegistered  \
        gpuErrorHostMemoryAlreadyRegistered
#define cudaErrorHostMemoryNotRegistered gpuErrorHostMemoryNotRegistered
#define cudaErrorIllegalAddress          gpuErrorIllegalAddress
#define cudaErrorIllegalState            gpuErrorIllegalState
#define cudaErrorInitializationError     gpuErrorNotInitialized
#define cudaErrorInsufficientDriver      gpuErrorInsufficientDriver
#define cudaErrorInvalidConfiguration    gpuErrorInvalidConfiguration
#define cudaErrorInvalidDevice           gpuErrorInvalidDevice
#define cudaErrorInvalidDeviceFunction   gpuErrorInvalidDeviceFunction
#define cudaErrorInvalidDevicePointer    gpuErrorInvalidDevicePointer
#define cudaErrorInvalidGraphicsContext  gpuErrorInvalidGraphicsContext
#define cudaErrorInvalidKernelImage      gpuErrorInvalidImage
#define cudaErrorInvalidMemcpyDirection  gpuErrorInvalidMemcpyDirection
#define cudaErrorInvalidPitchValue       gpuErrorInvalidPitchValue
#define cudaErrorInvalidPtx              gpuErrorInvalidKernelFile
#define cudaErrorInvalidResourceHandle   gpuErrorInvalidHandle
#define cudaErrorInvalidSource           gpuErrorInvalidSource
#define cudaErrorInvalidSymbol           gpuErrorInvalidSymbol
#define cudaErrorInvalidValue            gpuErrorInvalidValue
#define cudaErrorLaunchFailure           gpuErrorLaunchFailure
#define cudaErrorLaunchOutOfResources    gpuErrorLaunchOutOfResources
#define cudaErrorLaunchTimeout           gpuErrorLaunchTimeOut
#define cudaErrorMapBufferObjectFailed   gpuErrorMapFailed
#define cudaErrorMemoryAllocation        gpuErrorOutOfMemory
#define cudaErrorMissingConfiguration    gpuErrorMissingConfiguration
#define cudaErrorNoDevice                gpuErrorNoDevice
#define cudaErrorNoKernelImageForDevice  gpuErrorNoBinaryForGpu
#define cudaErrorNotMapped               gpuErrorNotMapped
#define cudaErrorNotMappedAsArray        gpuErrorNotMappedAsArray
#define cudaErrorNotMappedAsPointer      gpuErrorNotMappedAsPointer
#define cudaErrorNotReady                gpuErrorNotReady
#define cudaErrorNotSupported            gpuErrorNotSupported
#define cudaErrorOperatingSystem         gpuErrorOperatingSystem
#define cudaErrorPeerAccessAlreadyEnabled  \
        gpuErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled    gpuErrorPeerAccessNotEnabled
#define cudaErrorPeerAccessUnsupported   gpuErrorPeerAccessUnsupported
#define cudaErrorPriorLaunchFailure      gpuErrorPriorLaunchFailure
#define cudaErrorProfilerAlreadyStarted  gpuErrorProfilerAlreadyStarted
#define cudaErrorProfilerAlreadyStopped  gpuErrorProfilerAlreadyStopped
#define cudaErrorProfilerDisabled        gpuErrorProfilerDisabled
#define cudaErrorProfilerNotInitialized  gpuErrorProfilerNotInitialized
#define cudaErrorSetOnActiveProcess      gpuErrorSetOnActiveProcess
#define cudaErrorSharedObjectInitFailed  gpuErrorSharedObjectInitFailed
#define cudaErrorSharedObjectSymbolNotFound  \
        gpuErrorSharedObjectSymbolNotFound
#define cudaErrorStreamCaptureImplicit   gpuErrorStreamCaptureImplicit
#define cudaErrorStreamCaptureInvalidated  \
        gpuErrorStreamCaptureInvalidated
#define cudaErrorStreamCaptureIsolation  gpuErrorStreamCaptureIsolation
#define cudaErrorStreamCaptureMerge      gpuErrorStreamCaptureMerge
#define cudaErrorStreamCaptureUnjoined   gpuErrorStreamCaptureUnjoined
#define cudaErrorStreamCaptureUnmatched  gpuErrorStreamCaptureUnmatched
#define cudaErrorStreamCaptureUnsupported  \
        gpuErrorStreamCaptureUnsupported
#define cudaErrorStreamCaptureWrongThread  \
        gpuErrorStreamCaptureWrongThread
#define cudaErrorSymbolNotFound          gpuErrorNotFound
#define cudaErrorUnknown                 gpuErrorUnknown
#define cudaErrorUnmapBufferObjectFailed gpuErrorUnmapFailed
#define cudaErrorUnsupportedLimit        gpuErrorUnsupportedLimit
#define cudaError_t                      gpuError_t
#define cudaEventBlockingSync            gpuEventBlockingSync
#define cudaEventDefault                 gpuEventDefault
#define cudaEventDisableTiming           gpuEventDisableTiming
#define cudaEventInterprocess            gpuEventInterprocess
#define cudaEvent_t                      gpuEvent_t
#define cudaExtent                       gpuExtent
#define cudaExternalMemoryBufferDesc     gpuExternalMemoryBufferDesc
#define cudaExternalMemoryDedicated      gpuExternalMemoryDedicated
#define cudaExternalMemoryHandleDesc     gpuExternalMemoryHandleDesc
#define cudaExternalMemoryHandleType     gpuExternalMemoryHandleType
#define cudaExternalMemoryHandleTypeD3D11Resource  \
        gpuExternalMemoryHandleTypeD3D11Resource
#define cudaExternalMemoryHandleTypeD3D11ResourceKmt  \
        gpuExternalMemoryHandleTypeD3D11ResourceKmt
#define cudaExternalMemoryHandleTypeD3D12Heap  \
        gpuExternalMemoryHandleTypeD3D12Heap
#define cudaExternalMemoryHandleTypeD3D12Resource  \
        gpuExternalMemoryHandleTypeD3D12Resource
#define cudaExternalMemoryHandleTypeOpaqueFd  \
        gpuExternalMemoryHandleTypeOpaqueFd
#define cudaExternalMemoryHandleTypeOpaqueWin32  \
        gpuExternalMemoryHandleTypeOpaqueWin32
#define cudaExternalMemoryHandleTypeOpaqueWin32Kmt  \
        gpuExternalMemoryHandleTypeOpaqueWin32Kmt
#define cudaExternalMemory_t             gpuExternalMemory_t
#define cudaExternalSemaphoreHandleDesc  gpuExternalSemaphoreHandleDesc
#define cudaExternalSemaphoreHandleType  gpuExternalSemaphoreHandleType
#define cudaExternalSemaphoreHandleTypeD3D12Fence  \
        gpuExternalSemaphoreHandleTypeD3D12Fence
#define cudaExternalSemaphoreHandleTypeOpaqueFd  \
        gpuExternalSemaphoreHandleTypeOpaqueFd
#define cudaExternalSemaphoreHandleTypeOpaqueWin32  \
        gpuExternalSemaphoreHandleTypeOpaqueWin32
#define cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt  \
        gpuExternalSemaphoreHandleTypeOpaqueWin32Kmt
#define cudaExternalSemaphoreSignalParams  \
        gpuExternalSemaphoreSignalParams
#define cudaExternalSemaphoreSignalParams_v1  \
        gpuExternalSemaphoreSignalParams
#define cudaExternalSemaphoreWaitParams  gpuExternalSemaphoreWaitParams
#define cudaExternalSemaphoreWaitParams_v1  \
        gpuExternalSemaphoreWaitParams
#define cudaExternalSemaphore_t          gpuExternalSemaphore_t
#define cudaFuncAttribute                gpuFuncAttribute
#define cudaFuncAttributeMax             gpuFuncAttributeMax
#define cudaFuncAttributeMaxDynamicSharedMemorySize  \
        gpuFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout  \
        gpuFuncAttributePreferredSharedMemoryCarveout
#define cudaFuncAttributes               gpuFuncAttributes
#define cudaFuncCache                    gpuFuncCache_t
#define cudaFuncCachePreferEqual         gpuFuncCachePreferEqual
#define cudaFuncCachePreferL1            gpuFuncCachePreferL1
#define cudaFuncCachePreferNone          gpuFuncCachePreferNone
#define cudaFuncCachePreferShared        gpuFuncCachePreferShared
#define cudaFunction_t                   gpuFunction_t
#define cudaGraphDebugDotFlags           gpuGraphDebugDotFlags
#define cudaGraphDebugDotFlagsEventNodeParams  \
        gpuGraphDebugDotFlagsEventNodeParams
#define cudaGraphDebugDotFlagsExtSemasSignalNodeParams  \
        gpuGraphDebugDotFlagsExtSemasSignalNodeParams
#define cudaGraphDebugDotFlagsExtSemasWaitNodeParams  \
        gpuGraphDebugDotFlagsExtSemasWaitNodeParams
#define cudaGraphDebugDotFlagsHandles    gpuGraphDebugDotFlagsHandles
#define cudaGraphDebugDotFlagsHostNodeParams  \
        gpuGraphDebugDotFlagsHostNodeParams
#define cudaGraphDebugDotFlagsKernelNodeAttributes  \
        gpuGraphDebugDotFlagsKernelNodeAttributes
#define cudaGraphDebugDotFlagsKernelNodeParams  \
        gpuGraphDebugDotFlagsKernelNodeParams
#define cudaGraphDebugDotFlagsMemcpyNodeParams  \
        gpuGraphDebugDotFlagsMemcpyNodeParams
#define cudaGraphDebugDotFlagsMemsetNodeParams  \
        gpuGraphDebugDotFlagsMemsetNodeParams
#define cudaGraphDebugDotFlagsVerbose    gpuGraphDebugDotFlagsVerbose
#define cudaGraphExecUpdateError         gpuGraphExecUpdateError
#define cudaGraphExecUpdateErrorFunctionChanged  \
        gpuGraphExecUpdateErrorFunctionChanged
#define cudaGraphExecUpdateErrorNodeTypeChanged  \
        gpuGraphExecUpdateErrorNodeTypeChanged
#define cudaGraphExecUpdateErrorNotSupported  \
        gpuGraphExecUpdateErrorNotSupported
#define cudaGraphExecUpdateErrorParametersChanged  \
        gpuGraphExecUpdateErrorParametersChanged
#define cudaGraphExecUpdateErrorTopologyChanged  \
        gpuGraphExecUpdateErrorTopologyChanged
#define cudaGraphExecUpdateErrorUnsupportedFunctionChange  \
        gpuGraphExecUpdateErrorUnsupportedFunctionChange
#define cudaGraphExecUpdateResult        gpuGraphExecUpdateResult
#define cudaGraphExecUpdateSuccess       gpuGraphExecUpdateSuccess
#define cudaGraphExec_t                  gpuGraphExec_t
#define cudaGraphInstantiateFlagAutoFreeOnLaunch  \
        gpuGraphInstantiateFlagAutoFreeOnLaunch
#define cudaGraphInstantiateFlagDeviceLaunch  \
        gpuGraphInstantiateFlagDeviceLaunch
#define cudaGraphInstantiateFlagUpload   gpuGraphInstantiateFlagUpload
#define cudaGraphInstantiateFlagUseNodePriority  \
        gpuGraphInstantiateFlagUseNodePriority
#define cudaGraphInstantiateFlags        gpuGraphInstantiateFlags
#define cudaGraphMemAttrReservedMemCurrent  \
        gpuGraphMemAttrReservedMemCurrent
#define cudaGraphMemAttrReservedMemHigh  gpuGraphMemAttrReservedMemHigh
#define cudaGraphMemAttrUsedMemCurrent   gpuGraphMemAttrUsedMemCurrent
#define cudaGraphMemAttrUsedMemHigh      gpuGraphMemAttrUsedMemHigh
#define cudaGraphMemAttributeType        gpuGraphMemAttributeType
#define cudaGraphNodeType                gpuGraphNodeType
#define cudaGraphNodeTypeCount           gpuGraphNodeTypeCount
#define cudaGraphNodeTypeEmpty           gpuGraphNodeTypeEmpty
#define cudaGraphNodeTypeEventRecord     gpuGraphNodeTypeEventRecord
#define cudaGraphNodeTypeExtSemaphoreSignal  \
        gpuGraphNodeTypeExtSemaphoreSignal
#define cudaGraphNodeTypeExtSemaphoreWait  \
        gpuGraphNodeTypeExtSemaphoreWait
#define cudaGraphNodeTypeGraph           gpuGraphNodeTypeGraph
#define cudaGraphNodeTypeHost            gpuGraphNodeTypeHost
#define cudaGraphNodeTypeKernel          gpuGraphNodeTypeKernel
#define cudaGraphNodeTypeMemAlloc        gpuGraphNodeTypeMemAlloc
#define cudaGraphNodeTypeMemFree         gpuGraphNodeTypeMemFree
#define cudaGraphNodeTypeMemcpy          gpuGraphNodeTypeMemcpy
#define cudaGraphNodeTypeMemset          gpuGraphNodeTypeMemset
#define cudaGraphNodeTypeWaitEvent       gpuGraphNodeTypeWaitEvent
#define cudaGraphNode_t                  gpuGraphNode_t
#define cudaGraphUserObjectMove          gpuGraphUserObjectMove
#define cudaGraph_t                      gpuGraph_t
#define cudaGraphicsRegisterFlags        gpuGraphicsRegisterFlags
#define cudaGraphicsRegisterFlagsNone    gpuGraphicsRegisterFlagsNone
#define cudaGraphicsRegisterFlagsReadOnly  \
        gpuGraphicsRegisterFlagsReadOnly
#define cudaGraphicsRegisterFlagsSurfaceLoadStore  \
        gpuGraphicsRegisterFlagsSurfaceLoadStore
#define cudaGraphicsRegisterFlagsTextureGather  \
        gpuGraphicsRegisterFlagsTextureGather
#define cudaGraphicsRegisterFlagsWriteDiscard  \
        gpuGraphicsRegisterFlagsWriteDiscard
#define cudaGraphicsResource             gpuGraphicsResource
#define cudaGraphicsResource_t           gpuGraphicsResource_t
#define cudaHostAllocDefault             gpuHostMallocDefault
#define cudaHostAllocMapped              gpuHostMallocMapped
#define cudaHostAllocPortable            gpuHostMallocPortable
#define cudaHostAllocWriteCombined       gpuHostMallocWriteCombined
#define cudaHostFn_t                     gpuHostFn_t
#define cudaHostNodeParams               gpuHostNodeParams
#define cudaHostRegisterDefault          gpuHostRegisterDefault
#define cudaHostRegisterIoMemory         gpuHostRegisterIoMemory
#define cudaHostRegisterMapped           gpuHostRegisterMapped
#define cudaHostRegisterPortable         gpuHostRegisterPortable
#define cudaHostRegisterReadOnly         gpuHostRegisterReadOnly
#define cudaInvalidDeviceId              gpuInvalidDeviceId
#define cudaIpcEventHandle_st            gpuIpcEventHandle_st
#define cudaIpcEventHandle_t             gpuIpcEventHandle_t
#define cudaIpcMemHandle_st              gpuIpcMemHandle_st
#define cudaIpcMemHandle_t               gpuIpcMemHandle_t
#define cudaIpcMemLazyEnablePeerAccess   gpuIpcMemLazyEnablePeerAccess
#define cudaKernelNodeAttrID             gpuKernelNodeAttrID
#define cudaKernelNodeAttrValue          gpuKernelNodeAttrValue
#define cudaKernelNodeAttributeAccessPolicyWindow  \
        gpuKernelNodeAttributeAccessPolicyWindow
#define cudaKernelNodeAttributeCooperative  \
        gpuKernelNodeAttributeCooperative
#define cudaKernelNodeParams             gpuKernelNodeParams
#define cudaLaunchParams                 gpuLaunchParams
#define cudaLimit                        gpuLimit_t
#define cudaLimitMallocHeapSize          gpuLimitMallocHeapSize
#define cudaLimitPrintfFifoSize          gpuLimitPrintfFifoSize
#define cudaLimitStackSize               gpuLimitStackSize
#define cudaMemAccessDesc                gpuMemAccessDesc
#define cudaMemAccessFlags               gpuMemAccessFlags
#define cudaMemAccessFlagsProtNone       gpuMemAccessFlagsProtNone
#define cudaMemAccessFlagsProtRead       gpuMemAccessFlagsProtRead
#define cudaMemAccessFlagsProtReadWrite  gpuMemAccessFlagsProtReadWrite
#define cudaMemAdviseSetAccessedBy       gpuMemAdviseSetAccessedBy
#define cudaMemAdviseSetPreferredLocation  \
        gpuMemAdviseSetPreferredLocation
#define cudaMemAdviseSetReadMostly       gpuMemAdviseSetReadMostly
#define cudaMemAdviseUnsetAccessedBy     gpuMemAdviseUnsetAccessedBy
#define cudaMemAdviseUnsetPreferredLocation  \
        gpuMemAdviseUnsetPreferredLocation
#define cudaMemAdviseUnsetReadMostly     gpuMemAdviseUnsetReadMostly
#define cudaMemAllocNodeParams           gpuMemAllocNodeParams
#define cudaMemAllocationHandleType      gpuMemAllocationHandleType
#define cudaMemAllocationType            gpuMemAllocationType
#define cudaMemAllocationTypeInvalid     gpuMemAllocationTypeInvalid
#define cudaMemAllocationTypeMax         gpuMemAllocationTypeMax
#define cudaMemAllocationTypePinned      gpuMemAllocationTypePinned
#define cudaMemAttachGlobal              gpuMemAttachGlobal
#define cudaMemAttachHost                gpuMemAttachHost
#define cudaMemAttachSingle              gpuMemAttachSingle
#define cudaMemHandleTypeNone            gpuMemHandleTypeNone
#define cudaMemHandleTypePosixFileDescriptor  \
        gpuMemHandleTypePosixFileDescriptor
#define cudaMemHandleTypeWin32           gpuMemHandleTypeWin32
#define cudaMemHandleTypeWin32Kmt        gpuMemHandleTypeWin32Kmt
#define cudaMemLocation                  gpuMemLocation
#define cudaMemLocationType              gpuMemLocationType
#define cudaMemLocationTypeDevice        gpuMemLocationTypeDevice
#define cudaMemLocationTypeInvalid       gpuMemLocationTypeInvalid
#define cudaMemPoolAttr                  gpuMemPoolAttr
#define cudaMemPoolAttrReleaseThreshold  gpuMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReservedMemCurrent  \
        gpuMemPoolAttrReservedMemCurrent
#define cudaMemPoolAttrReservedMemHigh   gpuMemPoolAttrReservedMemHigh
#define cudaMemPoolAttrUsedMemCurrent    gpuMemPoolAttrUsedMemCurrent
#define cudaMemPoolAttrUsedMemHigh       gpuMemPoolAttrUsedMemHigh
#define cudaMemPoolProps                 gpuMemPoolProps
#define cudaMemPoolPtrExportData         gpuMemPoolPtrExportData
#define cudaMemPoolReuseAllowInternalDependencies  \
        gpuMemPoolReuseAllowInternalDependencies
#define cudaMemPoolReuseAllowOpportunistic  \
        gpuMemPoolReuseAllowOpportunistic
#define cudaMemPoolReuseFollowEventDependencies  \
        gpuMemPoolReuseFollowEventDependencies
#define cudaMemPool_t                    gpuMemPool_t
#define cudaMemRangeAttribute            gpuMemRangeAttribute
#define cudaMemRangeAttributeAccessedBy  gpuMemRangeAttributeAccessedBy
#define cudaMemRangeAttributeLastPrefetchLocation  \
        gpuMemRangeAttributeLastPrefetchLocation
#define cudaMemRangeAttributePreferredLocation  \
        gpuMemRangeAttributePreferredLocation
#define cudaMemRangeAttributeReadMostly  gpuMemRangeAttributeReadMostly
#define cudaMemcpy3DParms                gpuMemcpy3DParms
#define cudaMemcpyDefault                gpuMemcpyDefault
#define cudaMemcpyDeviceToDevice         gpuMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost           gpuMemcpyDeviceToHost
#define cudaMemcpyHostToDevice           gpuMemcpyHostToDevice
#define cudaMemcpyHostToHost             gpuMemcpyHostToHost
#define cudaMemcpyKind                   gpuMemcpyKind
#define cudaMemoryAdvise                 gpuMemoryAdvise
#define cudaMemoryType                   gpuMemoryType
#define cudaMemoryTypeDevice             gpuMemoryTypeDevice
#define cudaMemoryTypeHost               gpuMemoryTypeHost
#define cudaMemoryTypeManaged            gpuMemoryTypeManaged
#define cudaMemsetParams                 gpuMemsetParams
#define cudaMipmappedArray               gpuMipmappedArray
#define cudaMipmappedArray_const_t       gpuMipmappedArray_const_t
#define cudaMipmappedArray_t             gpuMipmappedArray_t
#define cudaOccupancyDefault             gpuOccupancyDefault
#define cudaOccupancyDisableCachingOverride  \
        gpuOccupancyDisableCachingOverride
#define cudaPitchedPtr                   gpuPitchedPtr
#define cudaPointerAttributes            gpuPointerAttribute_t
#define cudaPos                          gpuPos
#define cudaResViewFormatFloat1          gpuResViewFormatFloat1
#define cudaResViewFormatFloat2          gpuResViewFormatFloat2
#define cudaResViewFormatFloat4          gpuResViewFormatFloat4
#define cudaResViewFormatHalf1           gpuResViewFormatHalf1
#define cudaResViewFormatHalf2           gpuResViewFormatHalf2
#define cudaResViewFormatHalf4           gpuResViewFormatHalf4
#define cudaResViewFormatNone            gpuResViewFormatNone
#define cudaResViewFormatSignedBlockCompressed4  \
        gpuResViewFormatSignedBlockCompressed4
#define cudaResViewFormatSignedBlockCompressed5  \
        gpuResViewFormatSignedBlockCompressed5
#define cudaResViewFormatSignedBlockCompressed6H  \
        gpuResViewFormatSignedBlockCompressed6H
#define cudaResViewFormatSignedChar1     gpuResViewFormatSignedChar1
#define cudaResViewFormatSignedChar2     gpuResViewFormatSignedChar2
#define cudaResViewFormatSignedChar4     gpuResViewFormatSignedChar4
#define cudaResViewFormatSignedInt1      gpuResViewFormatSignedInt1
#define cudaResViewFormatSignedInt2      gpuResViewFormatSignedInt2
#define cudaResViewFormatSignedInt4      gpuResViewFormatSignedInt4
#define cudaResViewFormatSignedShort1    gpuResViewFormatSignedShort1
#define cudaResViewFormatSignedShort2    gpuResViewFormatSignedShort2
#define cudaResViewFormatSignedShort4    gpuResViewFormatSignedShort4
#define cudaResViewFormatUnsignedBlockCompressed1  \
        gpuResViewFormatUnsignedBlockCompressed1
#define cudaResViewFormatUnsignedBlockCompressed2  \
        gpuResViewFormatUnsignedBlockCompressed2
#define cudaResViewFormatUnsignedBlockCompressed3  \
        gpuResViewFormatUnsignedBlockCompressed3
#define cudaResViewFormatUnsignedBlockCompressed4  \
        gpuResViewFormatUnsignedBlockCompressed4
#define cudaResViewFormatUnsignedBlockCompressed5  \
        gpuResViewFormatUnsignedBlockCompressed5
#define cudaResViewFormatUnsignedBlockCompressed6H  \
        gpuResViewFormatUnsignedBlockCompressed6H
#define cudaResViewFormatUnsignedBlockCompressed7  \
        gpuResViewFormatUnsignedBlockCompressed7
#define cudaResViewFormatUnsignedChar1   gpuResViewFormatUnsignedChar1
#define cudaResViewFormatUnsignedChar2   gpuResViewFormatUnsignedChar2
#define cudaResViewFormatUnsignedChar4   gpuResViewFormatUnsignedChar4
#define cudaResViewFormatUnsignedInt1    gpuResViewFormatUnsignedInt1
#define cudaResViewFormatUnsignedInt2    gpuResViewFormatUnsignedInt2
#define cudaResViewFormatUnsignedInt4    gpuResViewFormatUnsignedInt4
#define cudaResViewFormatUnsignedShort1  gpuResViewFormatUnsignedShort1
#define cudaResViewFormatUnsignedShort2  gpuResViewFormatUnsignedShort2
#define cudaResViewFormatUnsignedShort4  gpuResViewFormatUnsignedShort4
#define cudaResourceDesc                 gpuResourceDesc
#define cudaResourceType                 gpuResourceType
#define cudaResourceTypeArray            gpuResourceTypeArray
#define cudaResourceTypeLinear           gpuResourceTypeLinear
#define cudaResourceTypeMipmappedArray   gpuResourceTypeMipmappedArray
#define cudaResourceTypePitch2D          gpuResourceTypePitch2D
#define cudaResourceViewDesc             gpuResourceViewDesc
#define cudaResourceViewFormat           gpuResourceViewFormat
#define cudaSharedMemBankSizeDefault     gpuSharedMemBankSizeDefault
#define cudaSharedMemBankSizeEightByte   gpuSharedMemBankSizeEightByte
#define cudaSharedMemBankSizeFourByte    gpuSharedMemBankSizeFourByte
#define cudaSharedMemConfig              gpuSharedMemConfig
#define cudaStreamAddCaptureDependencies gpuStreamAddCaptureDependencies
#define cudaStreamCaptureMode            gpuStreamCaptureMode
#define cudaStreamCaptureModeGlobal      gpuStreamCaptureModeGlobal
#define cudaStreamCaptureModeRelaxed     gpuStreamCaptureModeRelaxed
#define cudaStreamCaptureModeThreadLocal gpuStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus          gpuStreamCaptureStatus
#define cudaStreamCaptureStatusActive    gpuStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated  \
        gpuStreamCaptureStatusInvalidated
#define cudaStreamCaptureStatusNone      gpuStreamCaptureStatusNone
#define cudaStreamDefault                gpuStreamDefault
#define cudaStreamNonBlocking            gpuStreamNonBlocking
#define cudaStreamPerThread              gpuStreamPerThread
#define cudaStreamSetCaptureDependencies gpuStreamSetCaptureDependencies
#define cudaStreamUpdateCaptureDependenciesFlags  \
        gpuStreamUpdateCaptureDependenciesFlags
#define cudaStream_t                     gpuStream_t
#define cudaSuccess                      gpuSuccess
#define cudaUUID_t                       gpuUUID
#define cudaUserObjectFlags              gpuUserObjectFlags
#define cudaUserObjectNoDestructorSync   gpuUserObjectNoDestructorSync
#define cudaUserObjectRetainFlags        gpuUserObjectRetainFlags
#define cudaUserObject_t                 gpuUserObject_t

/* surface_types.h */
#define cudaBoundaryModeClamp            gpuBoundaryModeClamp
#define cudaBoundaryModeTrap             gpuBoundaryModeTrap
#define cudaBoundaryModeZero             gpuBoundaryModeZero
#define cudaSurfaceBoundaryMode          gpuSurfaceBoundaryMode
#define cudaSurfaceObject_t              gpuSurfaceObject_t

/* texture_types.h */
#define cudaAddressModeBorder            gpuAddressModeBorder
#define cudaAddressModeClamp             gpuAddressModeClamp
#define cudaAddressModeMirror            gpuAddressModeMirror
#define cudaAddressModeWrap              gpuAddressModeWrap
#define cudaFilterModeLinear             gpuFilterModeLinear
#define cudaFilterModePoint              gpuFilterModePoint
#define cudaReadModeElementType          gpuReadModeElementType
#define cudaReadModeNormalizedFloat      gpuReadModeNormalizedFloat
#define cudaTextureAddressMode           gpuTextureAddressMode
#define cudaTextureDesc                  gpuTextureDesc
#define cudaTextureFilterMode            gpuTextureFilterMode
#define cudaTextureObject_t              gpuTextureObject_t
#define cudaTextureReadMode              gpuTextureReadMode
#define cudaTextureType1D                gpuTextureType1D
#define cudaTextureType1DLayered         gpuTextureType1DLayered
#define cudaTextureType2D                gpuTextureType2D
#define cudaTextureType2DLayered         gpuTextureType2DLayered
#define cudaTextureType3D                gpuTextureType3D
#define cudaTextureTypeCubemap           gpuTextureTypeCubemap
#define cudaTextureTypeCubemapLayered    gpuTextureTypeCubemapLayered

#include <hop/hop_runtime_api.h>

#endif
