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

#ifndef __HOP_HOPRTC_CUDA_H__
#define __HOP_HOPRTC_CUDA_H__

#include <cuda.h>
#include <nvrtc.h>

#define GPURTC_ERROR_BUILTIN_OPERATION_FAILURE  \
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
#define GPURTC_ERROR_COMPILATION         NVRTC_ERROR_COMPILATION
#define GPURTC_ERROR_INTERNAL_ERROR      NVRTC_ERROR_INTERNAL_ERROR
#define GPURTC_ERROR_INVALID_INPUT       NVRTC_ERROR_INVALID_INPUT
#define GPURTC_ERROR_INVALID_OPTION      NVRTC_ERROR_INVALID_OPTION
#define GPURTC_ERROR_INVALID_PROGRAM     NVRTC_ERROR_INVALID_PROGRAM
#define GPURTC_ERROR_NAME_EXPRESSION_NOT_VALID  \
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
#define GPURTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION  \
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
#define GPURTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION  \
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
#define GPURTC_ERROR_OUT_OF_MEMORY       NVRTC_ERROR_OUT_OF_MEMORY
#define GPURTC_ERROR_PROGRAM_CREATION_FAILURE  \
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE
#define GPURTC_JIT_CACHE_MODE            CU_JIT_CACHE_MODE
#define GPURTC_JIT_ERROR_LOG_BUFFER      CU_JIT_ERROR_LOG_BUFFER
#define GPURTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES  \
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#define GPURTC_JIT_FALLBACK_STRATEGY     CU_JIT_FALLBACK_STRATEGY
#define GPURTC_JIT_FAST_COMPILE          CU_JIT_FAST_COMPILE
#define GPURTC_JIT_GENERATE_DEBUG_INFO   CU_JIT_GENERATE_DEBUG_INFO
#define GPURTC_JIT_GENERATE_LINE_INFO    CU_JIT_GENERATE_LINE_INFO
#define GPURTC_JIT_INFO_LOG_BUFFER       CU_JIT_INFO_LOG_BUFFER
#define GPURTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES  \
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
#define GPURTC_JIT_INPUT_CUBIN           CU_JIT_INPUT_CUBIN
#define GPURTC_JIT_INPUT_FATBINARY       CU_JIT_INPUT_FATBINARY
#define GPURTC_JIT_INPUT_LIBRARY         CU_JIT_INPUT_LIBRARY
#define GPURTC_JIT_INPUT_NVVM            CU_JIT_INPUT_NVVM
#define GPURTC_JIT_INPUT_OBJECT          CU_JIT_INPUT_OBJECT
#define GPURTC_JIT_INPUT_PTX             CU_JIT_INPUT_PTX
#define GPURTC_JIT_LOG_VERBOSE           CU_JIT_LOG_VERBOSE
#define GPURTC_JIT_MAX_REGISTERS         CU_JIT_MAX_REGISTERS
#define GPURTC_JIT_NEW_SM3X_OPT          CU_JIT_NEW_SM3X_OPT
#define GPURTC_JIT_NUM_LEGACY_INPUT_TYPES  \
        CU_JIT_NUM_INPUT_TYPES
#define GPURTC_JIT_NUM_OPTIONS           CU_JIT_NUM_OPTIONS
#define GPURTC_JIT_OPTIMIZATION_LEVEL    CU_JIT_OPTIMIZATION_LEVEL
#define GPURTC_JIT_TARGET                CU_JIT_TARGET
#define GPURTC_JIT_TARGET_FROM_HIPCONTEXT  \
        CU_JIT_TARGET_FROM_CUCONTEXT
#define GPURTC_JIT_THREADS_PER_BLOCK     CU_JIT_THREADS_PER_BLOCK
#define GPURTC_JIT_WALL_TIME             CU_JIT_WALL_TIME
#define GPURTC_SUCCESS                   NVRTC_SUCCESS
#define gpurtcAddNameExpression          nvrtcAddNameExpression
#define gpurtcCompileProgram             nvrtcCompileProgram
#define gpurtcCreateProgram              nvrtcCreateProgram
#define gpurtcDestroyProgram             nvrtcDestroyProgram
#define gpurtcGetBitcode                 nvrtcGetCUBIN
#define gpurtcGetBitcodeSize             nvrtcGetCUBINSize
#define gpurtcGetCode                    nvrtcGetPTX
#define gpurtcGetCodeSize                nvrtcGetPTXSize
#define gpurtcGetErrorString             nvrtcGetErrorString
#define gpurtcGetLoweredName             nvrtcGetLoweredName
#define gpurtcGetProgramLog              nvrtcGetProgramLog
#define gpurtcGetProgramLogSize          nvrtcGetProgramLogSize
#define gpurtcJITInputType               CUjitInputType
#define gpurtcLinkAddData                cuLinkAddData_v2
#define gpurtcLinkAddFile                cuLinkAddFile_v2
#define gpurtcLinkComplete               cuLinkComplete
#define gpurtcLinkCreate                 cuLinkCreate_v2
#define gpurtcLinkDestroy                cuLinkDestroy
#define gpurtcLinkState                  CUlinkState
#define gpurtcProgram                    nvrtcProgram
#define gpurtcResult                     nvrtcResult
#define gpurtcVersion                    nvrtcVersion


#endif
