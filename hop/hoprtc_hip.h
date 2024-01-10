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

#ifndef __HOP_HOPRTC_HIP_H__
#define __HOP_HOPRTC_HIP_H__

#include <hip/hiprtc.h>

#define GPURTC_ERROR_BUILTIN_OPERATION_FAILURE  \
        HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE
#define GPURTC_ERROR_COMPILATION         HIPRTC_ERROR_COMPILATION
#define GPURTC_ERROR_INTERNAL_ERROR      HIPRTC_ERROR_INTERNAL_ERROR
#define GPURTC_ERROR_INVALID_INPUT       HIPRTC_ERROR_INVALID_INPUT
#define GPURTC_ERROR_INVALID_OPTION      HIPRTC_ERROR_INVALID_OPTION
#define GPURTC_ERROR_INVALID_PROGRAM     HIPRTC_ERROR_INVALID_PROGRAM
#define GPURTC_ERROR_NAME_EXPRESSION_NOT_VALID  \
        HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID
#define GPURTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION  \
        HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
#define GPURTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION  \
        HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
#define GPURTC_ERROR_OUT_OF_MEMORY       HIPRTC_ERROR_OUT_OF_MEMORY
#define GPURTC_ERROR_PROGRAM_CREATION_FAILURE  \
        HIPRTC_ERROR_PROGRAM_CREATION_FAILURE
#define GPURTC_JIT_CACHE_MODE            HIPRTC_JIT_CACHE_MODE
#define GPURTC_JIT_ERROR_LOG_BUFFER      HIPRTC_JIT_ERROR_LOG_BUFFER
#define GPURTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES  \
        HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#define GPURTC_JIT_FALLBACK_STRATEGY     HIPRTC_JIT_FALLBACK_STRATEGY
#define GPURTC_JIT_FAST_COMPILE          HIPRTC_JIT_FAST_COMPILE
#define GPURTC_JIT_GENERATE_DEBUG_INFO   HIPRTC_JIT_GENERATE_DEBUG_INFO
#define GPURTC_JIT_GENERATE_LINE_INFO    HIPRTC_JIT_GENERATE_LINE_INFO
#define GPURTC_JIT_INFO_LOG_BUFFER       HIPRTC_JIT_INFO_LOG_BUFFER
#define GPURTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES  \
        HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES
#define GPURTC_JIT_INPUT_CUBIN           HIPRTC_JIT_INPUT_CUBIN
#define GPURTC_JIT_INPUT_FATBINARY       HIPRTC_JIT_INPUT_FATBINARY
#define GPURTC_JIT_INPUT_LIBRARY         HIPRTC_JIT_INPUT_LIBRARY
#define GPURTC_JIT_INPUT_NVVM            HIPRTC_JIT_INPUT_NVVM
#define GPURTC_JIT_INPUT_OBJECT          HIPRTC_JIT_INPUT_OBJECT
#define GPURTC_JIT_INPUT_PTX             HIPRTC_JIT_INPUT_PTX
#define GPURTC_JIT_LOG_VERBOSE           HIPRTC_JIT_LOG_VERBOSE
#define GPURTC_JIT_MAX_REGISTERS         HIPRTC_JIT_MAX_REGISTERS
#define GPURTC_JIT_NEW_SM3X_OPT          HIPRTC_JIT_NEW_SM3X_OPT
#define GPURTC_JIT_NUM_LEGACY_INPUT_TYPES  \
        HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES
#define GPURTC_JIT_NUM_OPTIONS           HIPRTC_JIT_NUM_OPTIONS
#define GPURTC_JIT_OPTIMIZATION_LEVEL    HIPRTC_JIT_OPTIMIZATION_LEVEL
#define GPURTC_JIT_TARGET                HIPRTC_JIT_TARGET
#define GPURTC_JIT_TARGET_FROM_HIPCONTEXT  \
        HIPRTC_JIT_TARGET_FROM_HIPCONTEXT
#define GPURTC_JIT_THREADS_PER_BLOCK     HIPRTC_JIT_THREADS_PER_BLOCK
#define GPURTC_JIT_WALL_TIME             HIPRTC_JIT_WALL_TIME
#define GPURTC_SUCCESS                   HIPRTC_SUCCESS
#define gpurtcAddNameExpression          hiprtcAddNameExpression
#define gpurtcCompileProgram             hiprtcCompileProgram
#define gpurtcCreateProgram              hiprtcCreateProgram
#define gpurtcDestroyProgram             hiprtcDestroyProgram
#define gpurtcGetBitcode                 hiprtcGetBitcode
#define gpurtcGetBitcodeSize             hiprtcGetBitcodeSize
#define gpurtcGetCode                    hiprtcGetCode
#define gpurtcGetCodeSize                hiprtcGetCodeSize
#define gpurtcGetErrorString             hiprtcGetErrorString
#define gpurtcGetLoweredName             hiprtcGetLoweredName
#define gpurtcGetProgramLog              hiprtcGetProgramLog
#define gpurtcGetProgramLogSize          hiprtcGetProgramLogSize
#define gpurtcJITInputType               hiprtcJITInputType
#define gpurtcLinkAddData                hiprtcLinkAddData
#define gpurtcLinkAddFile                hiprtcLinkAddFile
#define gpurtcLinkComplete               hiprtcLinkComplete
#define gpurtcLinkCreate                 hiprtcLinkCreate
#define gpurtcLinkDestroy                hiprtcLinkDestroy
#define gpurtcLinkState                  hiprtcLinkState
#define gpurtcProgram                    hiprtcProgram
#define gpurtcResult                     hiprtcResult
#define gpurtcVersion                    hiprtcVersion

#endif
