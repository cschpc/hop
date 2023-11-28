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

#ifndef __HOP_SOURCE_HIP_HIPRTC_H__
#define __HOP_SOURCE_HIP_HIPRTC_H__

#define HOP_SOURCE_HIP

#define HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE  \
        GPURTC_ERROR_BUILTIN_OPERATION_FAILURE
#define HIPRTC_ERROR_COMPILATION         GPURTC_ERROR_COMPILATION
#define HIPRTC_ERROR_INTERNAL_ERROR      GPURTC_ERROR_INTERNAL_ERROR
#define HIPRTC_ERROR_INVALID_INPUT       GPURTC_ERROR_INVALID_INPUT
#define HIPRTC_ERROR_INVALID_OPTION      GPURTC_ERROR_INVALID_OPTION
#define HIPRTC_ERROR_INVALID_PROGRAM     GPURTC_ERROR_INVALID_PROGRAM
#define HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID  \
        GPURTC_ERROR_NAME_EXPRESSION_NOT_VALID
#define HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION  \
        GPURTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
#define HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION  \
        GPURTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
#define HIPRTC_ERROR_OUT_OF_MEMORY       GPURTC_ERROR_OUT_OF_MEMORY
#define HIPRTC_ERROR_PROGRAM_CREATION_FAILURE  \
        GPURTC_ERROR_PROGRAM_CREATION_FAILURE
#define HIPRTC_JIT_CACHE_MODE            GPURTC_JIT_CACHE_MODE
#define HIPRTC_JIT_ERROR_LOG_BUFFER      GPURTC_JIT_ERROR_LOG_BUFFER
#define HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES  \
        GPURTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
#define HIPRTC_JIT_FALLBACK_STRATEGY     GPURTC_JIT_FALLBACK_STRATEGY
#define HIPRTC_JIT_FAST_COMPILE          GPURTC_JIT_FAST_COMPILE
#define HIPRTC_JIT_GENERATE_DEBUG_INFO   GPURTC_JIT_GENERATE_DEBUG_INFO
#define HIPRTC_JIT_GENERATE_LINE_INFO    GPURTC_JIT_GENERATE_LINE_INFO
#define HIPRTC_JIT_INFO_LOG_BUFFER       GPURTC_JIT_INFO_LOG_BUFFER
#define HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES  \
        GPURTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES
#define HIPRTC_JIT_INPUT_CUBIN           GPURTC_JIT_INPUT_CUBIN
#define HIPRTC_JIT_INPUT_FATBINARY       GPURTC_JIT_INPUT_FATBINARY
#define HIPRTC_JIT_INPUT_LIBRARY         GPURTC_JIT_INPUT_LIBRARY
#define HIPRTC_JIT_INPUT_NVVM            GPURTC_JIT_INPUT_NVVM
#define HIPRTC_JIT_INPUT_OBJECT          GPURTC_JIT_INPUT_OBJECT
#define HIPRTC_JIT_INPUT_PTX             GPURTC_JIT_INPUT_PTX
#define HIPRTC_JIT_LOG_VERBOSE           GPURTC_JIT_LOG_VERBOSE
#define HIPRTC_JIT_MAX_REGISTERS         GPURTC_JIT_MAX_REGISTERS
#define HIPRTC_JIT_NEW_SM3X_OPT          GPURTC_JIT_NEW_SM3X_OPT
#define HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES  \
        GPURTC_JIT_NUM_LEGACY_INPUT_TYPES
#define HIPRTC_JIT_NUM_OPTIONS           GPURTC_JIT_NUM_OPTIONS
#define HIPRTC_JIT_OPTIMIZATION_LEVEL    GPURTC_JIT_OPTIMIZATION_LEVEL
#define HIPRTC_JIT_TARGET                GPURTC_JIT_TARGET
#define HIPRTC_JIT_TARGET_FROM_HIPCONTEXT  \
        GPURTC_JIT_TARGET_FROM_HIPCONTEXT
#define HIPRTC_JIT_THREADS_PER_BLOCK     GPURTC_JIT_THREADS_PER_BLOCK
#define HIPRTC_JIT_WALL_TIME             GPURTC_JIT_WALL_TIME
#define HIPRTC_SUCCESS                   GPURTC_SUCCESS
#define hiprtcAddNameExpression          gpurtcAddNameExpression
#define hiprtcCompileProgram             gpurtcCompileProgram
#define hiprtcCreateProgram              gpurtcCreateProgram
#define hiprtcDestroyProgram             gpurtcDestroyProgram
#define hiprtcGetBitcode                 gpurtcGetBitcode
#define hiprtcGetBitcodeSize             gpurtcGetBitcodeSize
#define hiprtcGetCode                    gpurtcGetCode
#define hiprtcGetCodeSize                gpurtcGetCodeSize
#define hiprtcGetErrorString             gpurtcGetErrorString
#define hiprtcGetLoweredName             gpurtcGetLoweredName
#define hiprtcGetProgramLog              gpurtcGetProgramLog
#define hiprtcGetProgramLogSize          gpurtcGetProgramLogSize
#define hiprtcJITInputType               gpurtcJITInputType
#define hiprtcLinkAddData                gpurtcLinkAddData
#define hiprtcLinkAddFile                gpurtcLinkAddFile
#define hiprtcLinkComplete               gpurtcLinkComplete
#define hiprtcLinkCreate                 gpurtcLinkCreate
#define hiprtcLinkDestroy                gpurtcLinkDestroy
#define hiprtcLinkState                  gpurtcLinkState
#define hiprtcProgram                    gpurtcProgram
#define hiprtcResult                     gpurtcResult
#define hiprtcVersion                    gpurtcVersion

#include <hop/hoprtc.h>

#endif
