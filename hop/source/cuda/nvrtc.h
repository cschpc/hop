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

#ifndef __HOP_SOURCE_CUDA_NVRTC_H__
#define __HOP_SOURCE_CUDA_NVRTC_H__

#if !defined(HOP_SOURCE_CUDA)
#define HOP_SOURCE_CUDA
#endif

#define NVRTC_ERROR_BUILTIN_OPERATION_FAILURE  \
        GPURTC_ERROR_BUILTIN_OPERATION_FAILURE
#define NVRTC_ERROR_COMPILATION          GPURTC_ERROR_COMPILATION
#define NVRTC_ERROR_INTERNAL_ERROR       GPURTC_ERROR_INTERNAL_ERROR
#define NVRTC_ERROR_INVALID_INPUT        GPURTC_ERROR_INVALID_INPUT
#define NVRTC_ERROR_INVALID_OPTION       GPURTC_ERROR_INVALID_OPTION
#define NVRTC_ERROR_INVALID_PROGRAM      GPURTC_ERROR_INVALID_PROGRAM
#define NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID  \
        GPURTC_ERROR_NAME_EXPRESSION_NOT_VALID
#define NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION  \
        GPURTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
#define NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION  \
        GPURTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
#define NVRTC_ERROR_OUT_OF_MEMORY        GPURTC_ERROR_OUT_OF_MEMORY
#define NVRTC_ERROR_PROGRAM_CREATION_FAILURE  \
        GPURTC_ERROR_PROGRAM_CREATION_FAILURE
#define NVRTC_SUCCESS                    GPURTC_SUCCESS
#define nvrtcAddNameExpression           gpurtcAddNameExpression
#define nvrtcCompileProgram              gpurtcCompileProgram
#define nvrtcCreateProgram               gpurtcCreateProgram
#define nvrtcDestroyProgram              gpurtcDestroyProgram
#define nvrtcGetCUBIN                    gpurtcGetBitcode
#define nvrtcGetCUBINSize                gpurtcGetBitcodeSize
#define nvrtcGetErrorString              gpurtcGetErrorString
#define nvrtcGetLoweredName              gpurtcGetLoweredName
#define nvrtcGetPTX                      gpurtcGetCode
#define nvrtcGetPTXSize                  gpurtcGetCodeSize
#define nvrtcGetProgramLog               gpurtcGetProgramLog
#define nvrtcGetProgramLogSize           gpurtcGetProgramLogSize
#define nvrtcProgram                     gpurtcProgram
#define nvrtcResult                      gpurtcResult
#define nvrtcVersion                     gpurtcVersion

#include <hop/hoprtc.h>

#endif
