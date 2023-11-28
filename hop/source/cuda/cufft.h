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

#ifndef __HOP_SOURCE_CUDA_CUFFT_H__
#define __HOP_SOURCE_CUDA_CUFFT_H__

#define HOP_SOURCE_CUDA

#include <cuComplex.h>
#include <driver_types.h>
#include <library_types.h>

#define CUFFT_ALLOC_FAILED               GPUFFT_ALLOC_FAILED
#define CUFFT_C2C                        GPUFFT_C2C
#define CUFFT_C2R                        GPUFFT_C2R
#define CUFFT_D2Z                        GPUFFT_D2Z
#define CUFFT_EXEC_FAILED                GPUFFT_EXEC_FAILED
#define CUFFT_FORWARD                    GPUFFT_FORWARD
#define CUFFT_INCOMPLETE_PARAMETER_LIST  GPUFFT_INCOMPLETE_PARAMETER_LIST
#define CUFFT_INTERNAL_ERROR             GPUFFT_INTERNAL_ERROR
#define CUFFT_INVALID_DEVICE             GPUFFT_INVALID_DEVICE
#define CUFFT_INVALID_PLAN               GPUFFT_INVALID_PLAN
#define CUFFT_INVALID_SIZE               GPUFFT_INVALID_SIZE
#define CUFFT_INVALID_TYPE               GPUFFT_INVALID_TYPE
#define CUFFT_INVALID_VALUE              GPUFFT_INVALID_VALUE
#define CUFFT_INVERSE                    GPUFFT_BACKWARD
#define CUFFT_NOT_IMPLEMENTED            GPUFFT_NOT_IMPLEMENTED
#define CUFFT_NOT_SUPPORTED              GPUFFT_NOT_SUPPORTED
#define CUFFT_NO_WORKSPACE               GPUFFT_NO_WORKSPACE
#define CUFFT_PARSE_ERROR                GPUFFT_PARSE_ERROR
#define CUFFT_R2C                        GPUFFT_R2C
#define CUFFT_SETUP_FAILED               GPUFFT_SETUP_FAILED
#define CUFFT_SUCCESS                    GPUFFT_SUCCESS
#define CUFFT_UNALIGNED_DATA             GPUFFT_UNALIGNED_DATA
#define CUFFT_Z2D                        GPUFFT_Z2D
#define CUFFT_Z2Z                        GPUFFT_Z2Z
#define cufftComplex                     gpufftComplex
#define cufftCreate                      gpufftCreate
#define cufftDestroy                     gpufftDestroy
#define cufftDoubleComplex               gpufftDoubleComplex
#define cufftDoubleReal                  gpufftDoubleReal
#define cufftEstimate1d                  gpufftEstimate1d
#define cufftEstimate2d                  gpufftEstimate2d
#define cufftEstimate3d                  gpufftEstimate3d
#define cufftEstimateMany                gpufftEstimateMany
#define cufftExecC2C                     gpufftExecC2C
#define cufftExecC2R                     gpufftExecC2R
#define cufftExecD2Z                     gpufftExecD2Z
#define cufftExecR2C                     gpufftExecR2C
#define cufftExecZ2D                     gpufftExecZ2D
#define cufftExecZ2Z                     gpufftExecZ2Z
#define cufftGetProperty                 gpufftGetProperty
#define cufftGetSize                     gpufftGetSize
#define cufftGetSize1d                   gpufftGetSize1d
#define cufftGetSize2d                   gpufftGetSize2d
#define cufftGetSize3d                   gpufftGetSize3d
#define cufftGetSizeMany                 gpufftGetSizeMany
#define cufftGetSizeMany64               gpufftGetSizeMany64
#define cufftGetVersion                  gpufftGetVersion
#define cufftHandle                      gpufftHandle
#define cufftMakePlan1d                  gpufftMakePlan1d
#define cufftMakePlan2d                  gpufftMakePlan2d
#define cufftMakePlan3d                  gpufftMakePlan3d
#define cufftMakePlanMany                gpufftMakePlanMany
#define cufftMakePlanMany64              gpufftMakePlanMany64
#define cufftPlan1d                      gpufftPlan1d
#define cufftPlan2d                      gpufftPlan2d
#define cufftPlan3d                      gpufftPlan3d
#define cufftPlanMany                    gpufftPlanMany
#define cufftReal                        gpufftReal
#define cufftResult                      gpufftResult
#define cufftResult_t                    gpufftResult_t
#define cufftSetAutoAllocation           gpufftSetAutoAllocation
#define cufftSetStream                   gpufftSetStream
#define cufftSetWorkArea                 gpufftSetWorkArea
#define cufftType                        gpufftType
#define cufftType_t                      gpufftType_t

#include <hop/hopfft.h>

#endif
