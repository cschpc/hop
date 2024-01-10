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

#ifndef __HOP_HOPFFT_CUDA_H__
#define __HOP_HOPFFT_CUDA_H__

#include <cufft.h>

#define GPUFFT_ALLOC_FAILED              CUFFT_ALLOC_FAILED
#define GPUFFT_BACKWARD                  CUFFT_INVERSE
#define GPUFFT_C2C                       CUFFT_C2C
#define GPUFFT_C2R                       CUFFT_C2R
#define GPUFFT_D2Z                       CUFFT_D2Z
#define GPUFFT_EXEC_FAILED               CUFFT_EXEC_FAILED
#define GPUFFT_FORWARD                   CUFFT_FORWARD
#define GPUFFT_INCOMPLETE_PARAMETER_LIST CUFFT_INCOMPLETE_PARAMETER_LIST
#define GPUFFT_INTERNAL_ERROR            CUFFT_INTERNAL_ERROR
#define GPUFFT_INVALID_DEVICE            CUFFT_INVALID_DEVICE
#define GPUFFT_INVALID_PLAN              CUFFT_INVALID_PLAN
#define GPUFFT_INVALID_SIZE              CUFFT_INVALID_SIZE
#define GPUFFT_INVALID_TYPE              CUFFT_INVALID_TYPE
#define GPUFFT_INVALID_VALUE             CUFFT_INVALID_VALUE
#define GPUFFT_NOT_IMPLEMENTED           CUFFT_NOT_IMPLEMENTED
#define GPUFFT_NOT_SUPPORTED             CUFFT_NOT_SUPPORTED
#define GPUFFT_NO_WORKSPACE              CUFFT_NO_WORKSPACE
#define GPUFFT_PARSE_ERROR               CUFFT_PARSE_ERROR
#define GPUFFT_R2C                       CUFFT_R2C
#define GPUFFT_SETUP_FAILED              CUFFT_SETUP_FAILED
#define GPUFFT_SUCCESS                   CUFFT_SUCCESS
#define GPUFFT_UNALIGNED_DATA            CUFFT_UNALIGNED_DATA
#define GPUFFT_Z2D                       CUFFT_Z2D
#define GPUFFT_Z2Z                       CUFFT_Z2Z
#define gpufftComplex                    cufftComplex
#define gpufftCreate                     cufftCreate
#define gpufftDestroy                    cufftDestroy
#define gpufftDoubleComplex              cufftDoubleComplex
#define gpufftDoubleReal                 cufftDoubleReal
#define gpufftEstimate1d                 cufftEstimate1d
#define gpufftEstimate2d                 cufftEstimate2d
#define gpufftEstimate3d                 cufftEstimate3d
#define gpufftEstimateMany               cufftEstimateMany
#define gpufftExecC2C                    cufftExecC2C
#define gpufftExecC2R                    cufftExecC2R
#define gpufftExecD2Z                    cufftExecD2Z
#define gpufftExecR2C                    cufftExecR2C
#define gpufftExecZ2D                    cufftExecZ2D
#define gpufftExecZ2Z                    cufftExecZ2Z
#define gpufftGetProperty                cufftGetProperty
#define gpufftGetSize                    cufftGetSize
#define gpufftGetSize1d                  cufftGetSize1d
#define gpufftGetSize2d                  cufftGetSize2d
#define gpufftGetSize3d                  cufftGetSize3d
#define gpufftGetSizeMany                cufftGetSizeMany
#define gpufftGetSizeMany64              cufftGetSizeMany64
#define gpufftGetVersion                 cufftGetVersion
#define gpufftHandle                     cufftHandle
#define gpufftMakePlan1d                 cufftMakePlan1d
#define gpufftMakePlan2d                 cufftMakePlan2d
#define gpufftMakePlan3d                 cufftMakePlan3d
#define gpufftMakePlanMany               cufftMakePlanMany
#define gpufftMakePlanMany64             cufftMakePlanMany64
#define gpufftPlan1d                     cufftPlan1d
#define gpufftPlan2d                     cufftPlan2d
#define gpufftPlan3d                     cufftPlan3d
#define gpufftPlanMany                   cufftPlanMany
#define gpufftReal                       cufftReal
#define gpufftResult                     cufftResult
#define gpufftResult_t                   cufftResult_t
#define gpufftSetAutoAllocation          cufftSetAutoAllocation
#define gpufftSetStream                  cufftSetStream
#define gpufftSetWorkArea                cufftSetWorkArea
#define gpufftType                       cufftType
#define gpufftType_t                     cufftType_t

#endif
