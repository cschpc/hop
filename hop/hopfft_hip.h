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

#ifndef __HOP_HOPFFT_HIP_H__
#define __HOP_HOPFFT_HIP_H__

#include <hipfft/hipfft.h>

#define GPUFFT_ALLOC_FAILED              HIPFFT_ALLOC_FAILED
#define GPUFFT_BACKWARD                  HIPFFT_BACKWARD
#define GPUFFT_C2C                       HIPFFT_C2C
#define GPUFFT_C2R                       HIPFFT_C2R
#define GPUFFT_D2Z                       HIPFFT_D2Z
#define GPUFFT_EXEC_FAILED               HIPFFT_EXEC_FAILED
#define GPUFFT_FORWARD                   HIPFFT_FORWARD
#define GPUFFT_INCOMPLETE_PARAMETER_LIST HIPFFT_INCOMPLETE_PARAMETER_LIST
#define GPUFFT_INTERNAL_ERROR            HIPFFT_INTERNAL_ERROR
#define GPUFFT_INVALID_DEVICE            HIPFFT_INVALID_DEVICE
#define GPUFFT_INVALID_PLAN              HIPFFT_INVALID_PLAN
#define GPUFFT_INVALID_SIZE              HIPFFT_INVALID_SIZE
#define GPUFFT_INVALID_TYPE              HIPFFT_INVALID_TYPE
#define GPUFFT_INVALID_VALUE             HIPFFT_INVALID_VALUE
#define GPUFFT_NOT_IMPLEMENTED           HIPFFT_NOT_IMPLEMENTED
#define GPUFFT_NOT_SUPPORTED             HIPFFT_NOT_SUPPORTED
#define GPUFFT_NO_WORKSPACE              HIPFFT_NO_WORKSPACE
#define GPUFFT_PARSE_ERROR               HIPFFT_PARSE_ERROR
#define GPUFFT_R2C                       HIPFFT_R2C
#define GPUFFT_SETUP_FAILED              HIPFFT_SETUP_FAILED
#define GPUFFT_SUCCESS                   HIPFFT_SUCCESS
#define GPUFFT_UNALIGNED_DATA            HIPFFT_UNALIGNED_DATA
#define GPUFFT_Z2D                       HIPFFT_Z2D
#define GPUFFT_Z2Z                       HIPFFT_Z2Z
#define gpufftComplex                    hipfftComplex
#define gpufftCreate                     hipfftCreate
#define gpufftDestroy                    hipfftDestroy
#define gpufftDoubleComplex              hipfftDoubleComplex
#define gpufftDoubleReal                 hipfftDoubleReal
#define gpufftEstimate1d                 hipfftEstimate1d
#define gpufftEstimate2d                 hipfftEstimate2d
#define gpufftEstimate3d                 hipfftEstimate3d
#define gpufftEstimateMany               hipfftEstimateMany
#define gpufftExecC2C                    hipfftExecC2C
#define gpufftExecC2R                    hipfftExecC2R
#define gpufftExecD2Z                    hipfftExecD2Z
#define gpufftExecR2C                    hipfftExecR2C
#define gpufftExecZ2D                    hipfftExecZ2D
#define gpufftExecZ2Z                    hipfftExecZ2Z
#define gpufftGetProperty                hipfftGetProperty
#define gpufftGetSize                    hipfftGetSize
#define gpufftGetSize1d                  hipfftGetSize1d
#define gpufftGetSize2d                  hipfftGetSize2d
#define gpufftGetSize3d                  hipfftGetSize3d
#define gpufftGetSizeMany                hipfftGetSizeMany
#define gpufftGetSizeMany64              hipfftGetSizeMany64
#define gpufftGetVersion                 hipfftGetVersion
#define gpufftHandle                     hipfftHandle
#define gpufftMakePlan1d                 hipfftMakePlan1d
#define gpufftMakePlan2d                 hipfftMakePlan2d
#define gpufftMakePlan3d                 hipfftMakePlan3d
#define gpufftMakePlanMany               hipfftMakePlanMany
#define gpufftMakePlanMany64             hipfftMakePlanMany64
#define gpufftPlan1d                     hipfftPlan1d
#define gpufftPlan2d                     hipfftPlan2d
#define gpufftPlan3d                     hipfftPlan3d
#define gpufftPlanMany                   hipfftPlanMany
#define gpufftReal                       hipfftReal
#define gpufftResult                     hipfftResult
#define gpufftResult_t                   hipfftResult_t
#define gpufftSetAutoAllocation          hipfftSetAutoAllocation
#define gpufftSetStream                  hipfftSetStream
#define gpufftSetWorkArea                hipfftSetWorkArea
#define gpufftType                       hipfftType
#define gpufftType_t                     hipfftType_t

#endif
