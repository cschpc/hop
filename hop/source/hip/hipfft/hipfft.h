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

#ifndef __HOP_SOURCE_HIP_HIPFFT_H__
#define __HOP_SOURCE_HIP_HIPFFT_H__

#define HOP_SOURCE_HIP

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>

#define HIPFFT_ALLOC_FAILED              GPUFFT_ALLOC_FAILED
#define HIPFFT_BACKWARD                  GPUFFT_BACKWARD
#define HIPFFT_C2C                       GPUFFT_C2C
#define HIPFFT_C2R                       GPUFFT_C2R
#define HIPFFT_D2Z                       GPUFFT_D2Z
#define HIPFFT_EXEC_FAILED               GPUFFT_EXEC_FAILED
#define HIPFFT_FORWARD                   GPUFFT_FORWARD
#define HIPFFT_INCOMPLETE_PARAMETER_LIST GPUFFT_INCOMPLETE_PARAMETER_LIST
#define HIPFFT_INTERNAL_ERROR            GPUFFT_INTERNAL_ERROR
#define HIPFFT_INVALID_DEVICE            GPUFFT_INVALID_DEVICE
#define HIPFFT_INVALID_PLAN              GPUFFT_INVALID_PLAN
#define HIPFFT_INVALID_SIZE              GPUFFT_INVALID_SIZE
#define HIPFFT_INVALID_TYPE              GPUFFT_INVALID_TYPE
#define HIPFFT_INVALID_VALUE             GPUFFT_INVALID_VALUE
#define HIPFFT_NOT_IMPLEMENTED           GPUFFT_NOT_IMPLEMENTED
#define HIPFFT_NOT_SUPPORTED             GPUFFT_NOT_SUPPORTED
#define HIPFFT_NO_WORKSPACE              GPUFFT_NO_WORKSPACE
#define HIPFFT_PARSE_ERROR               GPUFFT_PARSE_ERROR
#define HIPFFT_R2C                       GPUFFT_R2C
#define HIPFFT_SETUP_FAILED              GPUFFT_SETUP_FAILED
#define HIPFFT_SUCCESS                   GPUFFT_SUCCESS
#define HIPFFT_UNALIGNED_DATA            GPUFFT_UNALIGNED_DATA
#define HIPFFT_Z2D                       GPUFFT_Z2D
#define HIPFFT_Z2Z                       GPUFFT_Z2Z
#define hipfftComplex                    gpufftComplex
#define hipfftCreate                     gpufftCreate
#define hipfftDestroy                    gpufftDestroy
#define hipfftDoubleComplex              gpufftDoubleComplex
#define hipfftDoubleReal                 gpufftDoubleReal
#define hipfftEstimate1d                 gpufftEstimate1d
#define hipfftEstimate2d                 gpufftEstimate2d
#define hipfftEstimate3d                 gpufftEstimate3d
#define hipfftEstimateMany               gpufftEstimateMany
#define hipfftExecC2C                    gpufftExecC2C
#define hipfftExecC2R                    gpufftExecC2R
#define hipfftExecD2Z                    gpufftExecD2Z
#define hipfftExecR2C                    gpufftExecR2C
#define hipfftExecZ2D                    gpufftExecZ2D
#define hipfftExecZ2Z                    gpufftExecZ2Z
#define hipfftGetProperty                gpufftGetProperty
#define hipfftGetSize                    gpufftGetSize
#define hipfftGetSize1d                  gpufftGetSize1d
#define hipfftGetSize2d                  gpufftGetSize2d
#define hipfftGetSize3d                  gpufftGetSize3d
#define hipfftGetSizeMany                gpufftGetSizeMany
#define hipfftGetSizeMany64              gpufftGetSizeMany64
#define hipfftGetVersion                 gpufftGetVersion
#define hipfftHandle                     gpufftHandle
#define hipfftMakePlan1d                 gpufftMakePlan1d
#define hipfftMakePlan2d                 gpufftMakePlan2d
#define hipfftMakePlan3d                 gpufftMakePlan3d
#define hipfftMakePlanMany               gpufftMakePlanMany
#define hipfftMakePlanMany64             gpufftMakePlanMany64
#define hipfftPlan1d                     gpufftPlan1d
#define hipfftPlan2d                     gpufftPlan2d
#define hipfftPlan3d                     gpufftPlan3d
#define hipfftPlanMany                   gpufftPlanMany
#define hipfftReal                       gpufftReal
#define hipfftResult                     gpufftResult
#define hipfftResult_t                   gpufftResult_t
#define hipfftSetAutoAllocation          gpufftSetAutoAllocation
#define hipfftSetStream                  gpufftSetStream
#define hipfftSetWorkArea                gpufftSetWorkArea
#define hipfftType                       gpufftType
#define hipfftType_t                     gpufftType_t

#include <hop/hopfft.h>

#endif
