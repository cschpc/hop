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

#ifndef __HOP_HOPRAND_CUDA_H__
#define __HOP_HOPRAND_CUDA_H__

#include <curand.h>

#define GPURAND_RNG_PSEUDO_DEFAULT       CURAND_RNG_PSEUDO_DEFAULT
#define GPURAND_RNG_PSEUDO_MRG32K3A      CURAND_RNG_PSEUDO_MRG32K3A
#define GPURAND_RNG_PSEUDO_MT19937       CURAND_RNG_PSEUDO_MT19937
#define GPURAND_RNG_PSEUDO_MTGP32        CURAND_RNG_PSEUDO_MTGP32
#define GPURAND_RNG_PSEUDO_PHILOX4_32_10 CURAND_RNG_PSEUDO_PHILOX4_32_10
#define GPURAND_RNG_PSEUDO_XORWOW        CURAND_RNG_PSEUDO_XORWOW
#define GPURAND_RNG_QUASI_DEFAULT        CURAND_RNG_QUASI_DEFAULT
#define GPURAND_RNG_QUASI_SCRAMBLED_SOBOL32  \
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL32
#define GPURAND_RNG_QUASI_SCRAMBLED_SOBOL64  \
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL64
#define GPURAND_RNG_QUASI_SOBOL32        CURAND_RNG_QUASI_SOBOL32
#define GPURAND_RNG_QUASI_SOBOL64        CURAND_RNG_QUASI_SOBOL64
#define GPURAND_RNG_TEST                 CURAND_RNG_TEST
#define GPURAND_STATUS_ALLOCATION_FAILED CURAND_STATUS_ALLOCATION_FAILED
#define GPURAND_STATUS_ARCH_MISMATCH     CURAND_STATUS_ARCH_MISMATCH
#define GPURAND_STATUS_DOUBLE_PRECISION_REQUIRED  \
        CURAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define GPURAND_STATUS_INITIALIZATION_FAILED  \
        CURAND_STATUS_INITIALIZATION_FAILED
#define GPURAND_STATUS_INTERNAL_ERROR    CURAND_STATUS_INTERNAL_ERROR
#define GPURAND_STATUS_LAUNCH_FAILURE    CURAND_STATUS_LAUNCH_FAILURE
#define GPURAND_STATUS_LENGTH_NOT_MULTIPLE  \
        CURAND_STATUS_LENGTH_NOT_MULTIPLE
#define GPURAND_STATUS_NOT_INITIALIZED   CURAND_STATUS_NOT_INITIALIZED
#define GPURAND_STATUS_OUT_OF_RANGE      CURAND_STATUS_OUT_OF_RANGE
#define GPURAND_STATUS_PREEXISTING_FAILURE  \
        CURAND_STATUS_PREEXISTING_FAILURE
#define GPURAND_STATUS_SUCCESS           CURAND_STATUS_SUCCESS
#define GPURAND_STATUS_TYPE_ERROR        CURAND_STATUS_TYPE_ERROR
#define GPURAND_STATUS_VERSION_MISMATCH  CURAND_STATUS_VERSION_MISMATCH
#define gpurandCreateGenerator           curandCreateGenerator
#define gpurandCreateGeneratorHost       curandCreateGeneratorHost
#define gpurandCreatePoissonDistribution curandCreatePoissonDistribution
#define gpurandDestroyDistribution       curandDestroyDistribution
#define gpurandDestroyGenerator          curandDestroyGenerator
#define gpurandDiscreteDistribution_st   curandDiscreteDistribution_st
#define gpurandDiscreteDistribution_t    curandDiscreteDistribution_t
#define gpurandGenerate                  curandGenerate
#define gpurandGenerateLogNormal         curandGenerateLogNormal
#define gpurandGenerateLogNormalDouble   curandGenerateLogNormalDouble
#define gpurandGenerateNormal            curandGenerateNormal
#define gpurandGenerateNormalDouble      curandGenerateNormalDouble
#define gpurandGeneratePoisson           curandGeneratePoisson
#define gpurandGenerateSeeds             curandGenerateSeeds
#define gpurandGenerateUniform           curandGenerateUniform
#define gpurandGenerateUniformDouble     curandGenerateUniformDouble
#define gpurandGenerator_st              curandGenerator_st
#define gpurandGenerator_t               curandGenerator_t
#define gpurandGetVersion                curandGetVersion
#define gpurandRngType_t                 curandRngType_t
#define gpurandSetGeneratorOffset        curandSetGeneratorOffset
#define gpurandSetPseudoRandomGeneratorSeed  \
        curandSetPseudoRandomGeneratorSeed
#define gpurandSetQuasiRandomGeneratorDimensions  \
        curandSetQuasiRandomGeneratorDimensions
#define gpurandSetStream                 curandSetStream
#define gpurandStatus_t                  curandStatus_t

#endif
