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

#ifndef __HOP_HOPRAND_HIP_H__
#define __HOP_HOPRAND_HIP_H__

#include <hiprand/hiprand.h>

#define GPURAND_RNG_PSEUDO_DEFAULT       HIPRAND_RNG_PSEUDO_DEFAULT
#define GPURAND_RNG_PSEUDO_MRG32K3A      HIPRAND_RNG_PSEUDO_MRG32K3A
#define GPURAND_RNG_PSEUDO_MT19937       HIPRAND_RNG_PSEUDO_MT19937
#define GPURAND_RNG_PSEUDO_MTGP32        HIPRAND_RNG_PSEUDO_MTGP32
#define GPURAND_RNG_PSEUDO_PHILOX4_32_10 HIPRAND_RNG_PSEUDO_PHILOX4_32_10
#define GPURAND_RNG_PSEUDO_XORWOW        HIPRAND_RNG_PSEUDO_XORWOW
#define GPURAND_RNG_QUASI_DEFAULT        HIPRAND_RNG_QUASI_DEFAULT
#define GPURAND_RNG_QUASI_SCRAMBLED_SOBOL32  \
        HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
#define GPURAND_RNG_QUASI_SCRAMBLED_SOBOL64  \
        HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
#define GPURAND_RNG_QUASI_SOBOL32        HIPRAND_RNG_QUASI_SOBOL32
#define GPURAND_RNG_QUASI_SOBOL64        HIPRAND_RNG_QUASI_SOBOL64
#define GPURAND_RNG_TEST                 HIPRAND_RNG_TEST
#define GPURAND_STATUS_ALLOCATION_FAILED HIPRAND_STATUS_ALLOCATION_FAILED
#define GPURAND_STATUS_ARCH_MISMATCH     HIPRAND_STATUS_ARCH_MISMATCH
#define GPURAND_STATUS_DOUBLE_PRECISION_REQUIRED  \
        HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define GPURAND_STATUS_INITIALIZATION_FAILED  \
        HIPRAND_STATUS_INITIALIZATION_FAILED
#define GPURAND_STATUS_INTERNAL_ERROR    HIPRAND_STATUS_INTERNAL_ERROR
#define GPURAND_STATUS_LAUNCH_FAILURE    HIPRAND_STATUS_LAUNCH_FAILURE
#define GPURAND_STATUS_LENGTH_NOT_MULTIPLE  \
        HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
#define GPURAND_STATUS_NOT_INITIALIZED   HIPRAND_STATUS_NOT_INITIALIZED
#define GPURAND_STATUS_OUT_OF_RANGE      HIPRAND_STATUS_OUT_OF_RANGE
#define GPURAND_STATUS_PREEXISTING_FAILURE  \
        HIPRAND_STATUS_PREEXISTING_FAILURE
#define GPURAND_STATUS_SUCCESS           HIPRAND_STATUS_SUCCESS
#define GPURAND_STATUS_TYPE_ERROR        HIPRAND_STATUS_TYPE_ERROR
#define GPURAND_STATUS_VERSION_MISMATCH  HIPRAND_STATUS_VERSION_MISMATCH
#define gpurandCreateGenerator           hiprandCreateGenerator
#define gpurandCreateGeneratorHost       hiprandCreateGeneratorHost
#define gpurandCreatePoissonDistribution hiprandCreatePoissonDistribution
#define gpurandDestroyDistribution       hiprandDestroyDistribution
#define gpurandDestroyGenerator          hiprandDestroyGenerator
#define gpurandDiscreteDistribution_st   hiprandDiscreteDistribution_st
#define gpurandDiscreteDistribution_t    hiprandDiscreteDistribution_t
#define gpurandGenerate                  hiprandGenerate
#define gpurandGenerateLogNormal         hiprandGenerateLogNormal
#define gpurandGenerateLogNormalDouble   hiprandGenerateLogNormalDouble
#define gpurandGenerateNormal            hiprandGenerateNormal
#define gpurandGenerateNormalDouble      hiprandGenerateNormalDouble
#define gpurandGeneratePoisson           hiprandGeneratePoisson
#define gpurandGenerateSeeds             hiprandGenerateSeeds
#define gpurandGenerateUniform           hiprandGenerateUniform
#define gpurandGenerateUniformDouble     hiprandGenerateUniformDouble
#define gpurandGenerator_st              hiprandGenerator_st
#define gpurandGenerator_t               hiprandGenerator_t
#define gpurandGetVersion                hiprandGetVersion
#define gpurandRngType_t                 hiprandRngType_t
#define gpurandSetGeneratorOffset        hiprandSetGeneratorOffset
#define gpurandSetPseudoRandomGeneratorSeed  \
        hiprandSetPseudoRandomGeneratorSeed
#define gpurandSetQuasiRandomGeneratorDimensions  \
        hiprandSetQuasiRandomGeneratorDimensions
#define gpurandSetStream                 hiprandSetStream
#define gpurandStatus_t                  hiprandStatus_t

#endif
