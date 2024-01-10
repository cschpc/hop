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

#ifndef __HOP_SOURCE_HIP_HIPRAND_H__
#define __HOP_SOURCE_HIP_HIPRAND_H__

#if !defined(HOP_SOURCE_HIP)
#define HOP_SOURCE_HIP
#endif

#include <hip/hip_runtime.h>

#define HIPRAND_RNG_PSEUDO_DEFAULT       GPURAND_RNG_PSEUDO_DEFAULT
#define HIPRAND_RNG_PSEUDO_MRG32K3A      GPURAND_RNG_PSEUDO_MRG32K3A
#define HIPRAND_RNG_PSEUDO_MT19937       GPURAND_RNG_PSEUDO_MT19937
#define HIPRAND_RNG_PSEUDO_MTGP32        GPURAND_RNG_PSEUDO_MTGP32
#define HIPRAND_RNG_PSEUDO_PHILOX4_32_10 GPURAND_RNG_PSEUDO_PHILOX4_32_10
#define HIPRAND_RNG_PSEUDO_XORWOW        GPURAND_RNG_PSEUDO_XORWOW
#define HIPRAND_RNG_QUASI_DEFAULT        GPURAND_RNG_QUASI_DEFAULT
#define HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32  \
        GPURAND_RNG_QUASI_SCRAMBLED_SOBOL32
#define HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64  \
        GPURAND_RNG_QUASI_SCRAMBLED_SOBOL64
#define HIPRAND_RNG_QUASI_SOBOL32        GPURAND_RNG_QUASI_SOBOL32
#define HIPRAND_RNG_QUASI_SOBOL64        GPURAND_RNG_QUASI_SOBOL64
#define HIPRAND_RNG_TEST                 GPURAND_RNG_TEST
#define HIPRAND_STATUS_ALLOCATION_FAILED GPURAND_STATUS_ALLOCATION_FAILED
#define HIPRAND_STATUS_ARCH_MISMATCH     GPURAND_STATUS_ARCH_MISMATCH
#define HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED  \
        GPURAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define HIPRAND_STATUS_INITIALIZATION_FAILED  \
        GPURAND_STATUS_INITIALIZATION_FAILED
#define HIPRAND_STATUS_INTERNAL_ERROR    GPURAND_STATUS_INTERNAL_ERROR
#define HIPRAND_STATUS_LAUNCH_FAILURE    GPURAND_STATUS_LAUNCH_FAILURE
#define HIPRAND_STATUS_LENGTH_NOT_MULTIPLE  \
        GPURAND_STATUS_LENGTH_NOT_MULTIPLE
#define HIPRAND_STATUS_NOT_INITIALIZED   GPURAND_STATUS_NOT_INITIALIZED
#define HIPRAND_STATUS_OUT_OF_RANGE      GPURAND_STATUS_OUT_OF_RANGE
#define HIPRAND_STATUS_PREEXISTING_FAILURE  \
        GPURAND_STATUS_PREEXISTING_FAILURE
#define HIPRAND_STATUS_SUCCESS           GPURAND_STATUS_SUCCESS
#define HIPRAND_STATUS_TYPE_ERROR        GPURAND_STATUS_TYPE_ERROR
#define HIPRAND_STATUS_VERSION_MISMATCH  GPURAND_STATUS_VERSION_MISMATCH
#define hiprandCreateGenerator           gpurandCreateGenerator
#define hiprandCreateGeneratorHost       gpurandCreateGeneratorHost
#define hiprandCreatePoissonDistribution gpurandCreatePoissonDistribution
#define hiprandDestroyDistribution       gpurandDestroyDistribution
#define hiprandDestroyGenerator          gpurandDestroyGenerator
#define hiprandDiscreteDistribution_st   gpurandDiscreteDistribution_st
#define hiprandDiscreteDistribution_t    gpurandDiscreteDistribution_t
#define hiprandGenerate                  gpurandGenerate
#define hiprandGenerateLogNormal         gpurandGenerateLogNormal
#define hiprandGenerateLogNormalDouble   gpurandGenerateLogNormalDouble
#define hiprandGenerateNormal            gpurandGenerateNormal
#define hiprandGenerateNormalDouble      gpurandGenerateNormalDouble
#define hiprandGeneratePoisson           gpurandGeneratePoisson
#define hiprandGenerateSeeds             gpurandGenerateSeeds
#define hiprandGenerateUniform           gpurandGenerateUniform
#define hiprandGenerateUniformDouble     gpurandGenerateUniformDouble
#define hiprandGenerator_st              gpurandGenerator_st
#define hiprandGenerator_t               gpurandGenerator_t
#define hiprandGetVersion                gpurandGetVersion
#define hiprandRngType_t                 gpurandRngType_t
#define hiprandSetGeneratorOffset        gpurandSetGeneratorOffset
#define hiprandSetPseudoRandomGeneratorSeed  \
        gpurandSetPseudoRandomGeneratorSeed
#define hiprandSetQuasiRandomGeneratorDimensions  \
        gpurandSetQuasiRandomGeneratorDimensions
#define hiprandSetStream                 gpurandSetStream
#define hiprandStatus_t                  gpurandStatus_t

/* hiprand/hiprand_mtgp32_host.h */
#define hiprandMakeMTGP32Constants       gpurandMakeMTGP32Constants
#define hiprandMakeMTGP32KernelState     gpurandMakeMTGP32KernelState

#include <hop/hoprand.h>

#endif
