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

#ifndef __HOP_SOURCE_CUDA_CURAND_H__
#define __HOP_SOURCE_CUDA_CURAND_H__

#define HOP_SOURCE_CUDA

#include <cuda_runtime.h>

#define CURAND_RNG_PSEUDO_DEFAULT        GPURAND_RNG_PSEUDO_DEFAULT
#define CURAND_RNG_PSEUDO_MRG32K3A       GPURAND_RNG_PSEUDO_MRG32K3A
#define CURAND_RNG_PSEUDO_MT19937        GPURAND_RNG_PSEUDO_MT19937
#define CURAND_RNG_PSEUDO_MTGP32         GPURAND_RNG_PSEUDO_MTGP32
#define CURAND_RNG_PSEUDO_PHILOX4_32_10  GPURAND_RNG_PSEUDO_PHILOX4_32_10
#define CURAND_RNG_PSEUDO_XORWOW         GPURAND_RNG_PSEUDO_XORWOW
#define CURAND_RNG_QUASI_DEFAULT         GPURAND_RNG_QUASI_DEFAULT
#define CURAND_RNG_QUASI_SCRAMBLED_SOBOL32  \
        GPURAND_RNG_QUASI_SCRAMBLED_SOBOL32
#define CURAND_RNG_QUASI_SCRAMBLED_SOBOL64  \
        GPURAND_RNG_QUASI_SCRAMBLED_SOBOL64
#define CURAND_RNG_QUASI_SOBOL32         GPURAND_RNG_QUASI_SOBOL32
#define CURAND_RNG_QUASI_SOBOL64         GPURAND_RNG_QUASI_SOBOL64
#define CURAND_RNG_TEST                  GPURAND_RNG_TEST
#define CURAND_STATUS_ALLOCATION_FAILED  GPURAND_STATUS_ALLOCATION_FAILED
#define CURAND_STATUS_ARCH_MISMATCH      GPURAND_STATUS_ARCH_MISMATCH
#define CURAND_STATUS_DOUBLE_PRECISION_REQUIRED  \
        GPURAND_STATUS_DOUBLE_PRECISION_REQUIRED
#define CURAND_STATUS_INITIALIZATION_FAILED  \
        GPURAND_STATUS_INITIALIZATION_FAILED
#define CURAND_STATUS_INTERNAL_ERROR     GPURAND_STATUS_INTERNAL_ERROR
#define CURAND_STATUS_LAUNCH_FAILURE     GPURAND_STATUS_LAUNCH_FAILURE
#define CURAND_STATUS_LENGTH_NOT_MULTIPLE  \
        GPURAND_STATUS_LENGTH_NOT_MULTIPLE
#define CURAND_STATUS_NOT_INITIALIZED    GPURAND_STATUS_NOT_INITIALIZED
#define CURAND_STATUS_OUT_OF_RANGE       GPURAND_STATUS_OUT_OF_RANGE
#define CURAND_STATUS_PREEXISTING_FAILURE  \
        GPURAND_STATUS_PREEXISTING_FAILURE
#define CURAND_STATUS_SUCCESS            GPURAND_STATUS_SUCCESS
#define CURAND_STATUS_TYPE_ERROR         GPURAND_STATUS_TYPE_ERROR
#define CURAND_STATUS_VERSION_MISMATCH   GPURAND_STATUS_VERSION_MISMATCH
#define curandCreateGenerator            gpurandCreateGenerator
#define curandCreateGeneratorHost        gpurandCreateGeneratorHost
#define curandCreatePoissonDistribution  gpurandCreatePoissonDistribution
#define curandDestroyDistribution        gpurandDestroyDistribution
#define curandDestroyGenerator           gpurandDestroyGenerator
#define curandDirectionVectors32_t       gpurandDirectionVectors32_t
#define curandDiscreteDistribution_t     gpurandDiscreteDistribution_t
#define curandGenerate                   gpurandGenerate
#define curandGenerateLogNormal          gpurandGenerateLogNormal
#define curandGenerateLogNormalDouble    gpurandGenerateLogNormalDouble
#define curandGenerateNormal             gpurandGenerateNormal
#define curandGenerateNormalDouble       gpurandGenerateNormalDouble
#define curandGeneratePoisson            gpurandGeneratePoisson
#define curandGenerateSeeds              gpurandGenerateSeeds
#define curandGenerateUniform            gpurandGenerateUniform
#define curandGenerateUniformDouble      gpurandGenerateUniformDouble
#define curandGenerator_st               gpurandGenerator_st
#define curandGenerator_t                gpurandGenerator_t
#define curandGetVersion                 gpurandGetVersion
#define curandRngType                    gpurandRngType_t
#define curandRngType_t                  gpurandRngType_t
#define curandSetGeneratorOffset         gpurandSetGeneratorOffset
#define curandSetPseudoRandomGeneratorSeed  \
        gpurandSetPseudoRandomGeneratorSeed
#define curandSetQuasiRandomGeneratorDimensions  \
        gpurandSetQuasiRandomGeneratorDimensions
#define curandSetStream                  gpurandSetStream
#define curandStatus                     gpurandStatus_t
#define curandStatus_t                   gpurandStatus_t

/* curand_discrete.h */
#define curandDiscreteDistribution_st    gpurandDiscreteDistribution_st

/* curand_kernel.h */
#define curand                           gpurand
#define curandState                      gpurandState
#define curandStateMRG32k3a              gpurandStateMRG32k3a
#define curandStateMRG32k3a_t            gpurandStateMRG32k3a_t
#define curandStatePhilox4_32_10         gpurandStatePhilox4_32_10
#define curandStatePhilox4_32_10_t       gpurandStatePhilox4_32_10_t
#define curandStateSobol32               gpurandStateSobol32
#define curandStateSobol32_t             gpurandStateSobol32_t
#define curandStateXORWOW                gpurandStateXORWOW
#define curandStateXORWOW_t              gpurandStateXORWOW_t
#define curandState_t                    gpurandState_t
#define curand_discrete                  gpurand_discrete
#define curand_discrete4                 gpurand_discrete4
#define curand_init                      gpurand_init
#define curand_log_normal                gpurand_log_normal
#define curand_log_normal2               gpurand_log_normal2
#define curand_log_normal2_double        gpurand_log_normal2_double
#define curand_log_normal4               gpurand_log_normal4
#define curand_log_normal4_double        gpurand_log_normal4_double
#define curand_log_normal_double         gpurand_log_normal_double
#define curand_normal                    gpurand_normal
#define curand_normal2                   gpurand_normal2
#define curand_normal2_double            gpurand_normal2_double
#define curand_normal4                   gpurand_normal4
#define curand_normal4_double            gpurand_normal4_double
#define curand_normal_double             gpurand_normal_double
#define curand_poisson                   gpurand_poisson
#define curand_poisson4                  gpurand_poisson4
#define curand_uniform                   gpurand_uniform
#define curand_uniform2_double           gpurand_uniform2_double
#define curand_uniform4                  gpurand_uniform4
#define curand_uniform4_double           gpurand_uniform4_double
#define curand_uniform_double            gpurand_uniform_double

/* curand_mtgp32.h */
#define curandStateMtgp32                gpurandStateMtgp32
#define curandStateMtgp32_t              gpurandStateMtgp32_t

/* curand_mtgp32_host.h */
#define curandMakeMTGP32Constants        gpurandMakeMTGP32Constants
#define curandMakeMTGP32KernelState      gpurandMakeMTGP32KernelState

#include <hop/hoprand.h>

#endif
