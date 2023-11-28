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

#ifndef __HOP_SOURCE_HIP_HIPBLAS_H__
#define __HOP_SOURCE_HIP_HIPBLAS_H__

#define HOP_SOURCE_HIP

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>

#define HIPBLAS_ATOMICS_ALLOWED          GPUBLAS_ATOMICS_ALLOWED
#define HIPBLAS_ATOMICS_NOT_ALLOWED      GPUBLAS_ATOMICS_NOT_ALLOWED
#define HIPBLAS_C_16B                    GPUBLAS_C_16B
#define HIPBLAS_C_16F                    GPUBLAS_C_16F
#define HIPBLAS_C_32F                    GPUBLAS_C_32F
#define HIPBLAS_C_32I                    GPUBLAS_C_32I
#define HIPBLAS_C_32U                    GPUBLAS_C_32U
#define HIPBLAS_C_64F                    GPUBLAS_C_64F
#define HIPBLAS_C_8I                     GPUBLAS_C_8I
#define HIPBLAS_C_8U                     GPUBLAS_C_8U
#define HIPBLAS_DIAG_NON_UNIT            GPUBLAS_DIAG_NON_UNIT
#define HIPBLAS_DIAG_UNIT                GPUBLAS_DIAG_UNIT
#define HIPBLAS_FILL_MODE_FULL           GPUBLAS_FILL_MODE_FULL
#define HIPBLAS_FILL_MODE_LOWER          GPUBLAS_FILL_MODE_LOWER
#define HIPBLAS_FILL_MODE_UPPER          GPUBLAS_FILL_MODE_UPPER
#define HIPBLAS_GEMM_DEFAULT             GPUBLAS_GEMM_DEFAULT
#define HIPBLAS_OP_C                     GPUBLAS_OP_C
#define HIPBLAS_OP_N                     GPUBLAS_OP_N
#define HIPBLAS_OP_T                     GPUBLAS_OP_T
#define HIPBLAS_POINTER_MODE_DEVICE      GPUBLAS_POINTER_MODE_DEVICE
#define HIPBLAS_POINTER_MODE_HOST        GPUBLAS_POINTER_MODE_HOST
#define HIPBLAS_R_16B                    GPUBLAS_R_16B
#define HIPBLAS_R_16F                    GPUBLAS_R_16F
#define HIPBLAS_R_32F                    GPUBLAS_R_32F
#define HIPBLAS_R_32I                    GPUBLAS_R_32I
#define HIPBLAS_R_32U                    GPUBLAS_R_32U
#define HIPBLAS_R_64F                    GPUBLAS_R_64F
#define HIPBLAS_R_8I                     GPUBLAS_R_8I
#define HIPBLAS_R_8U                     GPUBLAS_R_8U
#define HIPBLAS_SIDE_LEFT                GPUBLAS_SIDE_LEFT
#define HIPBLAS_SIDE_RIGHT               GPUBLAS_SIDE_RIGHT
#define HIPBLAS_STATUS_ALLOC_FAILED      GPUBLAS_STATUS_ALLOC_FAILED
#define HIPBLAS_STATUS_ARCH_MISMATCH     GPUBLAS_STATUS_ARCH_MISMATCH
#define HIPBLAS_STATUS_EXECUTION_FAILED  GPUBLAS_STATUS_EXECUTION_FAILED
#define HIPBLAS_STATUS_INTERNAL_ERROR    GPUBLAS_STATUS_INTERNAL_ERROR
#define HIPBLAS_STATUS_INVALID_VALUE     GPUBLAS_STATUS_INVALID_VALUE
#define HIPBLAS_STATUS_MAPPING_ERROR     GPUBLAS_STATUS_MAPPING_ERROR
#define HIPBLAS_STATUS_NOT_INITIALIZED   GPUBLAS_STATUS_NOT_INITIALIZED
#define HIPBLAS_STATUS_NOT_SUPPORTED     GPUBLAS_STATUS_NOT_SUPPORTED
#define HIPBLAS_STATUS_SUCCESS           GPUBLAS_STATUS_SUCCESS
#define HIPBLAS_STATUS_UNKNOWN           GPUBLAS_STATUS_UNKNOWN
#define hipblasAtomicsMode_t             gpublasAtomicsMode_t
#define hipblasAxpyEx                    gpublasAxpyEx
#define hipblasCaxpy                     gpublasCaxpy
#define hipblasCcopy                     gpublasCcopy
#define hipblasCdgmm                     gpublasCdgmm
#define hipblasCdotc                     gpublasCdotc
#define hipblasCdotu                     gpublasCdotu
#define hipblasCgbmv                     gpublasCgbmv
#define hipblasCgeam                     gpublasCgeam
#define hipblasCgelsBatched              gpublasCgelsBatched
#define hipblasCgemm                     gpublasCgemm
#define hipblasCgemmBatched              gpublasCgemmBatched
#define hipblasCgemmStridedBatched       gpublasCgemmStridedBatched
#define hipblasCgemv                     gpublasCgemv
#define hipblasCgeqrfBatched             gpublasCgeqrfBatched
#define hipblasCgerc                     gpublasCgerc
#define hipblasCgeru                     gpublasCgeru
#define hipblasCgetrfBatched             gpublasCgetrfBatched
#define hipblasCgetriBatched             gpublasCgetriBatched
#define hipblasCgetrsBatched             gpublasCgetrsBatched
#define hipblasChbmv                     gpublasChbmv
#define hipblasChemm                     gpublasChemm
#define hipblasChemv                     gpublasChemv
#define hipblasCher                      gpublasCher
#define hipblasCher2                     gpublasCher2
#define hipblasCher2k                    gpublasCher2k
#define hipblasCherk                     gpublasCherk
#define hipblasCherkx                    gpublasCherkx
#define hipblasChpmv                     gpublasChpmv
#define hipblasChpr                      gpublasChpr
#define hipblasChpr2                     gpublasChpr2
#define hipblasCreate                    gpublasCreate
#define hipblasCrot                      gpublasCrot
#define hipblasCrotg                     gpublasCrotg
#define hipblasCscal                     gpublasCscal
#define hipblasCsrot                     gpublasCsrot
#define hipblasCsscal                    gpublasCsscal
#define hipblasCswap                     gpublasCswap
#define hipblasCsymm                     gpublasCsymm
#define hipblasCsymv                     gpublasCsymv
#define hipblasCsyr                      gpublasCsyr
#define hipblasCsyr2                     gpublasCsyr2
#define hipblasCsyr2k                    gpublasCsyr2k
#define hipblasCsyrk                     gpublasCsyrk
#define hipblasCsyrkx                    gpublasCsyrkx
#define hipblasCtbmv                     gpublasCtbmv
#define hipblasCtbsv                     gpublasCtbsv
#define hipblasCtpmv                     gpublasCtpmv
#define hipblasCtpsv                     gpublasCtpsv
#define hipblasCtrmm                     gpublasCtrmm
#define hipblasCtrmv                     gpublasCtrmv
#define hipblasCtrsm                     gpublasCtrsm
#define hipblasCtrsmBatched              gpublasCtrsmBatched
#define hipblasCtrsv                     gpublasCtrsv
#define hipblasDasum                     gpublasDasum
#define hipblasDatatype_t                gpublasDatatype_t
#define hipblasDaxpy                     gpublasDaxpy
#define hipblasDcopy                     gpublasDcopy
#define hipblasDdgmm                     gpublasDdgmm
#define hipblasDdot                      gpublasDdot
#define hipblasDestroy                   gpublasDestroy
#define hipblasDgbmv                     gpublasDgbmv
#define hipblasDgeam                     gpublasDgeam
#define hipblasDgelsBatched              gpublasDgelsBatched
#define hipblasDgemm                     gpublasDgemm
#define hipblasDgemmBatched              gpublasDgemmBatched
#define hipblasDgemmStridedBatched       gpublasDgemmStridedBatched
#define hipblasDgemv                     gpublasDgemv
#define hipblasDgeqrfBatched             gpublasDgeqrfBatched
#define hipblasDger                      gpublasDger
#define hipblasDgetrfBatched             gpublasDgetrfBatched
#define hipblasDgetriBatched             gpublasDgetriBatched
#define hipblasDgetrsBatched             gpublasDgetrsBatched
#define hipblasDiagType_t                gpublasDiagType_t
#define hipblasDnrm2                     gpublasDnrm2
#define hipblasDotEx                     gpublasDotEx
#define hipblasDotcEx                    gpublasDotcEx
#define hipblasDrot                      gpublasDrot
#define hipblasDrotg                     gpublasDrotg
#define hipblasDrotm                     gpublasDrotm
#define hipblasDrotmg                    gpublasDrotmg
#define hipblasDsbmv                     gpublasDsbmv
#define hipblasDscal                     gpublasDscal
#define hipblasDspmv                     gpublasDspmv
#define hipblasDspr                      gpublasDspr
#define hipblasDspr2                     gpublasDspr2
#define hipblasDswap                     gpublasDswap
#define hipblasDsymm                     gpublasDsymm
#define hipblasDsymv                     gpublasDsymv
#define hipblasDsyr                      gpublasDsyr
#define hipblasDsyr2                     gpublasDsyr2
#define hipblasDsyr2k                    gpublasDsyr2k
#define hipblasDsyrk                     gpublasDsyrk
#define hipblasDsyrkx                    gpublasDsyrkx
#define hipblasDtbmv                     gpublasDtbmv
#define hipblasDtbsv                     gpublasDtbsv
#define hipblasDtpmv                     gpublasDtpmv
#define hipblasDtpsv                     gpublasDtpsv
#define hipblasDtrmm                     gpublasDtrmm
#define hipblasDtrmv                     gpublasDtrmv
#define hipblasDtrsm                     gpublasDtrsm
#define hipblasDtrsmBatched              gpublasDtrsmBatched
#define hipblasDtrsv                     gpublasDtrsv
#define hipblasDzasum                    gpublasDzasum
#define hipblasDznrm2                    gpublasDznrm2
#define hipblasFillMode_t                gpublasFillMode_t
#define hipblasGemmAlgo_t                gpublasGemmAlgo_t
#define hipblasGemmBatchedEx             gpublasGemmBatchedEx
#define hipblasGemmEx                    gpublasGemmEx
#define hipblasGemmStridedBatchedEx      gpublasGemmStridedBatchedEx
#define hipblasGetAtomicsMode            gpublasGetAtomicsMode
#define hipblasGetMatrix                 gpublasGetMatrix
#define hipblasGetMatrixAsync            gpublasGetMatrixAsync
#define hipblasGetPointerMode            gpublasGetPointerMode
#define hipblasGetStream                 gpublasGetStream
#define hipblasGetVector                 gpublasGetVector
#define hipblasGetVectorAsync            gpublasGetVectorAsync
#define hipblasHandle_t                  gpublasHandle_t
#define hipblasHgemm                     gpublasHgemm
#define hipblasHgemmBatched              gpublasHgemmBatched
#define hipblasHgemmStridedBatched       gpublasHgemmStridedBatched
#define hipblasIcamax                    gpublasIcamax
#define hipblasIcamin                    gpublasIcamin
#define hipblasIdamax                    gpublasIdamax
#define hipblasIdamin                    gpublasIdamin
#define hipblasIsamax                    gpublasIsamax
#define hipblasIsamin                    gpublasIsamin
#define hipblasIzamax                    gpublasIzamax
#define hipblasIzamin                    gpublasIzamin
#define hipblasNrm2Ex                    gpublasNrm2Ex
#define hipblasOperation_t               gpublasOperation_t
#define hipblasPointerMode_t             gpublasPointerMode_t
#define hipblasRotEx                     gpublasRotEx
#define hipblasSasum                     gpublasSasum
#define hipblasSaxpy                     gpublasSaxpy
#define hipblasScalEx                    gpublasScalEx
#define hipblasScasum                    gpublasScasum
#define hipblasScnrm2                    gpublasScnrm2
#define hipblasScopy                     gpublasScopy
#define hipblasSdgmm                     gpublasSdgmm
#define hipblasSdot                      gpublasSdot
#define hipblasSetAtomicsMode            gpublasSetAtomicsMode
#define hipblasSetMatrix                 gpublasSetMatrix
#define hipblasSetMatrixAsync            gpublasSetMatrixAsync
#define hipblasSetPointerMode            gpublasSetPointerMode
#define hipblasSetStream                 gpublasSetStream
#define hipblasSetVector                 gpublasSetVector
#define hipblasSetVectorAsync            gpublasSetVectorAsync
#define hipblasSgbmv                     gpublasSgbmv
#define hipblasSgeam                     gpublasSgeam
#define hipblasSgelsBatched              gpublasSgelsBatched
#define hipblasSgemm                     gpublasSgemm
#define hipblasSgemmBatched              gpublasSgemmBatched
#define hipblasSgemmStridedBatched       gpublasSgemmStridedBatched
#define hipblasSgemv                     gpublasSgemv
#define hipblasSgeqrfBatched             gpublasSgeqrfBatched
#define hipblasSger                      gpublasSger
#define hipblasSgetrfBatched             gpublasSgetrfBatched
#define hipblasSgetriBatched             gpublasSgetriBatched
#define hipblasSgetrsBatched             gpublasSgetrsBatched
#define hipblasSideMode_t                gpublasSideMode_t
#define hipblasSnrm2                     gpublasSnrm2
#define hipblasSrot                      gpublasSrot
#define hipblasSrotg                     gpublasSrotg
#define hipblasSrotm                     gpublasSrotm
#define hipblasSrotmg                    gpublasSrotmg
#define hipblasSsbmv                     gpublasSsbmv
#define hipblasSscal                     gpublasSscal
#define hipblasSspmv                     gpublasSspmv
#define hipblasSspr                      gpublasSspr
#define hipblasSspr2                     gpublasSspr2
#define hipblasSswap                     gpublasSswap
#define hipblasSsymm                     gpublasSsymm
#define hipblasSsymv                     gpublasSsymv
#define hipblasSsyr                      gpublasSsyr
#define hipblasSsyr2                     gpublasSsyr2
#define hipblasSsyr2k                    gpublasSsyr2k
#define hipblasSsyrk                     gpublasSsyrk
#define hipblasSsyrkx                    gpublasSsyrkx
#define hipblasStatus_t                  gpublasStatus_t
#define hipblasStbmv                     gpublasStbmv
#define hipblasStbsv                     gpublasStbsv
#define hipblasStpmv                     gpublasStpmv
#define hipblasStpsv                     gpublasStpsv
#define hipblasStrmm                     gpublasStrmm
#define hipblasStrmv                     gpublasStrmv
#define hipblasStrsm                     gpublasStrsm
#define hipblasStrsmBatched              gpublasStrsmBatched
#define hipblasStrsv                     gpublasStrsv
#define hipblasZaxpy                     gpublasZaxpy
#define hipblasZcopy                     gpublasZcopy
#define hipblasZdgmm                     gpublasZdgmm
#define hipblasZdotc                     gpublasZdotc
#define hipblasZdotu                     gpublasZdotu
#define hipblasZdrot                     gpublasZdrot
#define hipblasZdscal                    gpublasZdscal
#define hipblasZgbmv                     gpublasZgbmv
#define hipblasZgeam                     gpublasZgeam
#define hipblasZgelsBatched              gpublasZgelsBatched
#define hipblasZgemm                     gpublasZgemm
#define hipblasZgemmBatched              gpublasZgemmBatched
#define hipblasZgemmStridedBatched       gpublasZgemmStridedBatched
#define hipblasZgemv                     gpublasZgemv
#define hipblasZgeqrfBatched             gpublasZgeqrfBatched
#define hipblasZgerc                     gpublasZgerc
#define hipblasZgeru                     gpublasZgeru
#define hipblasZgetrfBatched             gpublasZgetrfBatched
#define hipblasZgetriBatched             gpublasZgetriBatched
#define hipblasZgetrsBatched             gpublasZgetrsBatched
#define hipblasZhbmv                     gpublasZhbmv
#define hipblasZhemm                     gpublasZhemm
#define hipblasZhemv                     gpublasZhemv
#define hipblasZher                      gpublasZher
#define hipblasZher2                     gpublasZher2
#define hipblasZher2k                    gpublasZher2k
#define hipblasZherk                     gpublasZherk
#define hipblasZherkx                    gpublasZherkx
#define hipblasZhpmv                     gpublasZhpmv
#define hipblasZhpr                      gpublasZhpr
#define hipblasZhpr2                     gpublasZhpr2
#define hipblasZrot                      gpublasZrot
#define hipblasZrotg                     gpublasZrotg
#define hipblasZscal                     gpublasZscal
#define hipblasZswap                     gpublasZswap
#define hipblasZsymm                     gpublasZsymm
#define hipblasZsymv                     gpublasZsymv
#define hipblasZsyr                      gpublasZsyr
#define hipblasZsyr2                     gpublasZsyr2
#define hipblasZsyr2k                    gpublasZsyr2k
#define hipblasZsyrk                     gpublasZsyrk
#define hipblasZsyrkx                    gpublasZsyrkx
#define hipblasZtbmv                     gpublasZtbmv
#define hipblasZtbsv                     gpublasZtbsv
#define hipblasZtpmv                     gpublasZtpmv
#define hipblasZtpsv                     gpublasZtpsv
#define hipblasZtrmm                     gpublasZtrmm
#define hipblasZtrmv                     gpublasZtrmv
#define hipblasZtrsm                     gpublasZtrsm
#define hipblasZtrsmBatched              gpublasZtrsmBatched
#define hipblasZtrsv                     gpublasZtrsv

#include <hop/hopblas.h>

#endif
