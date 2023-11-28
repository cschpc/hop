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

#ifndef __HOP_HOPBLAS_HIP_H__
#define __HOP_HOPBLAS_HIP_H__

#include <hipblas/hipblas.h>

#define GPUBLAS_ATOMICS_ALLOWED          HIPBLAS_ATOMICS_ALLOWED
#define GPUBLAS_ATOMICS_NOT_ALLOWED      HIPBLAS_ATOMICS_NOT_ALLOWED
#define GPUBLAS_C_16B                    HIPBLAS_C_16B
#define GPUBLAS_C_16F                    HIPBLAS_C_16F
#define GPUBLAS_C_32F                    HIPBLAS_C_32F
#define GPUBLAS_C_32I                    HIPBLAS_C_32I
#define GPUBLAS_C_32U                    HIPBLAS_C_32U
#define GPUBLAS_C_64F                    HIPBLAS_C_64F
#define GPUBLAS_C_8I                     HIPBLAS_C_8I
#define GPUBLAS_C_8U                     HIPBLAS_C_8U
#define GPUBLAS_DIAG_NON_UNIT            HIPBLAS_DIAG_NON_UNIT
#define GPUBLAS_DIAG_UNIT                HIPBLAS_DIAG_UNIT
#define GPUBLAS_FILL_MODE_FULL           HIPBLAS_FILL_MODE_FULL
#define GPUBLAS_FILL_MODE_LOWER          HIPBLAS_FILL_MODE_LOWER
#define GPUBLAS_FILL_MODE_UPPER          HIPBLAS_FILL_MODE_UPPER
#define GPUBLAS_GEMM_DEFAULT             HIPBLAS_GEMM_DEFAULT
#define GPUBLAS_OP_C                     HIPBLAS_OP_C
#define GPUBLAS_OP_N                     HIPBLAS_OP_N
#define GPUBLAS_OP_T                     HIPBLAS_OP_T
#define GPUBLAS_POINTER_MODE_DEVICE      HIPBLAS_POINTER_MODE_DEVICE
#define GPUBLAS_POINTER_MODE_HOST        HIPBLAS_POINTER_MODE_HOST
#define GPUBLAS_R_16B                    HIPBLAS_R_16B
#define GPUBLAS_R_16F                    HIPBLAS_R_16F
#define GPUBLAS_R_32F                    HIPBLAS_R_32F
#define GPUBLAS_R_32I                    HIPBLAS_R_32I
#define GPUBLAS_R_32U                    HIPBLAS_R_32U
#define GPUBLAS_R_64F                    HIPBLAS_R_64F
#define GPUBLAS_R_8I                     HIPBLAS_R_8I
#define GPUBLAS_R_8U                     HIPBLAS_R_8U
#define GPUBLAS_SIDE_LEFT                HIPBLAS_SIDE_LEFT
#define GPUBLAS_SIDE_RIGHT               HIPBLAS_SIDE_RIGHT
#define GPUBLAS_STATUS_ALLOC_FAILED      HIPBLAS_STATUS_ALLOC_FAILED
#define GPUBLAS_STATUS_ARCH_MISMATCH     HIPBLAS_STATUS_ARCH_MISMATCH
#define GPUBLAS_STATUS_EXECUTION_FAILED  HIPBLAS_STATUS_EXECUTION_FAILED
#define GPUBLAS_STATUS_INTERNAL_ERROR    HIPBLAS_STATUS_INTERNAL_ERROR
#define GPUBLAS_STATUS_INVALID_VALUE     HIPBLAS_STATUS_INVALID_VALUE
#define GPUBLAS_STATUS_MAPPING_ERROR     HIPBLAS_STATUS_MAPPING_ERROR
#define GPUBLAS_STATUS_NOT_INITIALIZED   HIPBLAS_STATUS_NOT_INITIALIZED
#define GPUBLAS_STATUS_NOT_SUPPORTED     HIPBLAS_STATUS_NOT_SUPPORTED
#define GPUBLAS_STATUS_SUCCESS           HIPBLAS_STATUS_SUCCESS
#define GPUBLAS_STATUS_UNKNOWN           HIPBLAS_STATUS_UNKNOWN
#define gpublasAtomicsMode_t             hipblasAtomicsMode_t
#define gpublasAxpyEx                    hipblasAxpyEx
#define gpublasCaxpy                     hipblasCaxpy
#define gpublasCcopy                     hipblasCcopy
#define gpublasCdgmm                     hipblasCdgmm
#define gpublasCdotc                     hipblasCdotc
#define gpublasCdotu                     hipblasCdotu
#define gpublasCgbmv                     hipblasCgbmv
#define gpublasCgeam                     hipblasCgeam
#define gpublasCgelsBatched              hipblasCgelsBatched
#define gpublasCgemm                     hipblasCgemm
#define gpublasCgemmBatched              hipblasCgemmBatched
#define gpublasCgemmStridedBatched       hipblasCgemmStridedBatched
#define gpublasCgemv                     hipblasCgemv
#define gpublasCgeqrfBatched             hipblasCgeqrfBatched
#define gpublasCgerc                     hipblasCgerc
#define gpublasCgeru                     hipblasCgeru
#define gpublasCgetrfBatched             hipblasCgetrfBatched
#define gpublasCgetriBatched             hipblasCgetriBatched
#define gpublasCgetrsBatched             hipblasCgetrsBatched
#define gpublasChbmv                     hipblasChbmv
#define gpublasChemm                     hipblasChemm
#define gpublasChemv                     hipblasChemv
#define gpublasCher                      hipblasCher
#define gpublasCher2                     hipblasCher2
#define gpublasCher2k                    hipblasCher2k
#define gpublasCherk                     hipblasCherk
#define gpublasCherkx                    hipblasCherkx
#define gpublasChpmv                     hipblasChpmv
#define gpublasChpr                      hipblasChpr
#define gpublasChpr2                     hipblasChpr2
#define gpublasCreate                    hipblasCreate
#define gpublasCrot                      hipblasCrot
#define gpublasCrotg                     hipblasCrotg
#define gpublasCscal                     hipblasCscal
#define gpublasCsrot                     hipblasCsrot
#define gpublasCsscal                    hipblasCsscal
#define gpublasCswap                     hipblasCswap
#define gpublasCsymm                     hipblasCsymm
#define gpublasCsymv                     hipblasCsymv
#define gpublasCsyr                      hipblasCsyr
#define gpublasCsyr2                     hipblasCsyr2
#define gpublasCsyr2k                    hipblasCsyr2k
#define gpublasCsyrk                     hipblasCsyrk
#define gpublasCsyrkx                    hipblasCsyrkx
#define gpublasCtbmv                     hipblasCtbmv
#define gpublasCtbsv                     hipblasCtbsv
#define gpublasCtpmv                     hipblasCtpmv
#define gpublasCtpsv                     hipblasCtpsv
#define gpublasCtrmm                     hipblasCtrmm
#define gpublasCtrmv                     hipblasCtrmv
#define gpublasCtrsm                     hipblasCtrsm
#define gpublasCtrsmBatched              hipblasCtrsmBatched
#define gpublasCtrsv                     hipblasCtrsv
#define gpublasDasum                     hipblasDasum
#define gpublasDatatype_t                hipblasDatatype_t
#define gpublasDaxpy                     hipblasDaxpy
#define gpublasDcopy                     hipblasDcopy
#define gpublasDdgmm                     hipblasDdgmm
#define gpublasDdot                      hipblasDdot
#define gpublasDestroy                   hipblasDestroy
#define gpublasDgbmv                     hipblasDgbmv
#define gpublasDgeam                     hipblasDgeam
#define gpublasDgelsBatched              hipblasDgelsBatched
#define gpublasDgemm                     hipblasDgemm
#define gpublasDgemmBatched              hipblasDgemmBatched
#define gpublasDgemmStridedBatched       hipblasDgemmStridedBatched
#define gpublasDgemv                     hipblasDgemv
#define gpublasDgeqrfBatched             hipblasDgeqrfBatched
#define gpublasDger                      hipblasDger
#define gpublasDgetrfBatched             hipblasDgetrfBatched
#define gpublasDgetriBatched             hipblasDgetriBatched
#define gpublasDgetrsBatched             hipblasDgetrsBatched
#define gpublasDiagType_t                hipblasDiagType_t
#define gpublasDnrm2                     hipblasDnrm2
#define gpublasDotEx                     hipblasDotEx
#define gpublasDotcEx                    hipblasDotcEx
#define gpublasDrot                      hipblasDrot
#define gpublasDrotg                     hipblasDrotg
#define gpublasDrotm                     hipblasDrotm
#define gpublasDrotmg                    hipblasDrotmg
#define gpublasDsbmv                     hipblasDsbmv
#define gpublasDscal                     hipblasDscal
#define gpublasDspmv                     hipblasDspmv
#define gpublasDspr                      hipblasDspr
#define gpublasDspr2                     hipblasDspr2
#define gpublasDswap                     hipblasDswap
#define gpublasDsymm                     hipblasDsymm
#define gpublasDsymv                     hipblasDsymv
#define gpublasDsyr                      hipblasDsyr
#define gpublasDsyr2                     hipblasDsyr2
#define gpublasDsyr2k                    hipblasDsyr2k
#define gpublasDsyrk                     hipblasDsyrk
#define gpublasDsyrkx                    hipblasDsyrkx
#define gpublasDtbmv                     hipblasDtbmv
#define gpublasDtbsv                     hipblasDtbsv
#define gpublasDtpmv                     hipblasDtpmv
#define gpublasDtpsv                     hipblasDtpsv
#define gpublasDtrmm                     hipblasDtrmm
#define gpublasDtrmv                     hipblasDtrmv
#define gpublasDtrsm                     hipblasDtrsm
#define gpublasDtrsmBatched              hipblasDtrsmBatched
#define gpublasDtrsv                     hipblasDtrsv
#define gpublasDzasum                    hipblasDzasum
#define gpublasDznrm2                    hipblasDznrm2
#define gpublasFillMode_t                hipblasFillMode_t
#define gpublasGemmAlgo_t                hipblasGemmAlgo_t
#define gpublasGemmBatchedEx             hipblasGemmBatchedEx
#define gpublasGemmEx                    hipblasGemmEx
#define gpublasGemmStridedBatchedEx      hipblasGemmStridedBatchedEx
#define gpublasGetAtomicsMode            hipblasGetAtomicsMode
#define gpublasGetMatrix                 hipblasGetMatrix
#define gpublasGetMatrixAsync            hipblasGetMatrixAsync
#define gpublasGetPointerMode            hipblasGetPointerMode
#define gpublasGetStream                 hipblasGetStream
#define gpublasGetVector                 hipblasGetVector
#define gpublasGetVectorAsync            hipblasGetVectorAsync
#define gpublasHandle_t                  hipblasHandle_t
#define gpublasHgemm                     hipblasHgemm
#define gpublasHgemmBatched              hipblasHgemmBatched
#define gpublasHgemmStridedBatched       hipblasHgemmStridedBatched
#define gpublasIcamax                    hipblasIcamax
#define gpublasIcamin                    hipblasIcamin
#define gpublasIdamax                    hipblasIdamax
#define gpublasIdamin                    hipblasIdamin
#define gpublasIsamax                    hipblasIsamax
#define gpublasIsamin                    hipblasIsamin
#define gpublasIzamax                    hipblasIzamax
#define gpublasIzamin                    hipblasIzamin
#define gpublasNrm2Ex                    hipblasNrm2Ex
#define gpublasOperation_t               hipblasOperation_t
#define gpublasPointerMode_t             hipblasPointerMode_t
#define gpublasRotEx                     hipblasRotEx
#define gpublasSasum                     hipblasSasum
#define gpublasSaxpy                     hipblasSaxpy
#define gpublasScalEx                    hipblasScalEx
#define gpublasScasum                    hipblasScasum
#define gpublasScnrm2                    hipblasScnrm2
#define gpublasScopy                     hipblasScopy
#define gpublasSdgmm                     hipblasSdgmm
#define gpublasSdot                      hipblasSdot
#define gpublasSetAtomicsMode            hipblasSetAtomicsMode
#define gpublasSetMatrix                 hipblasSetMatrix
#define gpublasSetMatrixAsync            hipblasSetMatrixAsync
#define gpublasSetPointerMode            hipblasSetPointerMode
#define gpublasSetStream                 hipblasSetStream
#define gpublasSetVector                 hipblasSetVector
#define gpublasSetVectorAsync            hipblasSetVectorAsync
#define gpublasSgbmv                     hipblasSgbmv
#define gpublasSgeam                     hipblasSgeam
#define gpublasSgelsBatched              hipblasSgelsBatched
#define gpublasSgemm                     hipblasSgemm
#define gpublasSgemmBatched              hipblasSgemmBatched
#define gpublasSgemmStridedBatched       hipblasSgemmStridedBatched
#define gpublasSgemv                     hipblasSgemv
#define gpublasSgeqrfBatched             hipblasSgeqrfBatched
#define gpublasSger                      hipblasSger
#define gpublasSgetrfBatched             hipblasSgetrfBatched
#define gpublasSgetriBatched             hipblasSgetriBatched
#define gpublasSgetrsBatched             hipblasSgetrsBatched
#define gpublasSideMode_t                hipblasSideMode_t
#define gpublasSnrm2                     hipblasSnrm2
#define gpublasSrot                      hipblasSrot
#define gpublasSrotg                     hipblasSrotg
#define gpublasSrotm                     hipblasSrotm
#define gpublasSrotmg                    hipblasSrotmg
#define gpublasSsbmv                     hipblasSsbmv
#define gpublasSscal                     hipblasSscal
#define gpublasSspmv                     hipblasSspmv
#define gpublasSspr                      hipblasSspr
#define gpublasSspr2                     hipblasSspr2
#define gpublasSswap                     hipblasSswap
#define gpublasSsymm                     hipblasSsymm
#define gpublasSsymv                     hipblasSsymv
#define gpublasSsyr                      hipblasSsyr
#define gpublasSsyr2                     hipblasSsyr2
#define gpublasSsyr2k                    hipblasSsyr2k
#define gpublasSsyrk                     hipblasSsyrk
#define gpublasSsyrkx                    hipblasSsyrkx
#define gpublasStatus_t                  hipblasStatus_t
#define gpublasStbmv                     hipblasStbmv
#define gpublasStbsv                     hipblasStbsv
#define gpublasStpmv                     hipblasStpmv
#define gpublasStpsv                     hipblasStpsv
#define gpublasStrmm                     hipblasStrmm
#define gpublasStrmv                     hipblasStrmv
#define gpublasStrsm                     hipblasStrsm
#define gpublasStrsmBatched              hipblasStrsmBatched
#define gpublasStrsv                     hipblasStrsv
#define gpublasZaxpy                     hipblasZaxpy
#define gpublasZcopy                     hipblasZcopy
#define gpublasZdgmm                     hipblasZdgmm
#define gpublasZdotc                     hipblasZdotc
#define gpublasZdotu                     hipblasZdotu
#define gpublasZdrot                     hipblasZdrot
#define gpublasZdscal                    hipblasZdscal
#define gpublasZgbmv                     hipblasZgbmv
#define gpublasZgeam                     hipblasZgeam
#define gpublasZgelsBatched              hipblasZgelsBatched
#define gpublasZgemm                     hipblasZgemm
#define gpublasZgemmBatched              hipblasZgemmBatched
#define gpublasZgemmStridedBatched       hipblasZgemmStridedBatched
#define gpublasZgemv                     hipblasZgemv
#define gpublasZgeqrfBatched             hipblasZgeqrfBatched
#define gpublasZgerc                     hipblasZgerc
#define gpublasZgeru                     hipblasZgeru
#define gpublasZgetrfBatched             hipblasZgetrfBatched
#define gpublasZgetriBatched             hipblasZgetriBatched
#define gpublasZgetrsBatched             hipblasZgetrsBatched
#define gpublasZhbmv                     hipblasZhbmv
#define gpublasZhemm                     hipblasZhemm
#define gpublasZhemv                     hipblasZhemv
#define gpublasZher                      hipblasZher
#define gpublasZher2                     hipblasZher2
#define gpublasZher2k                    hipblasZher2k
#define gpublasZherk                     hipblasZherk
#define gpublasZherkx                    hipblasZherkx
#define gpublasZhpmv                     hipblasZhpmv
#define gpublasZhpr                      hipblasZhpr
#define gpublasZhpr2                     hipblasZhpr2
#define gpublasZrot                      hipblasZrot
#define gpublasZrotg                     hipblasZrotg
#define gpublasZscal                     hipblasZscal
#define gpublasZswap                     hipblasZswap
#define gpublasZsymm                     hipblasZsymm
#define gpublasZsymv                     hipblasZsymv
#define gpublasZsyr                      hipblasZsyr
#define gpublasZsyr2                     hipblasZsyr2
#define gpublasZsyr2k                    hipblasZsyr2k
#define gpublasZsyrk                     hipblasZsyrk
#define gpublasZsyrkx                    hipblasZsyrkx
#define gpublasZtbmv                     hipblasZtbmv
#define gpublasZtbsv                     hipblasZtbsv
#define gpublasZtpmv                     hipblasZtpmv
#define gpublasZtpsv                     hipblasZtpsv
#define gpublasZtrmm                     hipblasZtrmm
#define gpublasZtrmv                     hipblasZtrmv
#define gpublasZtrsm                     hipblasZtrsm
#define gpublasZtrsmBatched              hipblasZtrsmBatched
#define gpublasZtrsv                     hipblasZtrsv


#endif
