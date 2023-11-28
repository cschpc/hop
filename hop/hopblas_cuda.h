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

#ifndef __HOP_HOPBLAS_CUDA_H__
#define __HOP_HOPBLAS_CUDA_H__

#if defined(HOP_OVERRIDE_CUBLAS_V1)
#include <cublas.h>
#else
#include <cublas_v2.h>
#endif

#define GPUBLAS_ATOMICS_ALLOWED          CUBLAS_ATOMICS_ALLOWED
#define GPUBLAS_ATOMICS_NOT_ALLOWED      CUBLAS_ATOMICS_NOT_ALLOWED
#define GPUBLAS_C_16B                    CUDA_C_16BF
#define GPUBLAS_C_16F                    CUDA_C_16F
#define GPUBLAS_C_32F                    CUDA_C_32F
#define GPUBLAS_C_32I                    CUDA_C_32I
#define GPUBLAS_C_32U                    CUDA_C_32U
#define GPUBLAS_C_64F                    CUDA_C_64F
#define GPUBLAS_C_8I                     CUDA_C_8I
#define GPUBLAS_C_8U                     CUDA_C_8U
#define GPUBLAS_DIAG_NON_UNIT            CUBLAS_DIAG_NON_UNIT
#define GPUBLAS_DIAG_UNIT                CUBLAS_DIAG_UNIT
#define GPUBLAS_FILL_MODE_FULL           CUBLAS_FILL_MODE_FULL
#define GPUBLAS_FILL_MODE_LOWER          CUBLAS_FILL_MODE_LOWER
#define GPUBLAS_FILL_MODE_UPPER          CUBLAS_FILL_MODE_UPPER
#define GPUBLAS_GEMM_DEFAULT             CUBLAS_GEMM_DFALT
#define GPUBLAS_OP_C                     CUBLAS_OP_HERMITAN
#define GPUBLAS_OP_N                     CUBLAS_OP_N
#define GPUBLAS_OP_T                     CUBLAS_OP_T
#define GPUBLAS_POINTER_MODE_DEVICE      CUBLAS_POINTER_MODE_DEVICE
#define GPUBLAS_POINTER_MODE_HOST        CUBLAS_POINTER_MODE_HOST
#define GPUBLAS_R_16B                    CUDA_R_16BF
#define GPUBLAS_R_16F                    CUDA_R_16F
#define GPUBLAS_R_32F                    CUDA_R_32F
#define GPUBLAS_R_32I                    CUDA_R_32I
#define GPUBLAS_R_32U                    CUDA_R_32U
#define GPUBLAS_R_64F                    CUDA_R_64F
#define GPUBLAS_R_8I                     CUDA_R_8I
#define GPUBLAS_R_8U                     CUDA_R_8U
#define GPUBLAS_SIDE_LEFT                CUBLAS_SIDE_LEFT
#define GPUBLAS_SIDE_RIGHT               CUBLAS_SIDE_RIGHT
#define GPUBLAS_STATUS_ALLOC_FAILED      CUBLAS_STATUS_ALLOC_FAILED
#define GPUBLAS_STATUS_ARCH_MISMATCH     CUBLAS_STATUS_ARCH_MISMATCH
#define GPUBLAS_STATUS_EXECUTION_FAILED  CUBLAS_STATUS_EXECUTION_FAILED
#define GPUBLAS_STATUS_INTERNAL_ERROR    CUBLAS_STATUS_INTERNAL_ERROR
#define GPUBLAS_STATUS_INVALID_VALUE     CUBLAS_STATUS_INVALID_VALUE
#define GPUBLAS_STATUS_MAPPING_ERROR     CUBLAS_STATUS_MAPPING_ERROR
#define GPUBLAS_STATUS_NOT_INITIALIZED   CUBLAS_STATUS_NOT_INITIALIZED
#define GPUBLAS_STATUS_NOT_SUPPORTED     CUBLAS_STATUS_NOT_SUPPORTED
#define GPUBLAS_STATUS_SUCCESS           CUBLAS_STATUS_SUCCESS
#define GPUBLAS_STATUS_UNKNOWN           CUBLAS_STATUS_LICENSE_ERROR
#define gpublasAtomicsMode_t             cublasAtomicsMode_t
#define gpublasAxpyEx                    cublasAxpyEx
#define gpublasCaxpy                     cublasCaxpy_v2
#define gpublasCcopy                     cublasCcopy_v2
#define gpublasCdgmm                     cublasCdgmm
#define gpublasCdotc                     cublasCdotc_v2
#define gpublasCdotu                     cublasCdotu_v2
#define gpublasCgbmv                     cublasCgbmv_v2
#define gpublasCgeam                     cublasCgeam
#define gpublasCgelsBatched              cublasCgelsBatched
#define gpublasCgemm                     cublasCgemm_v2
#define gpublasCgemmBatched              cublasCgemmBatched
#define gpublasCgemmStridedBatched       cublasCgemmStridedBatched
#define gpublasCgemv                     cublasCgemv_v2
#define gpublasCgeqrfBatched             cublasCgeqrfBatched
#define gpublasCgerc                     cublasCgerc_v2
#define gpublasCgeru                     cublasCgeru_v2
#define gpublasCgetrfBatched             cublasCgetrfBatched
#define gpublasCgetriBatched             cublasCgetriBatched
#define gpublasCgetrsBatched             cublasCgetrsBatched
#define gpublasChbmv                     cublasChbmv_v2
#define gpublasChemm                     cublasChemm_v2
#define gpublasChemv                     cublasChemv_v2
#define gpublasCher                      cublasCher_v2
#define gpublasCher2                     cublasCher2_v2
#define gpublasCher2k                    cublasCher2k_v2
#define gpublasCherk                     cublasCherk_v2
#define gpublasCherkx                    cublasCherkx
#define gpublasChpmv                     cublasChpmv_v2
#define gpublasChpr                      cublasChpr_v2
#define gpublasChpr2                     cublasChpr2_v2
#define gpublasCreate                    cublasCreate_v2
#define gpublasCrot                      cublasCrot_v2
#define gpublasCrotg                     cublasCrotg_v2
#define gpublasCscal                     cublasCscal_v2
#define gpublasCsrot                     cublasCsrot_v2
#define gpublasCsscal                    cublasCsscal_v2
#define gpublasCswap                     cublasCswap_v2
#define gpublasCsymm                     cublasCsymm_v2
#define gpublasCsymv                     cublasCsymv_v2
#define gpublasCsyr                      cublasCsyr_v2
#define gpublasCsyr2                     cublasCsyr2_v2
#define gpublasCsyr2k                    cublasCsyr2k_v2
#define gpublasCsyrk                     cublasCsyrk_v2
#define gpublasCsyrkx                    cublasCsyrkx
#define gpublasCtbmv                     cublasCtbmv_v2
#define gpublasCtbsv                     cublasCtbsv_v2
#define gpublasCtpmv                     cublasCtpmv_v2
#define gpublasCtpsv                     cublasCtpsv_v2
#define gpublasCtrmm                     cublasCtrmm_v2
#define gpublasCtrmv                     cublasCtrmv_v2
#define gpublasCtrsm                     cublasCtrsm_v2
#define gpublasCtrsmBatched              cublasCtrsmBatched
#define gpublasCtrsv                     cublasCtrsv_v2
#define gpublasDasum                     cublasDasum_v2
#define gpublasDatatype_t                cudaDataType
#define gpublasDaxpy                     cublasDaxpy_v2
#define gpublasDcopy                     cublasDcopy_v2
#define gpublasDdgmm                     cublasDdgmm
#define gpublasDdot                      cublasDdot_v2
#define gpublasDestroy                   cublasDestroy_v2
#define gpublasDgbmv                     cublasDgbmv_v2
#define gpublasDgeam                     cublasDgeam
#define gpublasDgelsBatched              cublasDgelsBatched
#define gpublasDgemm                     cublasDgemm_v2
#define gpublasDgemmBatched              cublasDgemmBatched
#define gpublasDgemmStridedBatched       cublasDgemmStridedBatched
#define gpublasDgemv                     cublasDgemv_v2
#define gpublasDgeqrfBatched             cublasDgeqrfBatched
#define gpublasDger                      cublasDger_v2
#define gpublasDgetrfBatched             cublasDgetrfBatched
#define gpublasDgetriBatched             cublasDgetriBatched
#define gpublasDgetrsBatched             cublasDgetrsBatched
#define gpublasDiagType_t                cublasDiagType_t
#define gpublasDnrm2                     cublasDnrm2_v2
#define gpublasDotEx                     cublasDotEx
#define gpublasDotcEx                    cublasDotcEx
#define gpublasDrot                      cublasDrot_v2
#define gpublasDrotg                     cublasDrotg_v2
#define gpublasDrotm                     cublasDrotm_v2
#define gpublasDrotmg                    cublasDrotmg_v2
#define gpublasDsbmv                     cublasDsbmv_v2
#define gpublasDscal                     cublasDscal_v2
#define gpublasDspmv                     cublasDspmv_v2
#define gpublasDspr                      cublasDspr_v2
#define gpublasDspr2                     cublasDspr2_v2
#define gpublasDswap                     cublasDswap_v2
#define gpublasDsymm                     cublasDsymm_v2
#define gpublasDsymv                     cublasDsymv_v2
#define gpublasDsyr                      cublasDsyr_v2
#define gpublasDsyr2                     cublasDsyr2_v2
#define gpublasDsyr2k                    cublasDsyr2k_v2
#define gpublasDsyrk                     cublasDsyrk_v2
#define gpublasDsyrkx                    cublasDsyrkx
#define gpublasDtbmv                     cublasDtbmv_v2
#define gpublasDtbsv                     cublasDtbsv_v2
#define gpublasDtpmv                     cublasDtpmv_v2
#define gpublasDtpsv                     cublasDtpsv_v2
#define gpublasDtrmm                     cublasDtrmm_v2
#define gpublasDtrmv                     cublasDtrmv_v2
#define gpublasDtrsm                     cublasDtrsm_v2
#define gpublasDtrsmBatched              cublasDtrsmBatched
#define gpublasDtrsv                     cublasDtrsv_v2
#define gpublasDzasum                    cublasDzasum_v2
#define gpublasDznrm2                    cublasDznrm2_v2
#define gpublasFillMode_t                cublasFillMode_t
#define gpublasGemmAlgo_t                cublasGemmAlgo_t
#define gpublasGemmBatchedEx             cublasGemmBatchedEx
#define gpublasGemmEx                    cublasGemmEx
#define gpublasGemmStridedBatchedEx      cublasGemmStridedBatchedEx
#define gpublasGetAtomicsMode            cublasGetAtomicsMode
#define gpublasGetMatrix                 cublasGetMatrix
#define gpublasGetMatrixAsync            cublasGetMatrixAsync
#define gpublasGetPointerMode            cublasGetPointerMode_v2
#define gpublasGetStream                 cublasGetStream_v2
#define gpublasGetVector                 cublasGetVector
#define gpublasGetVectorAsync            cublasGetVectorAsync
#define gpublasHandle_t                  cublasHandle_t
#define gpublasHgemm                     cublasHgemm
#define gpublasHgemmBatched              cublasHgemmBatched
#define gpublasHgemmStridedBatched       cublasHgemmStridedBatched
#define gpublasIcamax                    cublasIcamax_v2
#define gpublasIcamin                    cublasIcamin_v2
#define gpublasIdamax                    cublasIdamax_v2
#define gpublasIdamin                    cublasIdamin_v2
#define gpublasIsamax                    cublasIsamax_v2
#define gpublasIsamin                    cublasIsamin_v2
#define gpublasIzamax                    cublasIzamax_v2
#define gpublasIzamin                    cublasIzamin_v2
#define gpublasNrm2Ex                    cublasNrm2Ex
#define gpublasOperation_t               cublasOperation_t
#define gpublasPointerMode_t             cublasPointerMode_t
#define gpublasRotEx                     cublasRotEx
#define gpublasSasum                     cublasSasum_v2
#define gpublasSaxpy                     cublasSaxpy_v2
#define gpublasScalEx                    cublasScalEx
#define gpublasScasum                    cublasScasum_v2
#define gpublasScnrm2                    cublasScnrm2_v2
#define gpublasScopy                     cublasScopy_v2
#define gpublasSdgmm                     cublasSdgmm
#define gpublasSdot                      cublasSdot_v2
#define gpublasSetAtomicsMode            cublasSetAtomicsMode
#define gpublasSetMatrix                 cublasSetMatrix
#define gpublasSetMatrixAsync            cublasSetMatrixAsync
#define gpublasSetPointerMode            cublasSetPointerMode_v2
#define gpublasSetStream                 cublasSetStream_v2
#define gpublasSetVector                 cublasSetVector
#define gpublasSetVectorAsync            cublasSetVectorAsync
#define gpublasSgbmv                     cublasSgbmv_v2
#define gpublasSgeam                     cublasSgeam
#define gpublasSgelsBatched              cublasSgelsBatched
#define gpublasSgemm                     cublasSgemm_v2
#define gpublasSgemmBatched              cublasSgemmBatched
#define gpublasSgemmStridedBatched       cublasSgemmStridedBatched
#define gpublasSgemv                     cublasSgemv_v2
#define gpublasSgeqrfBatched             cublasSgeqrfBatched
#define gpublasSger                      cublasSger_v2
#define gpublasSgetrfBatched             cublasSgetrfBatched
#define gpublasSgetriBatched             cublasSgetriBatched
#define gpublasSgetrsBatched             cublasSgetrsBatched
#define gpublasSideMode_t                cublasSideMode_t
#define gpublasSnrm2                     cublasSnrm2_v2
#define gpublasSrot                      cublasSrot_v2
#define gpublasSrotg                     cublasSrotg_v2
#define gpublasSrotm                     cublasSrotm_v2
#define gpublasSrotmg                    cublasSrotmg_v2
#define gpublasSsbmv                     cublasSsbmv_v2
#define gpublasSscal                     cublasSscal_v2
#define gpublasSspmv                     cublasSspmv_v2
#define gpublasSspr                      cublasSspr_v2
#define gpublasSspr2                     cublasSspr2_v2
#define gpublasSswap                     cublasSswap_v2
#define gpublasSsymm                     cublasSsymm_v2
#define gpublasSsymv                     cublasSsymv_v2
#define gpublasSsyr                      cublasSsyr_v2
#define gpublasSsyr2                     cublasSsyr2_v2
#define gpublasSsyr2k                    cublasSsyr2k_v2
#define gpublasSsyrk                     cublasSsyrk_v2
#define gpublasSsyrkx                    cublasSsyrkx
#define gpublasStatus_t                  cublasStatus_t
#define gpublasStbmv                     cublasStbmv_v2
#define gpublasStbsv                     cublasStbsv_v2
#define gpublasStpmv                     cublasStpmv_v2
#define gpublasStpsv                     cublasStpsv_v2
#define gpublasStrmm                     cublasStrmm_v2
#define gpublasStrmv                     cublasStrmv_v2
#define gpublasStrsm                     cublasStrsm_v2
#define gpublasStrsmBatched              cublasStrsmBatched
#define gpublasStrsv                     cublasStrsv_v2
#define gpublasZaxpy                     cublasZaxpy_v2
#define gpublasZcopy                     cublasZcopy_v2
#define gpublasZdgmm                     cublasZdgmm
#define gpublasZdotc                     cublasZdotc_v2
#define gpublasZdotu                     cublasZdotu_v2
#define gpublasZdrot                     cublasZdrot_v2
#define gpublasZdscal                    cublasZdscal_v2
#define gpublasZgbmv                     cublasZgbmv_v2
#define gpublasZgeam                     cublasZgeam
#define gpublasZgelsBatched              cublasZgelsBatched
#define gpublasZgemm                     cublasZgemm_v2
#define gpublasZgemmBatched              cublasZgemmBatched
#define gpublasZgemmStridedBatched       cublasZgemmStridedBatched
#define gpublasZgemv                     cublasZgemv_v2
#define gpublasZgeqrfBatched             cublasZgeqrfBatched
#define gpublasZgerc                     cublasZgerc_v2
#define gpublasZgeru                     cublasZgeru_v2
#define gpublasZgetrfBatched             cublasZgetrfBatched
#define gpublasZgetriBatched             cublasZgetriBatched
#define gpublasZgetrsBatched             cublasZgetrsBatched
#define gpublasZhbmv                     cublasZhbmv_v2
#define gpublasZhemm                     cublasZhemm_v2
#define gpublasZhemv                     cublasZhemv_v2
#define gpublasZher                      cublasZher_v2
#define gpublasZher2                     cublasZher2_v2
#define gpublasZher2k                    cublasZher2k_v2
#define gpublasZherk                     cublasZherk_v2
#define gpublasZherkx                    cublasZherkx
#define gpublasZhpmv                     cublasZhpmv_v2
#define gpublasZhpr                      cublasZhpr_v2
#define gpublasZhpr2                     cublasZhpr2_v2
#define gpublasZrot                      cublasZrot_v2
#define gpublasZrotg                     cublasZrotg_v2
#define gpublasZscal                     cublasZscal_v2
#define gpublasZswap                     cublasZswap_v2
#define gpublasZsymm                     cublasZsymm_v2
#define gpublasZsymv                     cublasZsymv_v2
#define gpublasZsyr                      cublasZsyr_v2
#define gpublasZsyr2                     cublasZsyr2_v2
#define gpublasZsyr2k                    cublasZsyr2k_v2
#define gpublasZsyrk                     cublasZsyrk_v2
#define gpublasZsyrkx                    cublasZsyrkx
#define gpublasZtbmv                     cublasZtbmv_v2
#define gpublasZtbsv                     cublasZtbsv_v2
#define gpublasZtpmv                     cublasZtpmv_v2
#define gpublasZtpsv                     cublasZtpsv_v2
#define gpublasZtrmm                     cublasZtrmm_v2
#define gpublasZtrmv                     cublasZtrmv_v2
#define gpublasZtrsm                     cublasZtrsm_v2
#define gpublasZtrsmBatched              cublasZtrsmBatched
#define gpublasZtrsv                     cublasZtrsv_v2


#endif
