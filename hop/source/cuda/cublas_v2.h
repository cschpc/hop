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

#ifndef __HOP_SOURCE_CUDA_CUBLAS_V2_H__
#define __HOP_SOURCE_CUDA_CUBLAS_V2_H__

#if !defined(HOP_SOURCE_CUDA)
#define HOP_SOURCE_CUDA
#endif

#include <cuComplex.h>
#include <driver_types.h>
#include <library_types.h>

#define cublasCaxpy                      gpublasCaxpy
#define cublasCcopy                      gpublasCcopy
#define cublasCdotc                      gpublasCdotc
#define cublasCdotu                      gpublasCdotu
#define cublasCgbmv                      gpublasCgbmv
#define cublasCgemm                      gpublasCgemm
#define cublasCgemv                      gpublasCgemv
#define cublasCgerc                      gpublasCgerc
#define cublasCgeru                      gpublasCgeru
#define cublasChbmv                      gpublasChbmv
#define cublasChemm                      gpublasChemm
#define cublasChemv                      gpublasChemv
#define cublasCher                       gpublasCher
#define cublasCher2                      gpublasCher2
#define cublasCher2k                     gpublasCher2k
#define cublasCherk                      gpublasCherk
#define cublasChpmv                      gpublasChpmv
#define cublasChpr                       gpublasChpr
#define cublasChpr2                      gpublasChpr2
#define cublasCreate                     gpublasCreate
#define cublasCrot                       gpublasCrot
#define cublasCrotg                      gpublasCrotg
#define cublasCscal                      gpublasCscal
#define cublasCsrot                      gpublasCsrot
#define cublasCsscal                     gpublasCsscal
#define cublasCswap                      gpublasCswap
#define cublasCsymm                      gpublasCsymm
#define cublasCsymv                      gpublasCsymv
#define cublasCsyr                       gpublasCsyr
#define cublasCsyr2                      gpublasCsyr2
#define cublasCsyr2k                     gpublasCsyr2k
#define cublasCsyrk                      gpublasCsyrk
#define cublasCtbmv                      gpublasCtbmv
#define cublasCtbsv                      gpublasCtbsv
#define cublasCtpmv                      gpublasCtpmv
#define cublasCtpsv                      gpublasCtpsv
#define cublasCtrmm                      gpublasCtrmm
#define cublasCtrmv                      gpublasCtrmv
#define cublasCtrsm                      gpublasCtrsm
#define cublasCtrsv                      gpublasCtrsv
#define cublasDasum                      gpublasDasum
#define cublasDaxpy                      gpublasDaxpy
#define cublasDcopy                      gpublasDcopy
#define cublasDdot                       gpublasDdot
#define cublasDestroy                    gpublasDestroy
#define cublasDgbmv                      gpublasDgbmv
#define cublasDgemm                      gpublasDgemm
#define cublasDgemv                      gpublasDgemv
#define cublasDger                       gpublasDger
#define cublasDnrm2                      gpublasDnrm2
#define cublasDrot                       gpublasDrot
#define cublasDrotg                      gpublasDrotg
#define cublasDrotm                      gpublasDrotm
#define cublasDrotmg                     gpublasDrotmg
#define cublasDsbmv                      gpublasDsbmv
#define cublasDscal                      gpublasDscal
#define cublasDspmv                      gpublasDspmv
#define cublasDspr                       gpublasDspr
#define cublasDspr2                      gpublasDspr2
#define cublasDswap                      gpublasDswap
#define cublasDsymm                      gpublasDsymm
#define cublasDsymv                      gpublasDsymv
#define cublasDsyr                       gpublasDsyr
#define cublasDsyr2                      gpublasDsyr2
#define cublasDsyr2k                     gpublasDsyr2k
#define cublasDsyrk                      gpublasDsyrk
#define cublasDtbmv                      gpublasDtbmv
#define cublasDtbsv                      gpublasDtbsv
#define cublasDtpmv                      gpublasDtpmv
#define cublasDtpsv                      gpublasDtpsv
#define cublasDtrmm                      gpublasDtrmm
#define cublasDtrmv                      gpublasDtrmv
#define cublasDtrsm                      gpublasDtrsm
#define cublasDtrsv                      gpublasDtrsv
#define cublasDzasum                     gpublasDzasum
#define cublasDznrm2                     gpublasDznrm2
#define cublasGetPointerMode             gpublasGetPointerMode
#define cublasGetStream                  gpublasGetStream
#define cublasIcamax                     gpublasIcamax
#define cublasIcamin                     gpublasIcamin
#define cublasIdamax                     gpublasIdamax
#define cublasIdamin                     gpublasIdamin
#define cublasIsamax                     gpublasIsamax
#define cublasIsamin                     gpublasIsamin
#define cublasIzamax                     gpublasIzamax
#define cublasIzamin                     gpublasIzamin
#define cublasSasum                      gpublasSasum
#define cublasSaxpy                      gpublasSaxpy
#define cublasScasum                     gpublasScasum
#define cublasScnrm2                     gpublasScnrm2
#define cublasScopy                      gpublasScopy
#define cublasSdot                       gpublasSdot
#define cublasSetPointerMode             gpublasSetPointerMode
#define cublasSetStream                  gpublasSetStream
#define cublasSgbmv                      gpublasSgbmv
#define cublasSgemm                      gpublasSgemm
#define cublasSgemv                      gpublasSgemv
#define cublasSger                       gpublasSger
#define cublasSnrm2                      gpublasSnrm2
#define cublasSrot                       gpublasSrot
#define cublasSrotg                      gpublasSrotg
#define cublasSrotm                      gpublasSrotm
#define cublasSrotmg                     gpublasSrotmg
#define cublasSsbmv                      gpublasSsbmv
#define cublasSscal                      gpublasSscal
#define cublasSspmv                      gpublasSspmv
#define cublasSspr                       gpublasSspr
#define cublasSspr2                      gpublasSspr2
#define cublasSswap                      gpublasSswap
#define cublasSsymm                      gpublasSsymm
#define cublasSsymv                      gpublasSsymv
#define cublasSsyr                       gpublasSsyr
#define cublasSsyr2                      gpublasSsyr2
#define cublasSsyr2k                     gpublasSsyr2k
#define cublasSsyrk                      gpublasSsyrk
#define cublasStbmv                      gpublasStbmv
#define cublasStbsv                      gpublasStbsv
#define cublasStpmv                      gpublasStpmv
#define cublasStpsv                      gpublasStpsv
#define cublasStrmm                      gpublasStrmm
#define cublasStrmv                      gpublasStrmv
#define cublasStrsm                      gpublasStrsm
#define cublasStrsv                      gpublasStrsv
#define cublasZaxpy                      gpublasZaxpy
#define cublasZcopy                      gpublasZcopy
#define cublasZdotc                      gpublasZdotc
#define cublasZdotu                      gpublasZdotu
#define cublasZdrot                      gpublasZdrot
#define cublasZdscal                     gpublasZdscal
#define cublasZgbmv                      gpublasZgbmv
#define cublasZgemm                      gpublasZgemm
#define cublasZgemv                      gpublasZgemv
#define cublasZgerc                      gpublasZgerc
#define cublasZgeru                      gpublasZgeru
#define cublasZhbmv                      gpublasZhbmv
#define cublasZhemm                      gpublasZhemm
#define cublasZhemv                      gpublasZhemv
#define cublasZher                       gpublasZher
#define cublasZher2                      gpublasZher2
#define cublasZher2k                     gpublasZher2k
#define cublasZherk                      gpublasZherk
#define cublasZhpmv                      gpublasZhpmv
#define cublasZhpr                       gpublasZhpr
#define cublasZhpr2                      gpublasZhpr2
#define cublasZrot                       gpublasZrot
#define cublasZrotg                      gpublasZrotg
#define cublasZscal                      gpublasZscal
#define cublasZswap                      gpublasZswap
#define cublasZsymm                      gpublasZsymm
#define cublasZsymv                      gpublasZsymv
#define cublasZsyr                       gpublasZsyr
#define cublasZsyr2                      gpublasZsyr2
#define cublasZsyr2k                     gpublasZsyr2k
#define cublasZsyrk                      gpublasZsyrk
#define cublasZtbmv                      gpublasZtbmv
#define cublasZtbsv                      gpublasZtbsv
#define cublasZtpmv                      gpublasZtpmv
#define cublasZtpsv                      gpublasZtpsv
#define cublasZtrmm                      gpublasZtrmm
#define cublasZtrmv                      gpublasZtrmv
#define cublasZtrsm                      gpublasZtrsm
#define cublasZtrsv                      gpublasZtrsv

/* cublas_api.h */
#define CUBLAS_ATOMICS_ALLOWED           GPUBLAS_ATOMICS_ALLOWED
#define CUBLAS_ATOMICS_NOT_ALLOWED       GPUBLAS_ATOMICS_NOT_ALLOWED
#define CUBLAS_DIAG_NON_UNIT             GPUBLAS_DIAG_NON_UNIT
#define CUBLAS_DIAG_UNIT                 GPUBLAS_DIAG_UNIT
#define CUBLAS_FILL_MODE_FULL            GPUBLAS_FILL_MODE_FULL
#define CUBLAS_FILL_MODE_LOWER           GPUBLAS_FILL_MODE_LOWER
#define CUBLAS_FILL_MODE_UPPER           GPUBLAS_FILL_MODE_UPPER
#define CUBLAS_GEMM_DEFAULT              GPUBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DFALT                GPUBLAS_GEMM_DEFAULT
#define CUBLAS_OP_C                      GPUBLAS_OP_C
#define CUBLAS_OP_HERMITAN               GPUBLAS_OP_C
#define CUBLAS_OP_N                      GPUBLAS_OP_N
#define CUBLAS_OP_T                      GPUBLAS_OP_T
#define CUBLAS_POINTER_MODE_DEVICE       GPUBLAS_POINTER_MODE_DEVICE
#define CUBLAS_POINTER_MODE_HOST         GPUBLAS_POINTER_MODE_HOST
#define CUBLAS_SIDE_LEFT                 GPUBLAS_SIDE_LEFT
#define CUBLAS_SIDE_RIGHT                GPUBLAS_SIDE_RIGHT
#define CUBLAS_STATUS_ALLOC_FAILED       GPUBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_ARCH_MISMATCH      GPUBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_EXECUTION_FAILED   GPUBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR     GPUBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_INVALID_VALUE      GPUBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_LICENSE_ERROR      GPUBLAS_STATUS_UNKNOWN
#define CUBLAS_STATUS_MAPPING_ERROR      GPUBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_NOT_INITIALIZED    GPUBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_NOT_SUPPORTED      GPUBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_SUCCESS            GPUBLAS_STATUS_SUCCESS
#define cublasAtomicsMode_t              gpublasAtomicsMode_t
#define cublasAxpyEx                     gpublasAxpyEx
#define cublasCaxpy_v2                   gpublasCaxpy
#define cublasCcopy_v2                   gpublasCcopy
#define cublasCdgmm                      gpublasCdgmm
#define cublasCdotc_v2                   gpublasCdotc
#define cublasCdotu_v2                   gpublasCdotu
#define cublasCgbmv_v2                   gpublasCgbmv
#define cublasCgeam                      gpublasCgeam
#define cublasCgelsBatched               gpublasCgelsBatched
#define cublasCgemmBatched               gpublasCgemmBatched
#define cublasCgemmStridedBatched        gpublasCgemmStridedBatched
#define cublasCgemm_v2                   gpublasCgemm
#define cublasCgemv_v2                   gpublasCgemv
#define cublasCgeqrfBatched              gpublasCgeqrfBatched
#define cublasCgerc_v2                   gpublasCgerc
#define cublasCgeru_v2                   gpublasCgeru
#define cublasCgetrfBatched              gpublasCgetrfBatched
#define cublasCgetriBatched              gpublasCgetriBatched
#define cublasCgetrsBatched              gpublasCgetrsBatched
#define cublasChbmv_v2                   gpublasChbmv
#define cublasChemm_v2                   gpublasChemm
#define cublasChemv_v2                   gpublasChemv
#define cublasCher2_v2                   gpublasCher2
#define cublasCher2k_v2                  gpublasCher2k
#define cublasCher_v2                    gpublasCher
#define cublasCherk_v2                   gpublasCherk
#define cublasCherkx                     gpublasCherkx
#define cublasChpmv_v2                   gpublasChpmv
#define cublasChpr2_v2                   gpublasChpr2
#define cublasChpr_v2                    gpublasChpr
#define cublasComputeType_t              gpublasDatatype_t
#define cublasCreate_v2                  gpublasCreate
#define cublasCrot_v2                    gpublasCrot
#define cublasCrotg_v2                   gpublasCrotg
#define cublasCscal_v2                   gpublasCscal
#define cublasCsrot_v2                   gpublasCsrot
#define cublasCsscal_v2                  gpublasCsscal
#define cublasCswap_v2                   gpublasCswap
#define cublasCsymm_v2                   gpublasCsymm
#define cublasCsymv_v2                   gpublasCsymv
#define cublasCsyr2_v2                   gpublasCsyr2
#define cublasCsyr2k_v2                  gpublasCsyr2k
#define cublasCsyr_v2                    gpublasCsyr
#define cublasCsyrk_v2                   gpublasCsyrk
#define cublasCsyrkx                     gpublasCsyrkx
#define cublasCtbmv_v2                   gpublasCtbmv
#define cublasCtbsv_v2                   gpublasCtbsv
#define cublasCtpmv_v2                   gpublasCtpmv
#define cublasCtpsv_v2                   gpublasCtpsv
#define cublasCtrmm_v2                   gpublasCtrmm
#define cublasCtrmv_v2                   gpublasCtrmv
#define cublasCtrsmBatched               gpublasCtrsmBatched
#define cublasCtrsm_v2                   gpublasCtrsm
#define cublasCtrsv_v2                   gpublasCtrsv
#define cublasDasum_v2                   gpublasDasum
#define cublasDataType_t                 gpublasDatatype_t
#define cublasDaxpy_v2                   gpublasDaxpy
#define cublasDcopy_v2                   gpublasDcopy
#define cublasDdgmm                      gpublasDdgmm
#define cublasDdot_v2                    gpublasDdot
#define cublasDestroy_v2                 gpublasDestroy
#define cublasDgbmv_v2                   gpublasDgbmv
#define cublasDgeam                      gpublasDgeam
#define cublasDgelsBatched               gpublasDgelsBatched
#define cublasDgemmBatched               gpublasDgemmBatched
#define cublasDgemmStridedBatched        gpublasDgemmStridedBatched
#define cublasDgemm_v2                   gpublasDgemm
#define cublasDgemv_v2                   gpublasDgemv
#define cublasDgeqrfBatched              gpublasDgeqrfBatched
#define cublasDger_v2                    gpublasDger
#define cublasDgetrfBatched              gpublasDgetrfBatched
#define cublasDgetriBatched              gpublasDgetriBatched
#define cublasDgetrsBatched              gpublasDgetrsBatched
#define cublasDiagType_t                 gpublasDiagType_t
#define cublasDnrm2_v2                   gpublasDnrm2
#define cublasDotEx                      gpublasDotEx
#define cublasDotcEx                     gpublasDotcEx
#define cublasDrot_v2                    gpublasDrot
#define cublasDrotg_v2                   gpublasDrotg
#define cublasDrotm_v2                   gpublasDrotm
#define cublasDrotmg_v2                  gpublasDrotmg
#define cublasDsbmv_v2                   gpublasDsbmv
#define cublasDscal_v2                   gpublasDscal
#define cublasDspmv_v2                   gpublasDspmv
#define cublasDspr2_v2                   gpublasDspr2
#define cublasDspr_v2                    gpublasDspr
#define cublasDswap_v2                   gpublasDswap
#define cublasDsymm_v2                   gpublasDsymm
#define cublasDsymv_v2                   gpublasDsymv
#define cublasDsyr2_v2                   gpublasDsyr2
#define cublasDsyr2k_v2                  gpublasDsyr2k
#define cublasDsyr_v2                    gpublasDsyr
#define cublasDsyrk_v2                   gpublasDsyrk
#define cublasDsyrkx                     gpublasDsyrkx
#define cublasDtbmv_v2                   gpublasDtbmv
#define cublasDtbsv_v2                   gpublasDtbsv
#define cublasDtpmv_v2                   gpublasDtpmv
#define cublasDtpsv_v2                   gpublasDtpsv
#define cublasDtrmm_v2                   gpublasDtrmm
#define cublasDtrmv_v2                   gpublasDtrmv
#define cublasDtrsmBatched               gpublasDtrsmBatched
#define cublasDtrsm_v2                   gpublasDtrsm
#define cublasDtrsv_v2                   gpublasDtrsv
#define cublasDzasum_v2                  gpublasDzasum
#define cublasDznrm2_v2                  gpublasDznrm2
#define cublasFillMode_t                 gpublasFillMode_t
#define cublasGemmAlgo_t                 gpublasGemmAlgo_t
#define cublasGemmBatchedEx              gpublasGemmBatchedEx
#define cublasGemmEx                     gpublasGemmEx
#define cublasGemmStridedBatchedEx       gpublasGemmStridedBatchedEx
#define cublasGetAtomicsMode             gpublasGetAtomicsMode
#define cublasGetMatrix                  gpublasGetMatrix
#define cublasGetMatrixAsync             gpublasGetMatrixAsync
#define cublasGetPointerMode_v2          gpublasGetPointerMode
#define cublasGetStream_v2               gpublasGetStream
#define cublasGetVector                  gpublasGetVector
#define cublasGetVectorAsync             gpublasGetVectorAsync
#define cublasHandle_t                   gpublasHandle_t
#define cublasHgemm                      gpublasHgemm
#define cublasHgemmBatched               gpublasHgemmBatched
#define cublasHgemmStridedBatched        gpublasHgemmStridedBatched
#define cublasIcamax_v2                  gpublasIcamax
#define cublasIcamin_v2                  gpublasIcamin
#define cublasIdamax_v2                  gpublasIdamax
#define cublasIdamin_v2                  gpublasIdamin
#define cublasIsamax_v2                  gpublasIsamax
#define cublasIsamin_v2                  gpublasIsamin
#define cublasIzamax_v2                  gpublasIzamax
#define cublasIzamin_v2                  gpublasIzamin
#define cublasNrm2Ex                     gpublasNrm2Ex
#define cublasOperation_t                gpublasOperation_t
#define cublasPointerMode_t              gpublasPointerMode_t
#define cublasRotEx                      gpublasRotEx
#define cublasSasum_v2                   gpublasSasum
#define cublasSaxpy_v2                   gpublasSaxpy
#define cublasScalEx                     gpublasScalEx
#define cublasScasum_v2                  gpublasScasum
#define cublasScnrm2_v2                  gpublasScnrm2
#define cublasScopy_v2                   gpublasScopy
#define cublasSdgmm                      gpublasSdgmm
#define cublasSdot_v2                    gpublasSdot
#define cublasSetAtomicsMode             gpublasSetAtomicsMode
#define cublasSetMatrix                  gpublasSetMatrix
#define cublasSetMatrixAsync             gpublasSetMatrixAsync
#define cublasSetPointerMode_v2          gpublasSetPointerMode
#define cublasSetStream_v2               gpublasSetStream
#define cublasSetVector                  gpublasSetVector
#define cublasSetVectorAsync             gpublasSetVectorAsync
#define cublasSgbmv_v2                   gpublasSgbmv
#define cublasSgeam                      gpublasSgeam
#define cublasSgelsBatched               gpublasSgelsBatched
#define cublasSgemmBatched               gpublasSgemmBatched
#define cublasSgemmStridedBatched        gpublasSgemmStridedBatched
#define cublasSgemm_v2                   gpublasSgemm
#define cublasSgemv_v2                   gpublasSgemv
#define cublasSgeqrfBatched              gpublasSgeqrfBatched
#define cublasSger_v2                    gpublasSger
#define cublasSgetrfBatched              gpublasSgetrfBatched
#define cublasSgetriBatched              gpublasSgetriBatched
#define cublasSgetrsBatched              gpublasSgetrsBatched
#define cublasSideMode_t                 gpublasSideMode_t
#define cublasSnrm2_v2                   gpublasSnrm2
#define cublasSrot_v2                    gpublasSrot
#define cublasSrotg_v2                   gpublasSrotg
#define cublasSrotm_v2                   gpublasSrotm
#define cublasSrotmg_v2                  gpublasSrotmg
#define cublasSsbmv_v2                   gpublasSsbmv
#define cublasSscal_v2                   gpublasSscal
#define cublasSspmv_v2                   gpublasSspmv
#define cublasSspr2_v2                   gpublasSspr2
#define cublasSspr_v2                    gpublasSspr
#define cublasSswap_v2                   gpublasSswap
#define cublasSsymm_v2                   gpublasSsymm
#define cublasSsymv_v2                   gpublasSsymv
#define cublasSsyr2_v2                   gpublasSsyr2
#define cublasSsyr2k_v2                  gpublasSsyr2k
#define cublasSsyr_v2                    gpublasSsyr
#define cublasSsyrk_v2                   gpublasSsyrk
#define cublasSsyrkx                     gpublasSsyrkx
#define cublasStatus_t                   gpublasStatus_t
#define cublasStbmv_v2                   gpublasStbmv
#define cublasStbsv_v2                   gpublasStbsv
#define cublasStpmv_v2                   gpublasStpmv
#define cublasStpsv_v2                   gpublasStpsv
#define cublasStrmm_v2                   gpublasStrmm
#define cublasStrmv_v2                   gpublasStrmv
#define cublasStrsmBatched               gpublasStrsmBatched
#define cublasStrsm_v2                   gpublasStrsm
#define cublasStrsv_v2                   gpublasStrsv
#define cublasZaxpy_v2                   gpublasZaxpy
#define cublasZcopy_v2                   gpublasZcopy
#define cublasZdgmm                      gpublasZdgmm
#define cublasZdotc_v2                   gpublasZdotc
#define cublasZdotu_v2                   gpublasZdotu
#define cublasZdrot_v2                   gpublasZdrot
#define cublasZdscal_v2                  gpublasZdscal
#define cublasZgbmv_v2                   gpublasZgbmv
#define cublasZgeam                      gpublasZgeam
#define cublasZgelsBatched               gpublasZgelsBatched
#define cublasZgemmBatched               gpublasZgemmBatched
#define cublasZgemmStridedBatched        gpublasZgemmStridedBatched
#define cublasZgemm_v2                   gpublasZgemm
#define cublasZgemv_v2                   gpublasZgemv
#define cublasZgeqrfBatched              gpublasZgeqrfBatched
#define cublasZgerc_v2                   gpublasZgerc
#define cublasZgeru_v2                   gpublasZgeru
#define cublasZgetrfBatched              gpublasZgetrfBatched
#define cublasZgetriBatched              gpublasZgetriBatched
#define cublasZgetrsBatched              gpublasZgetrsBatched
#define cublasZhbmv_v2                   gpublasZhbmv
#define cublasZhemm_v2                   gpublasZhemm
#define cublasZhemv_v2                   gpublasZhemv
#define cublasZher2_v2                   gpublasZher2
#define cublasZher2k_v2                  gpublasZher2k
#define cublasZher_v2                    gpublasZher
#define cublasZherk_v2                   gpublasZherk
#define cublasZherkx                     gpublasZherkx
#define cublasZhpmv_v2                   gpublasZhpmv
#define cublasZhpr2_v2                   gpublasZhpr2
#define cublasZhpr_v2                    gpublasZhpr
#define cublasZrot_v2                    gpublasZrot
#define cublasZrotg_v2                   gpublasZrotg
#define cublasZscal_v2                   gpublasZscal
#define cublasZswap_v2                   gpublasZswap
#define cublasZsymm_v2                   gpublasZsymm
#define cublasZsymv_v2                   gpublasZsymv
#define cublasZsyr2_v2                   gpublasZsyr2
#define cublasZsyr2k_v2                  gpublasZsyr2k
#define cublasZsyr_v2                    gpublasZsyr
#define cublasZsyrk_v2                   gpublasZsyrk
#define cublasZsyrkx                     gpublasZsyrkx
#define cublasZtbmv_v2                   gpublasZtbmv
#define cublasZtbsv_v2                   gpublasZtbsv
#define cublasZtpmv_v2                   gpublasZtpmv
#define cublasZtpsv_v2                   gpublasZtpsv
#define cublasZtrmm_v2                   gpublasZtrmm
#define cublasZtrmv_v2                   gpublasZtrmv
#define cublasZtrsmBatched               gpublasZtrsmBatched
#define cublasZtrsm_v2                   gpublasZtrsm
#define cublasZtrsv_v2                   gpublasZtrsv

/* cublas.h */
#define cublasStatus                     gpublasStatus_t

#include <hop/hopblas.h>

#endif
