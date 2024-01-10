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

#ifndef __HOP_SOURCE_HIP_HIPSPARSE_H__
#define __HOP_SOURCE_HIP_HIPSPARSE_H__

#if !defined(HOP_SOURCE_HIP)
#define HOP_SOURCE_HIP
#endif

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

#define HIPSPARSE_ACTION_NUMERIC         GPUSPARSE_ACTION_NUMERIC
#define HIPSPARSE_ACTION_SYMBOLIC        GPUSPARSE_ACTION_SYMBOLIC
#define HIPSPARSE_COOMM_ALG1             GPUSPARSE_COOMM_ALG1
#define HIPSPARSE_COOMM_ALG2             GPUSPARSE_COOMM_ALG2
#define HIPSPARSE_COOMM_ALG3             GPUSPARSE_COOMM_ALG3
#define HIPSPARSE_COOMV_ALG              GPUSPARSE_COOMV_ALG
#define HIPSPARSE_CSR2CSC_ALG1           GPUSPARSE_CSR2CSC_ALG1
#define HIPSPARSE_CSR2CSC_ALG2           GPUSPARSE_CSR2CSC_ALG2
#define HIPSPARSE_CSR2CSC_ALG_DEFAULT    GPUSPARSE_CSR2CSC_ALG_DEFAULT
#define HIPSPARSE_CSRMM_ALG1             GPUSPARSE_CSRMM_ALG1
#define HIPSPARSE_CSRMV_ALG1             GPUSPARSE_CSRMV_ALG1
#define HIPSPARSE_CSRMV_ALG2             GPUSPARSE_CSRMV_ALG2
#define HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT  \
        GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT
#define HIPSPARSE_DIAG_TYPE_NON_UNIT     GPUSPARSE_DIAG_TYPE_NON_UNIT
#define HIPSPARSE_DIAG_TYPE_UNIT         GPUSPARSE_DIAG_TYPE_UNIT
#define HIPSPARSE_DIRECTION_COLUMN       GPUSPARSE_DIRECTION_COLUMN
#define HIPSPARSE_DIRECTION_ROW          GPUSPARSE_DIRECTION_ROW
#define HIPSPARSE_FILL_MODE_LOWER        GPUSPARSE_FILL_MODE_LOWER
#define HIPSPARSE_FILL_MODE_UPPER        GPUSPARSE_FILL_MODE_UPPER
#define HIPSPARSE_FORMAT_BLOCKED_ELL     GPUSPARSE_FORMAT_BLOCKED_ELL
#define HIPSPARSE_FORMAT_COO             GPUSPARSE_FORMAT_COO
#define HIPSPARSE_FORMAT_COO_AOS         GPUSPARSE_FORMAT_COO_AOS
#define HIPSPARSE_FORMAT_CSC             GPUSPARSE_FORMAT_CSC
#define HIPSPARSE_FORMAT_CSR             GPUSPARSE_FORMAT_CSR
#define HIPSPARSE_INDEX_16U              GPUSPARSE_INDEX_16U
#define HIPSPARSE_INDEX_32I              GPUSPARSE_INDEX_32I
#define HIPSPARSE_INDEX_64I              GPUSPARSE_INDEX_64I
#define HIPSPARSE_INDEX_BASE_ONE         GPUSPARSE_INDEX_BASE_ONE
#define HIPSPARSE_INDEX_BASE_ZERO        GPUSPARSE_INDEX_BASE_ZERO
#define HIPSPARSE_MATRIX_TYPE_GENERAL    GPUSPARSE_MATRIX_TYPE_GENERAL
#define HIPSPARSE_MATRIX_TYPE_HERMITIAN  GPUSPARSE_MATRIX_TYPE_HERMITIAN
#define HIPSPARSE_MATRIX_TYPE_SYMMETRIC  GPUSPARSE_MATRIX_TYPE_SYMMETRIC
#define HIPSPARSE_MATRIX_TYPE_TRIANGULAR GPUSPARSE_MATRIX_TYPE_TRIANGULAR
#define HIPSPARSE_MM_ALG_DEFAULT         GPUSPARSE_MM_ALG_DEFAULT
#define HIPSPARSE_MV_ALG_DEFAULT         GPUSPARSE_MV_ALG_DEFAULT
#define HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE  \
        GPUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
#define HIPSPARSE_OPERATION_NON_TRANSPOSE  \
        GPUSPARSE_OPERATION_NON_TRANSPOSE
#define HIPSPARSE_OPERATION_TRANSPOSE    GPUSPARSE_OPERATION_TRANSPOSE
#define HIPSPARSE_ORDER_COL              GPUSPARSE_ORDER_COL
#define HIPSPARSE_ORDER_ROW              GPUSPARSE_ORDER_ROW
#define HIPSPARSE_POINTER_MODE_DEVICE    GPUSPARSE_POINTER_MODE_DEVICE
#define HIPSPARSE_POINTER_MODE_HOST      GPUSPARSE_POINTER_MODE_HOST
#define HIPSPARSE_SDDMM_ALG_DEFAULT      GPUSPARSE_SDDMM_ALG_DEFAULT
#define HIPSPARSE_SOLVE_POLICY_NO_LEVEL  GPUSPARSE_SOLVE_POLICY_NO_LEVEL
#define HIPSPARSE_SOLVE_POLICY_USE_LEVEL GPUSPARSE_SOLVE_POLICY_USE_LEVEL
#define HIPSPARSE_SPARSETODENSE_ALG_DEFAULT  \
        GPUSPARSE_SPARSETODENSE_ALG_DEFAULT
#define HIPSPARSE_SPGEMM_ALG1            GPUSPARSE_SPGEMM_ALG1
#define HIPSPARSE_SPGEMM_ALG2            GPUSPARSE_SPGEMM_ALG2
#define HIPSPARSE_SPGEMM_ALG3            GPUSPARSE_SPGEMM_ALG3
#define HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC  \
        GPUSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC
#define HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC  \
        GPUSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC
#define HIPSPARSE_SPGEMM_DEFAULT         GPUSPARSE_SPGEMM_DEFAULT
#define HIPSPARSE_SPMAT_DIAG_TYPE        GPUSPARSE_SPMAT_DIAG_TYPE
#define HIPSPARSE_SPMAT_FILL_MODE        GPUSPARSE_SPMAT_FILL_MODE
#define HIPSPARSE_SPMM_ALG_DEFAULT       GPUSPARSE_SPMM_ALG_DEFAULT
#define HIPSPARSE_SPMM_BLOCKED_ELL_ALG1  GPUSPARSE_SPMM_BLOCKED_ELL_ALG1
#define HIPSPARSE_SPMM_COO_ALG1          GPUSPARSE_SPMM_COO_ALG1
#define HIPSPARSE_SPMM_COO_ALG2          GPUSPARSE_SPMM_COO_ALG2
#define HIPSPARSE_SPMM_COO_ALG3          GPUSPARSE_SPMM_COO_ALG3
#define HIPSPARSE_SPMM_COO_ALG4          GPUSPARSE_SPMM_COO_ALG4
#define HIPSPARSE_SPMM_CSR_ALG1          GPUSPARSE_SPMM_CSR_ALG1
#define HIPSPARSE_SPMM_CSR_ALG2          GPUSPARSE_SPMM_CSR_ALG2
#define HIPSPARSE_SPMM_CSR_ALG3          GPUSPARSE_SPMM_CSR_ALG3
#define HIPSPARSE_SPMV_ALG_DEFAULT       GPUSPARSE_SPMV_ALG_DEFAULT
#define HIPSPARSE_SPMV_COO_ALG1          GPUSPARSE_SPMV_COO_ALG1
#define HIPSPARSE_SPMV_COO_ALG2          GPUSPARSE_SPMV_COO_ALG2
#define HIPSPARSE_SPMV_CSR_ALG1          GPUSPARSE_SPMV_CSR_ALG1
#define HIPSPARSE_SPMV_CSR_ALG2          GPUSPARSE_SPMV_CSR_ALG2
#define HIPSPARSE_SPSM_ALG_DEFAULT       GPUSPARSE_SPSM_ALG_DEFAULT
#define HIPSPARSE_SPSV_ALG_DEFAULT       GPUSPARSE_SPSV_ALG_DEFAULT
#define HIPSPARSE_STATUS_ALLOC_FAILED    GPUSPARSE_STATUS_ALLOC_FAILED
#define HIPSPARSE_STATUS_ARCH_MISMATCH   GPUSPARSE_STATUS_ARCH_MISMATCH
#define HIPSPARSE_STATUS_EXECUTION_FAILED  \
        GPUSPARSE_STATUS_EXECUTION_FAILED
#define HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES  \
        GPUSPARSE_STATUS_INSUFFICIENT_RESOURCES
#define HIPSPARSE_STATUS_INTERNAL_ERROR  GPUSPARSE_STATUS_INTERNAL_ERROR
#define HIPSPARSE_STATUS_INVALID_VALUE   GPUSPARSE_STATUS_INVALID_VALUE
#define HIPSPARSE_STATUS_MAPPING_ERROR   GPUSPARSE_STATUS_MAPPING_ERROR
#define HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED  \
        GPUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define HIPSPARSE_STATUS_NOT_INITIALIZED GPUSPARSE_STATUS_NOT_INITIALIZED
#define HIPSPARSE_STATUS_NOT_SUPPORTED   GPUSPARSE_STATUS_NOT_SUPPORTED
#define HIPSPARSE_STATUS_SUCCESS         GPUSPARSE_STATUS_SUCCESS
#define HIPSPARSE_STATUS_ZERO_PIVOT      GPUSPARSE_STATUS_ZERO_PIVOT
#define hipsparseAction_t                gpusparseAction_t
#define hipsparseAxpby                   gpusparseAxpby
#define hipsparseBlockedEllGet           gpusparseBlockedEllGet
#define hipsparseCaxpyi                  gpusparseCaxpyi
#define hipsparseCbsr2csr                gpusparseCbsr2csr
#define hipsparseCbsric02                gpusparseCbsric02
#define hipsparseCbsric02_analysis       gpusparseCbsric02_analysis
#define hipsparseCbsric02_bufferSize     gpusparseCbsric02_bufferSize
#define hipsparseCbsrilu02               gpusparseCbsrilu02
#define hipsparseCbsrilu02_analysis      gpusparseCbsrilu02_analysis
#define hipsparseCbsrilu02_bufferSize    gpusparseCbsrilu02_bufferSize
#define hipsparseCbsrilu02_numericBoost  gpusparseCbsrilu02_numericBoost
#define hipsparseCbsrmm                  gpusparseCbsrmm
#define hipsparseCbsrmv                  gpusparseCbsrmv
#define hipsparseCbsrsm2_analysis        gpusparseCbsrsm2_analysis
#define hipsparseCbsrsm2_bufferSize      gpusparseCbsrsm2_bufferSize
#define hipsparseCbsrsm2_solve           gpusparseCbsrsm2_solve
#define hipsparseCbsrsv2_analysis        gpusparseCbsrsv2_analysis
#define hipsparseCbsrsv2_bufferSize      gpusparseCbsrsv2_bufferSize
#define hipsparseCbsrsv2_bufferSizeExt   gpusparseCbsrsv2_bufferSizeExt
#define hipsparseCbsrsv2_solve           gpusparseCbsrsv2_solve
#define hipsparseCbsrxmv                 gpusparseCbsrxmv
#define hipsparseCcsc2dense              gpusparseCcsc2dense
#define hipsparseCcsr2bsr                gpusparseCcsr2bsr
#define hipsparseCcsr2csr_compress       gpusparseCcsr2csr_compress
#define hipsparseCcsr2csru               gpusparseCcsr2csru
#define hipsparseCcsr2dense              gpusparseCcsr2dense
#define hipsparseCcsr2gebsr              gpusparseCcsr2gebsr
#define hipsparseCcsr2gebsr_bufferSize   gpusparseCcsr2gebsr_bufferSize
#define hipsparseCcsrcolor               gpusparseCcsrcolor
#define hipsparseCcsrgeam2               gpusparseCcsrgeam2
#define hipsparseCcsrgeam2_bufferSizeExt gpusparseCcsrgeam2_bufferSizeExt
#define hipsparseCcsrgemm2               gpusparseCcsrgemm2
#define hipsparseCcsrgemm2_bufferSizeExt gpusparseCcsrgemm2_bufferSizeExt
#define hipsparseCcsric02                gpusparseCcsric02
#define hipsparseCcsric02_analysis       gpusparseCcsric02_analysis
#define hipsparseCcsric02_bufferSize     gpusparseCcsric02_bufferSize
#define hipsparseCcsric02_bufferSizeExt  gpusparseCcsric02_bufferSizeExt
#define hipsparseCcsrilu02               gpusparseCcsrilu02
#define hipsparseCcsrilu02_analysis      gpusparseCcsrilu02_analysis
#define hipsparseCcsrilu02_bufferSize    gpusparseCcsrilu02_bufferSize
#define hipsparseCcsrilu02_bufferSizeExt gpusparseCcsrilu02_bufferSizeExt
#define hipsparseCcsrilu02_numericBoost  gpusparseCcsrilu02_numericBoost
#define hipsparseCcsrsm2_analysis        gpusparseCcsrsm2_analysis
#define hipsparseCcsrsm2_bufferSizeExt   gpusparseCcsrsm2_bufferSizeExt
#define hipsparseCcsrsm2_solve           gpusparseCcsrsm2_solve
#define hipsparseCcsrsv2_analysis        gpusparseCcsrsv2_analysis
#define hipsparseCcsrsv2_bufferSize      gpusparseCcsrsv2_bufferSize
#define hipsparseCcsrsv2_bufferSizeExt   gpusparseCcsrsv2_bufferSizeExt
#define hipsparseCcsrsv2_solve           gpusparseCcsrsv2_solve
#define hipsparseCcsru2csr               gpusparseCcsru2csr
#define hipsparseCcsru2csr_bufferSizeExt gpusparseCcsru2csr_bufferSizeExt
#define hipsparseCdense2csc              gpusparseCdense2csc
#define hipsparseCdense2csr              gpusparseCdense2csr
#define hipsparseCgebsr2csr              gpusparseCgebsr2csr
#define hipsparseCgebsr2gebsc            gpusparseCgebsr2gebsc
#define hipsparseCgebsr2gebsc_bufferSize gpusparseCgebsr2gebsc_bufferSize
#define hipsparseCgebsr2gebsr            gpusparseCgebsr2gebsr
#define hipsparseCgebsr2gebsr_bufferSize gpusparseCgebsr2gebsr_bufferSize
#define hipsparseCgemmi                  gpusparseCgemmi
#define hipsparseCgemvi                  gpusparseCgemvi
#define hipsparseCgemvi_bufferSize       gpusparseCgemvi_bufferSize
#define hipsparseCgpsvInterleavedBatch   gpusparseCgpsvInterleavedBatch
#define hipsparseCgpsvInterleavedBatch_bufferSizeExt  \
        gpusparseCgpsvInterleavedBatch_bufferSizeExt
#define hipsparseCgthr                   gpusparseCgthr
#define hipsparseCgthrz                  gpusparseCgthrz
#define hipsparseCgtsv2                  gpusparseCgtsv2
#define hipsparseCgtsv2StridedBatch      gpusparseCgtsv2StridedBatch
#define hipsparseCgtsv2StridedBatch_bufferSizeExt  \
        gpusparseCgtsv2StridedBatch_bufferSizeExt
#define hipsparseCgtsv2_bufferSizeExt    gpusparseCgtsv2_bufferSizeExt
#define hipsparseCgtsv2_nopivot          gpusparseCgtsv2_nopivot
#define hipsparseCgtsv2_nopivot_bufferSizeExt  \
        gpusparseCgtsv2_nopivot_bufferSizeExt
#define hipsparseCgtsvInterleavedBatch   gpusparseCgtsvInterleavedBatch
#define hipsparseCgtsvInterleavedBatch_bufferSizeExt  \
        gpusparseCgtsvInterleavedBatch_bufferSizeExt
#define hipsparseCnnz                    gpusparseCnnz
#define hipsparseCnnz_compress           gpusparseCnnz_compress
#define hipsparseColorInfo_t             gpusparseColorInfo_t
#define hipsparseCooAoSGet               gpusparseCooAoSGet
#define hipsparseCooGet                  gpusparseCooGet
#define hipsparseCooSetPointers          gpusparseCooSetPointers
#define hipsparseCooSetStridedBatch      gpusparseCooSetStridedBatch
#define hipsparseCopyMatDescr            gpusparseCopyMatDescr
#define hipsparseCreate                  gpusparseCreate
#define hipsparseCreateBlockedEll        gpusparseCreateBlockedEll
#define hipsparseCreateBsric02Info       gpusparseCreateBsric02Info
#define hipsparseCreateBsrilu02Info      gpusparseCreateBsrilu02Info
#define hipsparseCreateBsrsm2Info        gpusparseCreateBsrsm2Info
#define hipsparseCreateBsrsv2Info        gpusparseCreateBsrsv2Info
#define hipsparseCreateColorInfo         gpusparseCreateColorInfo
#define hipsparseCreateCoo               gpusparseCreateCoo
#define hipsparseCreateCooAoS            gpusparseCreateCooAoS
#define hipsparseCreateCsc               gpusparseCreateCsc
#define hipsparseCreateCsr               gpusparseCreateCsr
#define hipsparseCreateCsrgemm2Info      gpusparseCreateCsrgemm2Info
#define hipsparseCreateCsric02Info       gpusparseCreateCsric02Info
#define hipsparseCreateCsrilu02Info      gpusparseCreateCsrilu02Info
#define hipsparseCreateCsrsm2Info        gpusparseCreateCsrsm2Info
#define hipsparseCreateCsrsv2Info        gpusparseCreateCsrsv2Info
#define hipsparseCreateCsru2csrInfo      gpusparseCreateCsru2csrInfo
#define hipsparseCreateDnMat             gpusparseCreateDnMat
#define hipsparseCreateDnVec             gpusparseCreateDnVec
#define hipsparseCreateIdentityPermutation  \
        gpusparseCreateIdentityPermutation
#define hipsparseCreateMatDescr          gpusparseCreateMatDescr
#define hipsparseCreatePruneInfo         gpusparseCreatePruneInfo
#define hipsparseCreateSpVec             gpusparseCreateSpVec
#define hipsparseCscSetPointers          gpusparseCscSetPointers
#define hipsparseCsctr                   gpusparseCsctr
#define hipsparseCsr2CscAlg_t            gpusparseCsr2CscAlg_t
#define hipsparseCsr2cscEx2              gpusparseCsr2cscEx2
#define hipsparseCsr2cscEx2_bufferSize   gpusparseCsr2cscEx2_bufferSize
#define hipsparseCsrGet                  gpusparseCsrGet
#define hipsparseCsrSetPointers          gpusparseCsrSetPointers
#define hipsparseCsrSetStridedBatch      gpusparseCsrSetStridedBatch
#define hipsparseDaxpyi                  gpusparseDaxpyi
#define hipsparseDbsr2csr                gpusparseDbsr2csr
#define hipsparseDbsric02                gpusparseDbsric02
#define hipsparseDbsric02_analysis       gpusparseDbsric02_analysis
#define hipsparseDbsric02_bufferSize     gpusparseDbsric02_bufferSize
#define hipsparseDbsrilu02               gpusparseDbsrilu02
#define hipsparseDbsrilu02_analysis      gpusparseDbsrilu02_analysis
#define hipsparseDbsrilu02_bufferSize    gpusparseDbsrilu02_bufferSize
#define hipsparseDbsrilu02_numericBoost  gpusparseDbsrilu02_numericBoost
#define hipsparseDbsrmm                  gpusparseDbsrmm
#define hipsparseDbsrmv                  gpusparseDbsrmv
#define hipsparseDbsrsm2_analysis        gpusparseDbsrsm2_analysis
#define hipsparseDbsrsm2_bufferSize      gpusparseDbsrsm2_bufferSize
#define hipsparseDbsrsm2_solve           gpusparseDbsrsm2_solve
#define hipsparseDbsrsv2_analysis        gpusparseDbsrsv2_analysis
#define hipsparseDbsrsv2_bufferSize      gpusparseDbsrsv2_bufferSize
#define hipsparseDbsrsv2_bufferSizeExt   gpusparseDbsrsv2_bufferSizeExt
#define hipsparseDbsrsv2_solve           gpusparseDbsrsv2_solve
#define hipsparseDbsrxmv                 gpusparseDbsrxmv
#define hipsparseDcsc2dense              gpusparseDcsc2dense
#define hipsparseDcsr2bsr                gpusparseDcsr2bsr
#define hipsparseDcsr2csr_compress       gpusparseDcsr2csr_compress
#define hipsparseDcsr2csru               gpusparseDcsr2csru
#define hipsparseDcsr2dense              gpusparseDcsr2dense
#define hipsparseDcsr2gebsr              gpusparseDcsr2gebsr
#define hipsparseDcsr2gebsr_bufferSize   gpusparseDcsr2gebsr_bufferSize
#define hipsparseDcsrcolor               gpusparseDcsrcolor
#define hipsparseDcsrgeam2               gpusparseDcsrgeam2
#define hipsparseDcsrgeam2_bufferSizeExt gpusparseDcsrgeam2_bufferSizeExt
#define hipsparseDcsrgemm2               gpusparseDcsrgemm2
#define hipsparseDcsrgemm2_bufferSizeExt gpusparseDcsrgemm2_bufferSizeExt
#define hipsparseDcsric02                gpusparseDcsric02
#define hipsparseDcsric02_analysis       gpusparseDcsric02_analysis
#define hipsparseDcsric02_bufferSize     gpusparseDcsric02_bufferSize
#define hipsparseDcsric02_bufferSizeExt  gpusparseDcsric02_bufferSizeExt
#define hipsparseDcsrilu02               gpusparseDcsrilu02
#define hipsparseDcsrilu02_analysis      gpusparseDcsrilu02_analysis
#define hipsparseDcsrilu02_bufferSize    gpusparseDcsrilu02_bufferSize
#define hipsparseDcsrilu02_bufferSizeExt gpusparseDcsrilu02_bufferSizeExt
#define hipsparseDcsrilu02_numericBoost  gpusparseDcsrilu02_numericBoost
#define hipsparseDcsrsm2_analysis        gpusparseDcsrsm2_analysis
#define hipsparseDcsrsm2_bufferSizeExt   gpusparseDcsrsm2_bufferSizeExt
#define hipsparseDcsrsm2_solve           gpusparseDcsrsm2_solve
#define hipsparseDcsrsv2_analysis        gpusparseDcsrsv2_analysis
#define hipsparseDcsrsv2_bufferSize      gpusparseDcsrsv2_bufferSize
#define hipsparseDcsrsv2_bufferSizeExt   gpusparseDcsrsv2_bufferSizeExt
#define hipsparseDcsrsv2_solve           gpusparseDcsrsv2_solve
#define hipsparseDcsru2csr               gpusparseDcsru2csr
#define hipsparseDcsru2csr_bufferSizeExt gpusparseDcsru2csr_bufferSizeExt
#define hipsparseDdense2csc              gpusparseDdense2csc
#define hipsparseDdense2csr              gpusparseDdense2csr
#define hipsparseDenseToSparseAlg_t      gpusparseDenseToSparseAlg_t
#define hipsparseDenseToSparse_analysis  gpusparseDenseToSparse_analysis
#define hipsparseDenseToSparse_bufferSize  \
        gpusparseDenseToSparse_bufferSize
#define hipsparseDenseToSparse_convert   gpusparseDenseToSparse_convert
#define hipsparseDestroy                 gpusparseDestroy
#define hipsparseDestroyBsric02Info      gpusparseDestroyBsric02Info
#define hipsparseDestroyBsrilu02Info     gpusparseDestroyBsrilu02Info
#define hipsparseDestroyBsrsm2Info       gpusparseDestroyBsrsm2Info
#define hipsparseDestroyBsrsv2Info       gpusparseDestroyBsrsv2Info
#define hipsparseDestroyColorInfo        gpusparseDestroyColorInfo
#define hipsparseDestroyCsrgemm2Info     gpusparseDestroyCsrgemm2Info
#define hipsparseDestroyCsric02Info      gpusparseDestroyCsric02Info
#define hipsparseDestroyCsrilu02Info     gpusparseDestroyCsrilu02Info
#define hipsparseDestroyCsrsm2Info       gpusparseDestroyCsrsm2Info
#define hipsparseDestroyCsrsv2Info       gpusparseDestroyCsrsv2Info
#define hipsparseDestroyCsru2csrInfo     gpusparseDestroyCsru2csrInfo
#define hipsparseDestroyDnMat            gpusparseDestroyDnMat
#define hipsparseDestroyDnVec            gpusparseDestroyDnVec
#define hipsparseDestroyMatDescr         gpusparseDestroyMatDescr
#define hipsparseDestroyPruneInfo        gpusparseDestroyPruneInfo
#define hipsparseDestroySpMat            gpusparseDestroySpMat
#define hipsparseDestroySpVec            gpusparseDestroySpVec
#define hipsparseDgebsr2csr              gpusparseDgebsr2csr
#define hipsparseDgebsr2gebsc            gpusparseDgebsr2gebsc
#define hipsparseDgebsr2gebsc_bufferSize gpusparseDgebsr2gebsc_bufferSize
#define hipsparseDgebsr2gebsr            gpusparseDgebsr2gebsr
#define hipsparseDgebsr2gebsr_bufferSize gpusparseDgebsr2gebsr_bufferSize
#define hipsparseDgemmi                  gpusparseDgemmi
#define hipsparseDgemvi                  gpusparseDgemvi
#define hipsparseDgemvi_bufferSize       gpusparseDgemvi_bufferSize
#define hipsparseDgpsvInterleavedBatch   gpusparseDgpsvInterleavedBatch
#define hipsparseDgpsvInterleavedBatch_bufferSizeExt  \
        gpusparseDgpsvInterleavedBatch_bufferSizeExt
#define hipsparseDgthr                   gpusparseDgthr
#define hipsparseDgthrz                  gpusparseDgthrz
#define hipsparseDgtsv2                  gpusparseDgtsv2
#define hipsparseDgtsv2StridedBatch      gpusparseDgtsv2StridedBatch
#define hipsparseDgtsv2StridedBatch_bufferSizeExt  \
        gpusparseDgtsv2StridedBatch_bufferSizeExt
#define hipsparseDgtsv2_bufferSizeExt    gpusparseDgtsv2_bufferSizeExt
#define hipsparseDgtsv2_nopivot          gpusparseDgtsv2_nopivot
#define hipsparseDgtsv2_nopivot_bufferSizeExt  \
        gpusparseDgtsv2_nopivot_bufferSizeExt
#define hipsparseDgtsvInterleavedBatch   gpusparseDgtsvInterleavedBatch
#define hipsparseDgtsvInterleavedBatch_bufferSizeExt  \
        gpusparseDgtsvInterleavedBatch_bufferSizeExt
#define hipsparseDiagType_t              gpusparseDiagType_t
#define hipsparseDirection_t             gpusparseDirection_t
#define hipsparseDnMatDescr_t            gpusparseDnMatDescr_t
#define hipsparseDnMatGet                gpusparseDnMatGet
#define hipsparseDnMatGetStridedBatch    gpusparseDnMatGetStridedBatch
#define hipsparseDnMatGetValues          gpusparseDnMatGetValues
#define hipsparseDnMatSetStridedBatch    gpusparseDnMatSetStridedBatch
#define hipsparseDnMatSetValues          gpusparseDnMatSetValues
#define hipsparseDnVecDescr_t            gpusparseDnVecDescr_t
#define hipsparseDnVecGet                gpusparseDnVecGet
#define hipsparseDnVecGetValues          gpusparseDnVecGetValues
#define hipsparseDnVecSetValues          gpusparseDnVecSetValues
#define hipsparseDnnz                    gpusparseDnnz
#define hipsparseDnnz_compress           gpusparseDnnz_compress
#define hipsparseDpruneCsr2csr           gpusparseDpruneCsr2csr
#define hipsparseDpruneCsr2csrByPercentage  \
        gpusparseDpruneCsr2csrByPercentage
#define hipsparseDpruneCsr2csrByPercentage_bufferSizeExt  \
        gpusparseDpruneCsr2csrByPercentage_bufferSizeExt
#define hipsparseDpruneCsr2csrNnz        gpusparseDpruneCsr2csrNnz
#define hipsparseDpruneCsr2csrNnzByPercentage  \
        gpusparseDpruneCsr2csrNnzByPercentage
#define hipsparseDpruneCsr2csr_bufferSizeExt  \
        gpusparseDpruneCsr2csr_bufferSizeExt
#define hipsparseDpruneDense2csr         gpusparseDpruneDense2csr
#define hipsparseDpruneDense2csrByPercentage  \
        gpusparseDpruneDense2csrByPercentage
#define hipsparseDpruneDense2csrByPercentage_bufferSizeExt  \
        gpusparseDpruneDense2csrByPercentage_bufferSizeExt
#define hipsparseDpruneDense2csrNnz      gpusparseDpruneDense2csrNnz
#define hipsparseDpruneDense2csrNnzByPercentage  \
        gpusparseDpruneDense2csrNnzByPercentage
#define hipsparseDpruneDense2csr_bufferSizeExt  \
        gpusparseDpruneDense2csr_bufferSizeExt
#define hipsparseDroti                   gpusparseDroti
#define hipsparseDsctr                   gpusparseDsctr
#define hipsparseFillMode_t              gpusparseFillMode_t
#define hipsparseFormat_t                gpusparseFormat_t
#define hipsparseGather                  gpusparseGather
#define hipsparseGetMatDiagType          gpusparseGetMatDiagType
#define hipsparseGetMatFillMode          gpusparseGetMatFillMode
#define hipsparseGetMatIndexBase         gpusparseGetMatIndexBase
#define hipsparseGetMatType              gpusparseGetMatType
#define hipsparseGetPointerMode          gpusparseGetPointerMode
#define hipsparseGetStream               gpusparseGetStream
#define hipsparseGetVersion              gpusparseGetVersion
#define hipsparseHandle_t                gpusparseHandle_t
#define hipsparseIndexBase_t             gpusparseIndexBase_t
#define hipsparseIndexType_t             gpusparseIndexType_t
#define hipsparseMatDescr_t              gpusparseMatDescr_t
#define hipsparseMatrixType_t            gpusparseMatrixType_t
#define hipsparseOperation_t             gpusparseOperation_t
#define hipsparseOrder_t                 gpusparseOrder_t
#define hipsparsePointerMode_t           gpusparsePointerMode_t
#define hipsparseRot                     gpusparseRot
#define hipsparseSDDMM                   gpusparseSDDMM
#define hipsparseSDDMMAlg_t              gpusparseSDDMMAlg_t
#define hipsparseSDDMM_bufferSize        gpusparseSDDMM_bufferSize
#define hipsparseSDDMM_preprocess        gpusparseSDDMM_preprocess
#define hipsparseSaxpyi                  gpusparseSaxpyi
#define hipsparseSbsr2csr                gpusparseSbsr2csr
#define hipsparseSbsric02                gpusparseSbsric02
#define hipsparseSbsric02_analysis       gpusparseSbsric02_analysis
#define hipsparseSbsric02_bufferSize     gpusparseSbsric02_bufferSize
#define hipsparseSbsrilu02               gpusparseSbsrilu02
#define hipsparseSbsrilu02_analysis      gpusparseSbsrilu02_analysis
#define hipsparseSbsrilu02_bufferSize    gpusparseSbsrilu02_bufferSize
#define hipsparseSbsrilu02_numericBoost  gpusparseSbsrilu02_numericBoost
#define hipsparseSbsrmm                  gpusparseSbsrmm
#define hipsparseSbsrmv                  gpusparseSbsrmv
#define hipsparseSbsrsm2_analysis        gpusparseSbsrsm2_analysis
#define hipsparseSbsrsm2_bufferSize      gpusparseSbsrsm2_bufferSize
#define hipsparseSbsrsm2_solve           gpusparseSbsrsm2_solve
#define hipsparseSbsrsv2_analysis        gpusparseSbsrsv2_analysis
#define hipsparseSbsrsv2_bufferSize      gpusparseSbsrsv2_bufferSize
#define hipsparseSbsrsv2_bufferSizeExt   gpusparseSbsrsv2_bufferSizeExt
#define hipsparseSbsrsv2_solve           gpusparseSbsrsv2_solve
#define hipsparseSbsrxmv                 gpusparseSbsrxmv
#define hipsparseScatter                 gpusparseScatter
#define hipsparseScsc2dense              gpusparseScsc2dense
#define hipsparseScsr2bsr                gpusparseScsr2bsr
#define hipsparseScsr2csr_compress       gpusparseScsr2csr_compress
#define hipsparseScsr2csru               gpusparseScsr2csru
#define hipsparseScsr2dense              gpusparseScsr2dense
#define hipsparseScsr2gebsr              gpusparseScsr2gebsr
#define hipsparseScsr2gebsr_bufferSize   gpusparseScsr2gebsr_bufferSize
#define hipsparseScsrcolor               gpusparseScsrcolor
#define hipsparseScsrgeam2               gpusparseScsrgeam2
#define hipsparseScsrgeam2_bufferSizeExt gpusparseScsrgeam2_bufferSizeExt
#define hipsparseScsrgemm2               gpusparseScsrgemm2
#define hipsparseScsrgemm2_bufferSizeExt gpusparseScsrgemm2_bufferSizeExt
#define hipsparseScsric02                gpusparseScsric02
#define hipsparseScsric02_analysis       gpusparseScsric02_analysis
#define hipsparseScsric02_bufferSize     gpusparseScsric02_bufferSize
#define hipsparseScsric02_bufferSizeExt  gpusparseScsric02_bufferSizeExt
#define hipsparseScsrilu02               gpusparseScsrilu02
#define hipsparseScsrilu02_analysis      gpusparseScsrilu02_analysis
#define hipsparseScsrilu02_bufferSize    gpusparseScsrilu02_bufferSize
#define hipsparseScsrilu02_bufferSizeExt gpusparseScsrilu02_bufferSizeExt
#define hipsparseScsrilu02_numericBoost  gpusparseScsrilu02_numericBoost
#define hipsparseScsrsm2_analysis        gpusparseScsrsm2_analysis
#define hipsparseScsrsm2_bufferSizeExt   gpusparseScsrsm2_bufferSizeExt
#define hipsparseScsrsm2_solve           gpusparseScsrsm2_solve
#define hipsparseScsrsv2_analysis        gpusparseScsrsv2_analysis
#define hipsparseScsrsv2_bufferSize      gpusparseScsrsv2_bufferSize
#define hipsparseScsrsv2_bufferSizeExt   gpusparseScsrsv2_bufferSizeExt
#define hipsparseScsrsv2_solve           gpusparseScsrsv2_solve
#define hipsparseScsru2csr               gpusparseScsru2csr
#define hipsparseScsru2csr_bufferSizeExt gpusparseScsru2csr_bufferSizeExt
#define hipsparseSdense2csc              gpusparseSdense2csc
#define hipsparseSdense2csr              gpusparseSdense2csr
#define hipsparseSetMatDiagType          gpusparseSetMatDiagType
#define hipsparseSetMatFillMode          gpusparseSetMatFillMode
#define hipsparseSetMatIndexBase         gpusparseSetMatIndexBase
#define hipsparseSetMatType              gpusparseSetMatType
#define hipsparseSetPointerMode          gpusparseSetPointerMode
#define hipsparseSetStream               gpusparseSetStream
#define hipsparseSgebsr2csr              gpusparseSgebsr2csr
#define hipsparseSgebsr2gebsc            gpusparseSgebsr2gebsc
#define hipsparseSgebsr2gebsc_bufferSize gpusparseSgebsr2gebsc_bufferSize
#define hipsparseSgebsr2gebsr            gpusparseSgebsr2gebsr
#define hipsparseSgebsr2gebsr_bufferSize gpusparseSgebsr2gebsr_bufferSize
#define hipsparseSgemmi                  gpusparseSgemmi
#define hipsparseSgemvi                  gpusparseSgemvi
#define hipsparseSgemvi_bufferSize       gpusparseSgemvi_bufferSize
#define hipsparseSgpsvInterleavedBatch   gpusparseSgpsvInterleavedBatch
#define hipsparseSgpsvInterleavedBatch_bufferSizeExt  \
        gpusparseSgpsvInterleavedBatch_bufferSizeExt
#define hipsparseSgthr                   gpusparseSgthr
#define hipsparseSgthrz                  gpusparseSgthrz
#define hipsparseSgtsv2                  gpusparseSgtsv2
#define hipsparseSgtsv2StridedBatch      gpusparseSgtsv2StridedBatch
#define hipsparseSgtsv2StridedBatch_bufferSizeExt  \
        gpusparseSgtsv2StridedBatch_bufferSizeExt
#define hipsparseSgtsv2_bufferSizeExt    gpusparseSgtsv2_bufferSizeExt
#define hipsparseSgtsv2_nopivot          gpusparseSgtsv2_nopivot
#define hipsparseSgtsv2_nopivot_bufferSizeExt  \
        gpusparseSgtsv2_nopivot_bufferSizeExt
#define hipsparseSgtsvInterleavedBatch   gpusparseSgtsvInterleavedBatch
#define hipsparseSgtsvInterleavedBatch_bufferSizeExt  \
        gpusparseSgtsvInterleavedBatch_bufferSizeExt
#define hipsparseSnnz                    gpusparseSnnz
#define hipsparseSnnz_compress           gpusparseSnnz_compress
#define hipsparseSolvePolicy_t           gpusparseSolvePolicy_t
#define hipsparseSpGEMMAlg_t             gpusparseSpGEMMAlg_t
#define hipsparseSpGEMMDescr_t           gpusparseSpGEMMDescr_t
#define hipsparseSpGEMM_compute          gpusparseSpGEMM_compute
#define hipsparseSpGEMM_copy             gpusparseSpGEMM_copy
#define hipsparseSpGEMM_createDescr      gpusparseSpGEMM_createDescr
#define hipsparseSpGEMM_destroyDescr     gpusparseSpGEMM_destroyDescr
#define hipsparseSpGEMM_workEstimation   gpusparseSpGEMM_workEstimation
#define hipsparseSpMM                    gpusparseSpMM
#define hipsparseSpMMAlg_t               gpusparseSpMMAlg_t
#define hipsparseSpMM_bufferSize         gpusparseSpMM_bufferSize
#define hipsparseSpMM_preprocess         gpusparseSpMM_preprocess
#define hipsparseSpMV                    gpusparseSpMV
#define hipsparseSpMVAlg_t               gpusparseSpMVAlg_t
#define hipsparseSpMV_bufferSize         gpusparseSpMV_bufferSize
#define hipsparseSpMatAttribute_t        gpusparseSpMatAttribute_t
#define hipsparseSpMatDescr_t            gpusparseSpMatDescr_t
#define hipsparseSpMatGetAttribute       gpusparseSpMatGetAttribute
#define hipsparseSpMatGetFormat          gpusparseSpMatGetFormat
#define hipsparseSpMatGetIndexBase       gpusparseSpMatGetIndexBase
#define hipsparseSpMatGetSize            gpusparseSpMatGetSize
#define hipsparseSpMatGetStridedBatch    gpusparseSpMatGetStridedBatch
#define hipsparseSpMatGetValues          gpusparseSpMatGetValues
#define hipsparseSpMatSetAttribute       gpusparseSpMatSetAttribute
#define hipsparseSpMatSetStridedBatch    gpusparseSpMatSetStridedBatch
#define hipsparseSpMatSetValues          gpusparseSpMatSetValues
#define hipsparseSpSMAlg_t               gpusparseSpSMAlg_t
#define hipsparseSpSMDescr_t             gpusparseSpSMDescr_t
#define hipsparseSpSM_analysis           gpusparseSpSM_analysis
#define hipsparseSpSM_bufferSize         gpusparseSpSM_bufferSize
#define hipsparseSpSM_createDescr        gpusparseSpSM_createDescr
#define hipsparseSpSM_destroyDescr       gpusparseSpSM_destroyDescr
#define hipsparseSpSM_solve              gpusparseSpSM_solve
#define hipsparseSpSVAlg_t               gpusparseSpSVAlg_t
#define hipsparseSpSVDescr_t             gpusparseSpSVDescr_t
#define hipsparseSpSV_analysis           gpusparseSpSV_analysis
#define hipsparseSpSV_bufferSize         gpusparseSpSV_bufferSize
#define hipsparseSpSV_createDescr        gpusparseSpSV_createDescr
#define hipsparseSpSV_destroyDescr       gpusparseSpSV_destroyDescr
#define hipsparseSpSV_solve              gpusparseSpSV_solve
#define hipsparseSpVV                    gpusparseSpVV
#define hipsparseSpVV_bufferSize         gpusparseSpVV_bufferSize
#define hipsparseSpVecDescr_t            gpusparseSpVecDescr_t
#define hipsparseSpVecGet                gpusparseSpVecGet
#define hipsparseSpVecGetIndexBase       gpusparseSpVecGetIndexBase
#define hipsparseSpVecGetValues          gpusparseSpVecGetValues
#define hipsparseSpVecSetValues          gpusparseSpVecSetValues
#define hipsparseSparseToDense           gpusparseSparseToDense
#define hipsparseSparseToDenseAlg_t      gpusparseSparseToDenseAlg_t
#define hipsparseSparseToDense_bufferSize  \
        gpusparseSparseToDense_bufferSize
#define hipsparseSpruneCsr2csr           gpusparseSpruneCsr2csr
#define hipsparseSpruneCsr2csrByPercentage  \
        gpusparseSpruneCsr2csrByPercentage
#define hipsparseSpruneCsr2csrByPercentage_bufferSizeExt  \
        gpusparseSpruneCsr2csrByPercentage_bufferSizeExt
#define hipsparseSpruneCsr2csrNnz        gpusparseSpruneCsr2csrNnz
#define hipsparseSpruneCsr2csrNnzByPercentage  \
        gpusparseSpruneCsr2csrNnzByPercentage
#define hipsparseSpruneCsr2csr_bufferSizeExt  \
        gpusparseSpruneCsr2csr_bufferSizeExt
#define hipsparseSpruneDense2csr         gpusparseSpruneDense2csr
#define hipsparseSpruneDense2csrByPercentage  \
        gpusparseSpruneDense2csrByPercentage
#define hipsparseSpruneDense2csrByPercentage_bufferSizeExt  \
        gpusparseSpruneDense2csrByPercentage_bufferSizeExt
#define hipsparseSpruneDense2csrNnz      gpusparseSpruneDense2csrNnz
#define hipsparseSpruneDense2csrNnzByPercentage  \
        gpusparseSpruneDense2csrNnzByPercentage
#define hipsparseSpruneDense2csr_bufferSizeExt  \
        gpusparseSpruneDense2csr_bufferSizeExt
#define hipsparseSroti                   gpusparseSroti
#define hipsparseSsctr                   gpusparseSsctr
#define hipsparseStatus_t                gpusparseStatus_t
#define hipsparseXbsric02_zeroPivot      gpusparseXbsric02_zeroPivot
#define hipsparseXbsrilu02_zeroPivot     gpusparseXbsrilu02_zeroPivot
#define hipsparseXbsrsm2_zeroPivot       gpusparseXbsrsm2_zeroPivot
#define hipsparseXbsrsv2_zeroPivot       gpusparseXbsrsv2_zeroPivot
#define hipsparseXcoo2csr                gpusparseXcoo2csr
#define hipsparseXcoosortByColumn        gpusparseXcoosortByColumn
#define hipsparseXcoosortByRow           gpusparseXcoosortByRow
#define hipsparseXcoosort_bufferSizeExt  gpusparseXcoosort_bufferSizeExt
#define hipsparseXcscsort                gpusparseXcscsort
#define hipsparseXcscsort_bufferSizeExt  gpusparseXcscsort_bufferSizeExt
#define hipsparseXcsr2bsrNnz             gpusparseXcsr2bsrNnz
#define hipsparseXcsr2coo                gpusparseXcsr2coo
#define hipsparseXcsr2gebsrNnz           gpusparseXcsr2gebsrNnz
#define hipsparseXcsrgeam2Nnz            gpusparseXcsrgeam2Nnz
#define hipsparseXcsrgemm2Nnz            gpusparseXcsrgemm2Nnz
#define hipsparseXcsric02_zeroPivot      gpusparseXcsric02_zeroPivot
#define hipsparseXcsrilu02_zeroPivot     gpusparseXcsrilu02_zeroPivot
#define hipsparseXcsrsm2_zeroPivot       gpusparseXcsrsm2_zeroPivot
#define hipsparseXcsrsort                gpusparseXcsrsort
#define hipsparseXcsrsort_bufferSizeExt  gpusparseXcsrsort_bufferSizeExt
#define hipsparseXcsrsv2_zeroPivot       gpusparseXcsrsv2_zeroPivot
#define hipsparseXgebsr2gebsrNnz         gpusparseXgebsr2gebsrNnz
#define hipsparseZaxpyi                  gpusparseZaxpyi
#define hipsparseZbsr2csr                gpusparseZbsr2csr
#define hipsparseZbsric02                gpusparseZbsric02
#define hipsparseZbsric02_analysis       gpusparseZbsric02_analysis
#define hipsparseZbsric02_bufferSize     gpusparseZbsric02_bufferSize
#define hipsparseZbsrilu02               gpusparseZbsrilu02
#define hipsparseZbsrilu02_analysis      gpusparseZbsrilu02_analysis
#define hipsparseZbsrilu02_bufferSize    gpusparseZbsrilu02_bufferSize
#define hipsparseZbsrilu02_numericBoost  gpusparseZbsrilu02_numericBoost
#define hipsparseZbsrmm                  gpusparseZbsrmm
#define hipsparseZbsrmv                  gpusparseZbsrmv
#define hipsparseZbsrsm2_analysis        gpusparseZbsrsm2_analysis
#define hipsparseZbsrsm2_bufferSize      gpusparseZbsrsm2_bufferSize
#define hipsparseZbsrsm2_solve           gpusparseZbsrsm2_solve
#define hipsparseZbsrsv2_analysis        gpusparseZbsrsv2_analysis
#define hipsparseZbsrsv2_bufferSize      gpusparseZbsrsv2_bufferSize
#define hipsparseZbsrsv2_bufferSizeExt   gpusparseZbsrsv2_bufferSizeExt
#define hipsparseZbsrsv2_solve           gpusparseZbsrsv2_solve
#define hipsparseZbsrxmv                 gpusparseZbsrxmv
#define hipsparseZcsc2dense              gpusparseZcsc2dense
#define hipsparseZcsr2bsr                gpusparseZcsr2bsr
#define hipsparseZcsr2csr_compress       gpusparseZcsr2csr_compress
#define hipsparseZcsr2csru               gpusparseZcsr2csru
#define hipsparseZcsr2dense              gpusparseZcsr2dense
#define hipsparseZcsr2gebsr              gpusparseZcsr2gebsr
#define hipsparseZcsr2gebsr_bufferSize   gpusparseZcsr2gebsr_bufferSize
#define hipsparseZcsrcolor               gpusparseZcsrcolor
#define hipsparseZcsrgeam2               gpusparseZcsrgeam2
#define hipsparseZcsrgeam2_bufferSizeExt gpusparseZcsrgeam2_bufferSizeExt
#define hipsparseZcsrgemm2               gpusparseZcsrgemm2
#define hipsparseZcsrgemm2_bufferSizeExt gpusparseZcsrgemm2_bufferSizeExt
#define hipsparseZcsric02                gpusparseZcsric02
#define hipsparseZcsric02_analysis       gpusparseZcsric02_analysis
#define hipsparseZcsric02_bufferSize     gpusparseZcsric02_bufferSize
#define hipsparseZcsric02_bufferSizeExt  gpusparseZcsric02_bufferSizeExt
#define hipsparseZcsrilu02               gpusparseZcsrilu02
#define hipsparseZcsrilu02_analysis      gpusparseZcsrilu02_analysis
#define hipsparseZcsrilu02_bufferSize    gpusparseZcsrilu02_bufferSize
#define hipsparseZcsrilu02_bufferSizeExt gpusparseZcsrilu02_bufferSizeExt
#define hipsparseZcsrilu02_numericBoost  gpusparseZcsrilu02_numericBoost
#define hipsparseZcsrsm2_analysis        gpusparseZcsrsm2_analysis
#define hipsparseZcsrsm2_bufferSizeExt   gpusparseZcsrsm2_bufferSizeExt
#define hipsparseZcsrsm2_solve           gpusparseZcsrsm2_solve
#define hipsparseZcsrsv2_analysis        gpusparseZcsrsv2_analysis
#define hipsparseZcsrsv2_bufferSize      gpusparseZcsrsv2_bufferSize
#define hipsparseZcsrsv2_bufferSizeExt   gpusparseZcsrsv2_bufferSizeExt
#define hipsparseZcsrsv2_solve           gpusparseZcsrsv2_solve
#define hipsparseZcsru2csr               gpusparseZcsru2csr
#define hipsparseZcsru2csr_bufferSizeExt gpusparseZcsru2csr_bufferSizeExt
#define hipsparseZdense2csc              gpusparseZdense2csc
#define hipsparseZdense2csr              gpusparseZdense2csr
#define hipsparseZgebsr2csr              gpusparseZgebsr2csr
#define hipsparseZgebsr2gebsc            gpusparseZgebsr2gebsc
#define hipsparseZgebsr2gebsc_bufferSize gpusparseZgebsr2gebsc_bufferSize
#define hipsparseZgebsr2gebsr            gpusparseZgebsr2gebsr
#define hipsparseZgebsr2gebsr_bufferSize gpusparseZgebsr2gebsr_bufferSize
#define hipsparseZgemmi                  gpusparseZgemmi
#define hipsparseZgemvi                  gpusparseZgemvi
#define hipsparseZgemvi_bufferSize       gpusparseZgemvi_bufferSize
#define hipsparseZgpsvInterleavedBatch   gpusparseZgpsvInterleavedBatch
#define hipsparseZgpsvInterleavedBatch_bufferSizeExt  \
        gpusparseZgpsvInterleavedBatch_bufferSizeExt
#define hipsparseZgthr                   gpusparseZgthr
#define hipsparseZgthrz                  gpusparseZgthrz
#define hipsparseZgtsv2                  gpusparseZgtsv2
#define hipsparseZgtsv2StridedBatch      gpusparseZgtsv2StridedBatch
#define hipsparseZgtsv2StridedBatch_bufferSizeExt  \
        gpusparseZgtsv2StridedBatch_bufferSizeExt
#define hipsparseZgtsv2_bufferSizeExt    gpusparseZgtsv2_bufferSizeExt
#define hipsparseZgtsv2_nopivot          gpusparseZgtsv2_nopivot
#define hipsparseZgtsv2_nopivot_bufferSizeExt  \
        gpusparseZgtsv2_nopivot_bufferSizeExt
#define hipsparseZgtsvInterleavedBatch   gpusparseZgtsvInterleavedBatch
#define hipsparseZgtsvInterleavedBatch_bufferSizeExt  \
        gpusparseZgtsvInterleavedBatch_bufferSizeExt
#define hipsparseZnnz                    gpusparseZnnz
#define hipsparseZnnz_compress           gpusparseZnnz_compress
#define hipsparseZsctr                   gpusparseZsctr

#include <hop/hopsparse.h>

#endif
