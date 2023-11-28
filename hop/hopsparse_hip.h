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

#ifndef __HOP_HOPSPARSE_HIP_H__
#define __HOP_HOPSPARSE_HIP_H__

#include <hipsparse/hipsparse.h>

#define GPUSPARSE_ACTION_NUMERIC         HIPSPARSE_ACTION_NUMERIC
#define GPUSPARSE_ACTION_SYMBOLIC        HIPSPARSE_ACTION_SYMBOLIC
#define GPUSPARSE_COOMM_ALG1             HIPSPARSE_COOMM_ALG1
#define GPUSPARSE_COOMM_ALG2             HIPSPARSE_COOMM_ALG2
#define GPUSPARSE_COOMM_ALG3             HIPSPARSE_COOMM_ALG3
#define GPUSPARSE_COOMV_ALG              HIPSPARSE_COOMV_ALG
#define GPUSPARSE_CSR2CSC_ALG1           HIPSPARSE_CSR2CSC_ALG1
#define GPUSPARSE_CSR2CSC_ALG2           HIPSPARSE_CSR2CSC_ALG2
#define GPUSPARSE_CSR2CSC_ALG_DEFAULT    HIPSPARSE_CSR2CSC_ALG_DEFAULT
#define GPUSPARSE_CSRMM_ALG1             HIPSPARSE_CSRMM_ALG1
#define GPUSPARSE_CSRMV_ALG1             HIPSPARSE_CSRMV_ALG1
#define GPUSPARSE_CSRMV_ALG2             HIPSPARSE_CSRMV_ALG2
#define GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT  \
        HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT
#define GPUSPARSE_DIAG_TYPE_NON_UNIT     HIPSPARSE_DIAG_TYPE_NON_UNIT
#define GPUSPARSE_DIAG_TYPE_UNIT         HIPSPARSE_DIAG_TYPE_UNIT
#define GPUSPARSE_DIRECTION_COLUMN       HIPSPARSE_DIRECTION_COLUMN
#define GPUSPARSE_DIRECTION_ROW          HIPSPARSE_DIRECTION_ROW
#define GPUSPARSE_FILL_MODE_LOWER        HIPSPARSE_FILL_MODE_LOWER
#define GPUSPARSE_FILL_MODE_UPPER        HIPSPARSE_FILL_MODE_UPPER
#define GPUSPARSE_FORMAT_BLOCKED_ELL     HIPSPARSE_FORMAT_BLOCKED_ELL
#define GPUSPARSE_FORMAT_COO             HIPSPARSE_FORMAT_COO
#define GPUSPARSE_FORMAT_COO_AOS         HIPSPARSE_FORMAT_COO_AOS
#define GPUSPARSE_FORMAT_CSC             HIPSPARSE_FORMAT_CSC
#define GPUSPARSE_FORMAT_CSR             HIPSPARSE_FORMAT_CSR
#define GPUSPARSE_INDEX_16U              HIPSPARSE_INDEX_16U
#define GPUSPARSE_INDEX_32I              HIPSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_64I              HIPSPARSE_INDEX_64I
#define GPUSPARSE_INDEX_BASE_ONE         HIPSPARSE_INDEX_BASE_ONE
#define GPUSPARSE_INDEX_BASE_ZERO        HIPSPARSE_INDEX_BASE_ZERO
#define GPUSPARSE_MATRIX_TYPE_GENERAL    HIPSPARSE_MATRIX_TYPE_GENERAL
#define GPUSPARSE_MATRIX_TYPE_HERMITIAN  HIPSPARSE_MATRIX_TYPE_HERMITIAN
#define GPUSPARSE_MATRIX_TYPE_SYMMETRIC  HIPSPARSE_MATRIX_TYPE_SYMMETRIC
#define GPUSPARSE_MATRIX_TYPE_TRIANGULAR HIPSPARSE_MATRIX_TYPE_TRIANGULAR
#define GPUSPARSE_MM_ALG_DEFAULT         HIPSPARSE_MM_ALG_DEFAULT
#define GPUSPARSE_MV_ALG_DEFAULT         HIPSPARSE_MV_ALG_DEFAULT
#define GPUSPARSE_OPERATION_CONJUGATE_TRANSPOSE  \
        HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE
#define GPUSPARSE_OPERATION_NON_TRANSPOSE  \
        HIPSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_OPERATION_TRANSPOSE    HIPSPARSE_OPERATION_TRANSPOSE
#define GPUSPARSE_ORDER_COL              HIPSPARSE_ORDER_COL
#define GPUSPARSE_ORDER_ROW              HIPSPARSE_ORDER_ROW
#define GPUSPARSE_POINTER_MODE_DEVICE    HIPSPARSE_POINTER_MODE_DEVICE
#define GPUSPARSE_POINTER_MODE_HOST      HIPSPARSE_POINTER_MODE_HOST
#define GPUSPARSE_SDDMM_ALG_DEFAULT      HIPSPARSE_SDDMM_ALG_DEFAULT
#define GPUSPARSE_SOLVE_POLICY_NO_LEVEL  HIPSPARSE_SOLVE_POLICY_NO_LEVEL
#define GPUSPARSE_SOLVE_POLICY_USE_LEVEL HIPSPARSE_SOLVE_POLICY_USE_LEVEL
#define GPUSPARSE_SPARSETODENSE_ALG_DEFAULT  \
        HIPSPARSE_SPARSETODENSE_ALG_DEFAULT
#define GPUSPARSE_SPGEMM_ALG1            HIPSPARSE_SPGEMM_ALG1
#define GPUSPARSE_SPGEMM_ALG2            HIPSPARSE_SPGEMM_ALG2
#define GPUSPARSE_SPGEMM_ALG3            HIPSPARSE_SPGEMM_ALG3
#define GPUSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC  \
        HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC
#define GPUSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC  \
        HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC
#define GPUSPARSE_SPGEMM_DEFAULT         HIPSPARSE_SPGEMM_DEFAULT
#define GPUSPARSE_SPMAT_DIAG_TYPE        HIPSPARSE_SPMAT_DIAG_TYPE
#define GPUSPARSE_SPMAT_FILL_MODE        HIPSPARSE_SPMAT_FILL_MODE
#define GPUSPARSE_SPMM_ALG_DEFAULT       HIPSPARSE_SPMM_ALG_DEFAULT
#define GPUSPARSE_SPMM_BLOCKED_ELL_ALG1  HIPSPARSE_SPMM_BLOCKED_ELL_ALG1
#define GPUSPARSE_SPMM_COO_ALG1          HIPSPARSE_SPMM_COO_ALG1
#define GPUSPARSE_SPMM_COO_ALG2          HIPSPARSE_SPMM_COO_ALG2
#define GPUSPARSE_SPMM_COO_ALG3          HIPSPARSE_SPMM_COO_ALG3
#define GPUSPARSE_SPMM_COO_ALG4          HIPSPARSE_SPMM_COO_ALG4
#define GPUSPARSE_SPMM_CSR_ALG1          HIPSPARSE_SPMM_CSR_ALG1
#define GPUSPARSE_SPMM_CSR_ALG2          HIPSPARSE_SPMM_CSR_ALG2
#define GPUSPARSE_SPMM_CSR_ALG3          HIPSPARSE_SPMM_CSR_ALG3
#define GPUSPARSE_SPMV_ALG_DEFAULT       HIPSPARSE_SPMV_ALG_DEFAULT
#define GPUSPARSE_SPMV_COO_ALG1          HIPSPARSE_SPMV_COO_ALG1
#define GPUSPARSE_SPMV_COO_ALG2          HIPSPARSE_SPMV_COO_ALG2
#define GPUSPARSE_SPMV_CSR_ALG1          HIPSPARSE_SPMV_CSR_ALG1
#define GPUSPARSE_SPMV_CSR_ALG2          HIPSPARSE_SPMV_CSR_ALG2
#define GPUSPARSE_SPSM_ALG_DEFAULT       HIPSPARSE_SPSM_ALG_DEFAULT
#define GPUSPARSE_SPSV_ALG_DEFAULT       HIPSPARSE_SPSV_ALG_DEFAULT
#define GPUSPARSE_STATUS_ALLOC_FAILED    HIPSPARSE_STATUS_ALLOC_FAILED
#define GPUSPARSE_STATUS_ARCH_MISMATCH   HIPSPARSE_STATUS_ARCH_MISMATCH
#define GPUSPARSE_STATUS_EXECUTION_FAILED  \
        HIPSPARSE_STATUS_EXECUTION_FAILED
#define GPUSPARSE_STATUS_INSUFFICIENT_RESOURCES  \
        HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES
#define GPUSPARSE_STATUS_INTERNAL_ERROR  HIPSPARSE_STATUS_INTERNAL_ERROR
#define GPUSPARSE_STATUS_INVALID_VALUE   HIPSPARSE_STATUS_INVALID_VALUE
#define GPUSPARSE_STATUS_MAPPING_ERROR   HIPSPARSE_STATUS_MAPPING_ERROR
#define GPUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED  \
        HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define GPUSPARSE_STATUS_NOT_INITIALIZED HIPSPARSE_STATUS_NOT_INITIALIZED
#define GPUSPARSE_STATUS_NOT_SUPPORTED   HIPSPARSE_STATUS_NOT_SUPPORTED
#define GPUSPARSE_STATUS_SUCCESS         HIPSPARSE_STATUS_SUCCESS
#define GPUSPARSE_STATUS_ZERO_PIVOT      HIPSPARSE_STATUS_ZERO_PIVOT
#define gpusparseAction_t                hipsparseAction_t
#define gpusparseAxpby                   hipsparseAxpby
#define gpusparseBlockedEllGet           hipsparseBlockedEllGet
#define gpusparseCaxpyi                  hipsparseCaxpyi
#define gpusparseCbsr2csr                hipsparseCbsr2csr
#define gpusparseCbsric02                hipsparseCbsric02
#define gpusparseCbsric02_analysis       hipsparseCbsric02_analysis
#define gpusparseCbsric02_bufferSize     hipsparseCbsric02_bufferSize
#define gpusparseCbsrilu02               hipsparseCbsrilu02
#define gpusparseCbsrilu02_analysis      hipsparseCbsrilu02_analysis
#define gpusparseCbsrilu02_bufferSize    hipsparseCbsrilu02_bufferSize
#define gpusparseCbsrilu02_numericBoost  hipsparseCbsrilu02_numericBoost
#define gpusparseCbsrmm                  hipsparseCbsrmm
#define gpusparseCbsrmv                  hipsparseCbsrmv
#define gpusparseCbsrsm2_analysis        hipsparseCbsrsm2_analysis
#define gpusparseCbsrsm2_bufferSize      hipsparseCbsrsm2_bufferSize
#define gpusparseCbsrsm2_solve           hipsparseCbsrsm2_solve
#define gpusparseCbsrsv2_analysis        hipsparseCbsrsv2_analysis
#define gpusparseCbsrsv2_bufferSize      hipsparseCbsrsv2_bufferSize
#define gpusparseCbsrsv2_bufferSizeExt   hipsparseCbsrsv2_bufferSizeExt
#define gpusparseCbsrsv2_solve           hipsparseCbsrsv2_solve
#define gpusparseCbsrxmv                 hipsparseCbsrxmv
#define gpusparseCcsc2dense              hipsparseCcsc2dense
#define gpusparseCcsr2bsr                hipsparseCcsr2bsr
#define gpusparseCcsr2csr_compress       hipsparseCcsr2csr_compress
#define gpusparseCcsr2csru               hipsparseCcsr2csru
#define gpusparseCcsr2dense              hipsparseCcsr2dense
#define gpusparseCcsr2gebsr              hipsparseCcsr2gebsr
#define gpusparseCcsr2gebsr_bufferSize   hipsparseCcsr2gebsr_bufferSize
#define gpusparseCcsrcolor               hipsparseCcsrcolor
#define gpusparseCcsrgeam2               hipsparseCcsrgeam2
#define gpusparseCcsrgeam2_bufferSizeExt hipsparseCcsrgeam2_bufferSizeExt
#define gpusparseCcsrgemm2               hipsparseCcsrgemm2
#define gpusparseCcsrgemm2_bufferSizeExt hipsparseCcsrgemm2_bufferSizeExt
#define gpusparseCcsric02                hipsparseCcsric02
#define gpusparseCcsric02_analysis       hipsparseCcsric02_analysis
#define gpusparseCcsric02_bufferSize     hipsparseCcsric02_bufferSize
#define gpusparseCcsric02_bufferSizeExt  hipsparseCcsric02_bufferSizeExt
#define gpusparseCcsrilu02               hipsparseCcsrilu02
#define gpusparseCcsrilu02_analysis      hipsparseCcsrilu02_analysis
#define gpusparseCcsrilu02_bufferSize    hipsparseCcsrilu02_bufferSize
#define gpusparseCcsrilu02_bufferSizeExt hipsparseCcsrilu02_bufferSizeExt
#define gpusparseCcsrilu02_numericBoost  hipsparseCcsrilu02_numericBoost
#define gpusparseCcsrsm2_analysis        hipsparseCcsrsm2_analysis
#define gpusparseCcsrsm2_bufferSizeExt   hipsparseCcsrsm2_bufferSizeExt
#define gpusparseCcsrsm2_solve           hipsparseCcsrsm2_solve
#define gpusparseCcsrsv2_analysis        hipsparseCcsrsv2_analysis
#define gpusparseCcsrsv2_bufferSize      hipsparseCcsrsv2_bufferSize
#define gpusparseCcsrsv2_bufferSizeExt   hipsparseCcsrsv2_bufferSizeExt
#define gpusparseCcsrsv2_solve           hipsparseCcsrsv2_solve
#define gpusparseCcsru2csr               hipsparseCcsru2csr
#define gpusparseCcsru2csr_bufferSizeExt hipsparseCcsru2csr_bufferSizeExt
#define gpusparseCdense2csc              hipsparseCdense2csc
#define gpusparseCdense2csr              hipsparseCdense2csr
#define gpusparseCgebsr2csr              hipsparseCgebsr2csr
#define gpusparseCgebsr2gebsc            hipsparseCgebsr2gebsc
#define gpusparseCgebsr2gebsc_bufferSize hipsparseCgebsr2gebsc_bufferSize
#define gpusparseCgebsr2gebsr            hipsparseCgebsr2gebsr
#define gpusparseCgebsr2gebsr_bufferSize hipsparseCgebsr2gebsr_bufferSize
#define gpusparseCgemmi                  hipsparseCgemmi
#define gpusparseCgemvi                  hipsparseCgemvi
#define gpusparseCgemvi_bufferSize       hipsparseCgemvi_bufferSize
#define gpusparseCgpsvInterleavedBatch   hipsparseCgpsvInterleavedBatch
#define gpusparseCgpsvInterleavedBatch_bufferSizeExt  \
        hipsparseCgpsvInterleavedBatch_bufferSizeExt
#define gpusparseCgthr                   hipsparseCgthr
#define gpusparseCgthrz                  hipsparseCgthrz
#define gpusparseCgtsv2                  hipsparseCgtsv2
#define gpusparseCgtsv2StridedBatch      hipsparseCgtsv2StridedBatch
#define gpusparseCgtsv2StridedBatch_bufferSizeExt  \
        hipsparseCgtsv2StridedBatch_bufferSizeExt
#define gpusparseCgtsv2_bufferSizeExt    hipsparseCgtsv2_bufferSizeExt
#define gpusparseCgtsv2_nopivot          hipsparseCgtsv2_nopivot
#define gpusparseCgtsv2_nopivot_bufferSizeExt  \
        hipsparseCgtsv2_nopivot_bufferSizeExt
#define gpusparseCgtsvInterleavedBatch   hipsparseCgtsvInterleavedBatch
#define gpusparseCgtsvInterleavedBatch_bufferSizeExt  \
        hipsparseCgtsvInterleavedBatch_bufferSizeExt
#define gpusparseCnnz                    hipsparseCnnz
#define gpusparseCnnz_compress           hipsparseCnnz_compress
#define gpusparseColorInfo_t             hipsparseColorInfo_t
#define gpusparseCooAoSGet               hipsparseCooAoSGet
#define gpusparseCooGet                  hipsparseCooGet
#define gpusparseCooSetPointers          hipsparseCooSetPointers
#define gpusparseCooSetStridedBatch      hipsparseCooSetStridedBatch
#define gpusparseCopyMatDescr            hipsparseCopyMatDescr
#define gpusparseCreate                  hipsparseCreate
#define gpusparseCreateBlockedEll        hipsparseCreateBlockedEll
#define gpusparseCreateBsric02Info       hipsparseCreateBsric02Info
#define gpusparseCreateBsrilu02Info      hipsparseCreateBsrilu02Info
#define gpusparseCreateBsrsm2Info        hipsparseCreateBsrsm2Info
#define gpusparseCreateBsrsv2Info        hipsparseCreateBsrsv2Info
#define gpusparseCreateColorInfo         hipsparseCreateColorInfo
#define gpusparseCreateCoo               hipsparseCreateCoo
#define gpusparseCreateCooAoS            hipsparseCreateCooAoS
#define gpusparseCreateCsc               hipsparseCreateCsc
#define gpusparseCreateCsr               hipsparseCreateCsr
#define gpusparseCreateCsrgemm2Info      hipsparseCreateCsrgemm2Info
#define gpusparseCreateCsric02Info       hipsparseCreateCsric02Info
#define gpusparseCreateCsrilu02Info      hipsparseCreateCsrilu02Info
#define gpusparseCreateCsrsm2Info        hipsparseCreateCsrsm2Info
#define gpusparseCreateCsrsv2Info        hipsparseCreateCsrsv2Info
#define gpusparseCreateCsru2csrInfo      hipsparseCreateCsru2csrInfo
#define gpusparseCreateDnMat             hipsparseCreateDnMat
#define gpusparseCreateDnVec             hipsparseCreateDnVec
#define gpusparseCreateIdentityPermutation  \
        hipsparseCreateIdentityPermutation
#define gpusparseCreateMatDescr          hipsparseCreateMatDescr
#define gpusparseCreatePruneInfo         hipsparseCreatePruneInfo
#define gpusparseCreateSpVec             hipsparseCreateSpVec
#define gpusparseCscSetPointers          hipsparseCscSetPointers
#define gpusparseCsctr                   hipsparseCsctr
#define gpusparseCsr2CscAlg_t            hipsparseCsr2CscAlg_t
#define gpusparseCsr2cscEx2              hipsparseCsr2cscEx2
#define gpusparseCsr2cscEx2_bufferSize   hipsparseCsr2cscEx2_bufferSize
#define gpusparseCsrGet                  hipsparseCsrGet
#define gpusparseCsrSetPointers          hipsparseCsrSetPointers
#define gpusparseCsrSetStridedBatch      hipsparseCsrSetStridedBatch
#define gpusparseDaxpyi                  hipsparseDaxpyi
#define gpusparseDbsr2csr                hipsparseDbsr2csr
#define gpusparseDbsric02                hipsparseDbsric02
#define gpusparseDbsric02_analysis       hipsparseDbsric02_analysis
#define gpusparseDbsric02_bufferSize     hipsparseDbsric02_bufferSize
#define gpusparseDbsrilu02               hipsparseDbsrilu02
#define gpusparseDbsrilu02_analysis      hipsparseDbsrilu02_analysis
#define gpusparseDbsrilu02_bufferSize    hipsparseDbsrilu02_bufferSize
#define gpusparseDbsrilu02_numericBoost  hipsparseDbsrilu02_numericBoost
#define gpusparseDbsrmm                  hipsparseDbsrmm
#define gpusparseDbsrmv                  hipsparseDbsrmv
#define gpusparseDbsrsm2_analysis        hipsparseDbsrsm2_analysis
#define gpusparseDbsrsm2_bufferSize      hipsparseDbsrsm2_bufferSize
#define gpusparseDbsrsm2_solve           hipsparseDbsrsm2_solve
#define gpusparseDbsrsv2_analysis        hipsparseDbsrsv2_analysis
#define gpusparseDbsrsv2_bufferSize      hipsparseDbsrsv2_bufferSize
#define gpusparseDbsrsv2_bufferSizeExt   hipsparseDbsrsv2_bufferSizeExt
#define gpusparseDbsrsv2_solve           hipsparseDbsrsv2_solve
#define gpusparseDbsrxmv                 hipsparseDbsrxmv
#define gpusparseDcsc2dense              hipsparseDcsc2dense
#define gpusparseDcsr2bsr                hipsparseDcsr2bsr
#define gpusparseDcsr2csr_compress       hipsparseDcsr2csr_compress
#define gpusparseDcsr2csru               hipsparseDcsr2csru
#define gpusparseDcsr2dense              hipsparseDcsr2dense
#define gpusparseDcsr2gebsr              hipsparseDcsr2gebsr
#define gpusparseDcsr2gebsr_bufferSize   hipsparseDcsr2gebsr_bufferSize
#define gpusparseDcsrcolor               hipsparseDcsrcolor
#define gpusparseDcsrgeam2               hipsparseDcsrgeam2
#define gpusparseDcsrgeam2_bufferSizeExt hipsparseDcsrgeam2_bufferSizeExt
#define gpusparseDcsrgemm2               hipsparseDcsrgemm2
#define gpusparseDcsrgemm2_bufferSizeExt hipsparseDcsrgemm2_bufferSizeExt
#define gpusparseDcsric02                hipsparseDcsric02
#define gpusparseDcsric02_analysis       hipsparseDcsric02_analysis
#define gpusparseDcsric02_bufferSize     hipsparseDcsric02_bufferSize
#define gpusparseDcsric02_bufferSizeExt  hipsparseDcsric02_bufferSizeExt
#define gpusparseDcsrilu02               hipsparseDcsrilu02
#define gpusparseDcsrilu02_analysis      hipsparseDcsrilu02_analysis
#define gpusparseDcsrilu02_bufferSize    hipsparseDcsrilu02_bufferSize
#define gpusparseDcsrilu02_bufferSizeExt hipsparseDcsrilu02_bufferSizeExt
#define gpusparseDcsrilu02_numericBoost  hipsparseDcsrilu02_numericBoost
#define gpusparseDcsrsm2_analysis        hipsparseDcsrsm2_analysis
#define gpusparseDcsrsm2_bufferSizeExt   hipsparseDcsrsm2_bufferSizeExt
#define gpusparseDcsrsm2_solve           hipsparseDcsrsm2_solve
#define gpusparseDcsrsv2_analysis        hipsparseDcsrsv2_analysis
#define gpusparseDcsrsv2_bufferSize      hipsparseDcsrsv2_bufferSize
#define gpusparseDcsrsv2_bufferSizeExt   hipsparseDcsrsv2_bufferSizeExt
#define gpusparseDcsrsv2_solve           hipsparseDcsrsv2_solve
#define gpusparseDcsru2csr               hipsparseDcsru2csr
#define gpusparseDcsru2csr_bufferSizeExt hipsparseDcsru2csr_bufferSizeExt
#define gpusparseDdense2csc              hipsparseDdense2csc
#define gpusparseDdense2csr              hipsparseDdense2csr
#define gpusparseDenseToSparseAlg_t      hipsparseDenseToSparseAlg_t
#define gpusparseDenseToSparse_analysis  hipsparseDenseToSparse_analysis
#define gpusparseDenseToSparse_bufferSize  \
        hipsparseDenseToSparse_bufferSize
#define gpusparseDenseToSparse_convert   hipsparseDenseToSparse_convert
#define gpusparseDestroy                 hipsparseDestroy
#define gpusparseDestroyBsric02Info      hipsparseDestroyBsric02Info
#define gpusparseDestroyBsrilu02Info     hipsparseDestroyBsrilu02Info
#define gpusparseDestroyBsrsm2Info       hipsparseDestroyBsrsm2Info
#define gpusparseDestroyBsrsv2Info       hipsparseDestroyBsrsv2Info
#define gpusparseDestroyColorInfo        hipsparseDestroyColorInfo
#define gpusparseDestroyCsrgemm2Info     hipsparseDestroyCsrgemm2Info
#define gpusparseDestroyCsric02Info      hipsparseDestroyCsric02Info
#define gpusparseDestroyCsrilu02Info     hipsparseDestroyCsrilu02Info
#define gpusparseDestroyCsrsm2Info       hipsparseDestroyCsrsm2Info
#define gpusparseDestroyCsrsv2Info       hipsparseDestroyCsrsv2Info
#define gpusparseDestroyCsru2csrInfo     hipsparseDestroyCsru2csrInfo
#define gpusparseDestroyDnMat            hipsparseDestroyDnMat
#define gpusparseDestroyDnVec            hipsparseDestroyDnVec
#define gpusparseDestroyMatDescr         hipsparseDestroyMatDescr
#define gpusparseDestroyPruneInfo        hipsparseDestroyPruneInfo
#define gpusparseDestroySpMat            hipsparseDestroySpMat
#define gpusparseDestroySpVec            hipsparseDestroySpVec
#define gpusparseDgebsr2csr              hipsparseDgebsr2csr
#define gpusparseDgebsr2gebsc            hipsparseDgebsr2gebsc
#define gpusparseDgebsr2gebsc_bufferSize hipsparseDgebsr2gebsc_bufferSize
#define gpusparseDgebsr2gebsr            hipsparseDgebsr2gebsr
#define gpusparseDgebsr2gebsr_bufferSize hipsparseDgebsr2gebsr_bufferSize
#define gpusparseDgemmi                  hipsparseDgemmi
#define gpusparseDgemvi                  hipsparseDgemvi
#define gpusparseDgemvi_bufferSize       hipsparseDgemvi_bufferSize
#define gpusparseDgpsvInterleavedBatch   hipsparseDgpsvInterleavedBatch
#define gpusparseDgpsvInterleavedBatch_bufferSizeExt  \
        hipsparseDgpsvInterleavedBatch_bufferSizeExt
#define gpusparseDgthr                   hipsparseDgthr
#define gpusparseDgthrz                  hipsparseDgthrz
#define gpusparseDgtsv2                  hipsparseDgtsv2
#define gpusparseDgtsv2StridedBatch      hipsparseDgtsv2StridedBatch
#define gpusparseDgtsv2StridedBatch_bufferSizeExt  \
        hipsparseDgtsv2StridedBatch_bufferSizeExt
#define gpusparseDgtsv2_bufferSizeExt    hipsparseDgtsv2_bufferSizeExt
#define gpusparseDgtsv2_nopivot          hipsparseDgtsv2_nopivot
#define gpusparseDgtsv2_nopivot_bufferSizeExt  \
        hipsparseDgtsv2_nopivot_bufferSizeExt
#define gpusparseDgtsvInterleavedBatch   hipsparseDgtsvInterleavedBatch
#define gpusparseDgtsvInterleavedBatch_bufferSizeExt  \
        hipsparseDgtsvInterleavedBatch_bufferSizeExt
#define gpusparseDiagType_t              hipsparseDiagType_t
#define gpusparseDirection_t             hipsparseDirection_t
#define gpusparseDnMatDescr_t            hipsparseDnMatDescr_t
#define gpusparseDnMatGet                hipsparseDnMatGet
#define gpusparseDnMatGetStridedBatch    hipsparseDnMatGetStridedBatch
#define gpusparseDnMatGetValues          hipsparseDnMatGetValues
#define gpusparseDnMatSetStridedBatch    hipsparseDnMatSetStridedBatch
#define gpusparseDnMatSetValues          hipsparseDnMatSetValues
#define gpusparseDnVecDescr_t            hipsparseDnVecDescr_t
#define gpusparseDnVecGet                hipsparseDnVecGet
#define gpusparseDnVecGetValues          hipsparseDnVecGetValues
#define gpusparseDnVecSetValues          hipsparseDnVecSetValues
#define gpusparseDnnz                    hipsparseDnnz
#define gpusparseDnnz_compress           hipsparseDnnz_compress
#define gpusparseDpruneCsr2csr           hipsparseDpruneCsr2csr
#define gpusparseDpruneCsr2csrByPercentage  \
        hipsparseDpruneCsr2csrByPercentage
#define gpusparseDpruneCsr2csrByPercentage_bufferSizeExt  \
        hipsparseDpruneCsr2csrByPercentage_bufferSizeExt
#define gpusparseDpruneCsr2csrNnz        hipsparseDpruneCsr2csrNnz
#define gpusparseDpruneCsr2csrNnzByPercentage  \
        hipsparseDpruneCsr2csrNnzByPercentage
#define gpusparseDpruneCsr2csr_bufferSizeExt  \
        hipsparseDpruneCsr2csr_bufferSizeExt
#define gpusparseDpruneDense2csr         hipsparseDpruneDense2csr
#define gpusparseDpruneDense2csrByPercentage  \
        hipsparseDpruneDense2csrByPercentage
#define gpusparseDpruneDense2csrByPercentage_bufferSizeExt  \
        hipsparseDpruneDense2csrByPercentage_bufferSizeExt
#define gpusparseDpruneDense2csrNnz      hipsparseDpruneDense2csrNnz
#define gpusparseDpruneDense2csrNnzByPercentage  \
        hipsparseDpruneDense2csrNnzByPercentage
#define gpusparseDpruneDense2csr_bufferSizeExt  \
        hipsparseDpruneDense2csr_bufferSizeExt
#define gpusparseDroti                   hipsparseDroti
#define gpusparseDsctr                   hipsparseDsctr
#define gpusparseFillMode_t              hipsparseFillMode_t
#define gpusparseFormat_t                hipsparseFormat_t
#define gpusparseGather                  hipsparseGather
#define gpusparseGetMatDiagType          hipsparseGetMatDiagType
#define gpusparseGetMatFillMode          hipsparseGetMatFillMode
#define gpusparseGetMatIndexBase         hipsparseGetMatIndexBase
#define gpusparseGetMatType              hipsparseGetMatType
#define gpusparseGetPointerMode          hipsparseGetPointerMode
#define gpusparseGetStream               hipsparseGetStream
#define gpusparseGetVersion              hipsparseGetVersion
#define gpusparseHandle_t                hipsparseHandle_t
#define gpusparseIndexBase_t             hipsparseIndexBase_t
#define gpusparseIndexType_t             hipsparseIndexType_t
#define gpusparseMatDescr_t              hipsparseMatDescr_t
#define gpusparseMatrixType_t            hipsparseMatrixType_t
#define gpusparseOperation_t             hipsparseOperation_t
#define gpusparseOrder_t                 hipsparseOrder_t
#define gpusparsePointerMode_t           hipsparsePointerMode_t
#define gpusparseRot                     hipsparseRot
#define gpusparseSDDMM                   hipsparseSDDMM
#define gpusparseSDDMMAlg_t              hipsparseSDDMMAlg_t
#define gpusparseSDDMM_bufferSize        hipsparseSDDMM_bufferSize
#define gpusparseSDDMM_preprocess        hipsparseSDDMM_preprocess
#define gpusparseSaxpyi                  hipsparseSaxpyi
#define gpusparseSbsr2csr                hipsparseSbsr2csr
#define gpusparseSbsric02                hipsparseSbsric02
#define gpusparseSbsric02_analysis       hipsparseSbsric02_analysis
#define gpusparseSbsric02_bufferSize     hipsparseSbsric02_bufferSize
#define gpusparseSbsrilu02               hipsparseSbsrilu02
#define gpusparseSbsrilu02_analysis      hipsparseSbsrilu02_analysis
#define gpusparseSbsrilu02_bufferSize    hipsparseSbsrilu02_bufferSize
#define gpusparseSbsrilu02_numericBoost  hipsparseSbsrilu02_numericBoost
#define gpusparseSbsrmm                  hipsparseSbsrmm
#define gpusparseSbsrmv                  hipsparseSbsrmv
#define gpusparseSbsrsm2_analysis        hipsparseSbsrsm2_analysis
#define gpusparseSbsrsm2_bufferSize      hipsparseSbsrsm2_bufferSize
#define gpusparseSbsrsm2_solve           hipsparseSbsrsm2_solve
#define gpusparseSbsrsv2_analysis        hipsparseSbsrsv2_analysis
#define gpusparseSbsrsv2_bufferSize      hipsparseSbsrsv2_bufferSize
#define gpusparseSbsrsv2_bufferSizeExt   hipsparseSbsrsv2_bufferSizeExt
#define gpusparseSbsrsv2_solve           hipsparseSbsrsv2_solve
#define gpusparseSbsrxmv                 hipsparseSbsrxmv
#define gpusparseScatter                 hipsparseScatter
#define gpusparseScsc2dense              hipsparseScsc2dense
#define gpusparseScsr2bsr                hipsparseScsr2bsr
#define gpusparseScsr2csr_compress       hipsparseScsr2csr_compress
#define gpusparseScsr2csru               hipsparseScsr2csru
#define gpusparseScsr2dense              hipsparseScsr2dense
#define gpusparseScsr2gebsr              hipsparseScsr2gebsr
#define gpusparseScsr2gebsr_bufferSize   hipsparseScsr2gebsr_bufferSize
#define gpusparseScsrcolor               hipsparseScsrcolor
#define gpusparseScsrgeam2               hipsparseScsrgeam2
#define gpusparseScsrgeam2_bufferSizeExt hipsparseScsrgeam2_bufferSizeExt
#define gpusparseScsrgemm2               hipsparseScsrgemm2
#define gpusparseScsrgemm2_bufferSizeExt hipsparseScsrgemm2_bufferSizeExt
#define gpusparseScsric02                hipsparseScsric02
#define gpusparseScsric02_analysis       hipsparseScsric02_analysis
#define gpusparseScsric02_bufferSize     hipsparseScsric02_bufferSize
#define gpusparseScsric02_bufferSizeExt  hipsparseScsric02_bufferSizeExt
#define gpusparseScsrilu02               hipsparseScsrilu02
#define gpusparseScsrilu02_analysis      hipsparseScsrilu02_analysis
#define gpusparseScsrilu02_bufferSize    hipsparseScsrilu02_bufferSize
#define gpusparseScsrilu02_bufferSizeExt hipsparseScsrilu02_bufferSizeExt
#define gpusparseScsrilu02_numericBoost  hipsparseScsrilu02_numericBoost
#define gpusparseScsrsm2_analysis        hipsparseScsrsm2_analysis
#define gpusparseScsrsm2_bufferSizeExt   hipsparseScsrsm2_bufferSizeExt
#define gpusparseScsrsm2_solve           hipsparseScsrsm2_solve
#define gpusparseScsrsv2_analysis        hipsparseScsrsv2_analysis
#define gpusparseScsrsv2_bufferSize      hipsparseScsrsv2_bufferSize
#define gpusparseScsrsv2_bufferSizeExt   hipsparseScsrsv2_bufferSizeExt
#define gpusparseScsrsv2_solve           hipsparseScsrsv2_solve
#define gpusparseScsru2csr               hipsparseScsru2csr
#define gpusparseScsru2csr_bufferSizeExt hipsparseScsru2csr_bufferSizeExt
#define gpusparseSdense2csc              hipsparseSdense2csc
#define gpusparseSdense2csr              hipsparseSdense2csr
#define gpusparseSetMatDiagType          hipsparseSetMatDiagType
#define gpusparseSetMatFillMode          hipsparseSetMatFillMode
#define gpusparseSetMatIndexBase         hipsparseSetMatIndexBase
#define gpusparseSetMatType              hipsparseSetMatType
#define gpusparseSetPointerMode          hipsparseSetPointerMode
#define gpusparseSetStream               hipsparseSetStream
#define gpusparseSgebsr2csr              hipsparseSgebsr2csr
#define gpusparseSgebsr2gebsc            hipsparseSgebsr2gebsc
#define gpusparseSgebsr2gebsc_bufferSize hipsparseSgebsr2gebsc_bufferSize
#define gpusparseSgebsr2gebsr            hipsparseSgebsr2gebsr
#define gpusparseSgebsr2gebsr_bufferSize hipsparseSgebsr2gebsr_bufferSize
#define gpusparseSgemmi                  hipsparseSgemmi
#define gpusparseSgemvi                  hipsparseSgemvi
#define gpusparseSgemvi_bufferSize       hipsparseSgemvi_bufferSize
#define gpusparseSgpsvInterleavedBatch   hipsparseSgpsvInterleavedBatch
#define gpusparseSgpsvInterleavedBatch_bufferSizeExt  \
        hipsparseSgpsvInterleavedBatch_bufferSizeExt
#define gpusparseSgthr                   hipsparseSgthr
#define gpusparseSgthrz                  hipsparseSgthrz
#define gpusparseSgtsv2                  hipsparseSgtsv2
#define gpusparseSgtsv2StridedBatch      hipsparseSgtsv2StridedBatch
#define gpusparseSgtsv2StridedBatch_bufferSizeExt  \
        hipsparseSgtsv2StridedBatch_bufferSizeExt
#define gpusparseSgtsv2_bufferSizeExt    hipsparseSgtsv2_bufferSizeExt
#define gpusparseSgtsv2_nopivot          hipsparseSgtsv2_nopivot
#define gpusparseSgtsv2_nopivot_bufferSizeExt  \
        hipsparseSgtsv2_nopivot_bufferSizeExt
#define gpusparseSgtsvInterleavedBatch   hipsparseSgtsvInterleavedBatch
#define gpusparseSgtsvInterleavedBatch_bufferSizeExt  \
        hipsparseSgtsvInterleavedBatch_bufferSizeExt
#define gpusparseSnnz                    hipsparseSnnz
#define gpusparseSnnz_compress           hipsparseSnnz_compress
#define gpusparseSolvePolicy_t           hipsparseSolvePolicy_t
#define gpusparseSpGEMMAlg_t             hipsparseSpGEMMAlg_t
#define gpusparseSpGEMMDescr_t           hipsparseSpGEMMDescr_t
#define gpusparseSpGEMM_compute          hipsparseSpGEMM_compute
#define gpusparseSpGEMM_copy             hipsparseSpGEMM_copy
#define gpusparseSpGEMM_createDescr      hipsparseSpGEMM_createDescr
#define gpusparseSpGEMM_destroyDescr     hipsparseSpGEMM_destroyDescr
#define gpusparseSpGEMM_workEstimation   hipsparseSpGEMM_workEstimation
#define gpusparseSpMM                    hipsparseSpMM
#define gpusparseSpMMAlg_t               hipsparseSpMMAlg_t
#define gpusparseSpMM_bufferSize         hipsparseSpMM_bufferSize
#define gpusparseSpMM_preprocess         hipsparseSpMM_preprocess
#define gpusparseSpMV                    hipsparseSpMV
#define gpusparseSpMVAlg_t               hipsparseSpMVAlg_t
#define gpusparseSpMV_bufferSize         hipsparseSpMV_bufferSize
#define gpusparseSpMatAttribute_t        hipsparseSpMatAttribute_t
#define gpusparseSpMatDescr_t            hipsparseSpMatDescr_t
#define gpusparseSpMatGetAttribute       hipsparseSpMatGetAttribute
#define gpusparseSpMatGetFormat          hipsparseSpMatGetFormat
#define gpusparseSpMatGetIndexBase       hipsparseSpMatGetIndexBase
#define gpusparseSpMatGetSize            hipsparseSpMatGetSize
#define gpusparseSpMatGetStridedBatch    hipsparseSpMatGetStridedBatch
#define gpusparseSpMatGetValues          hipsparseSpMatGetValues
#define gpusparseSpMatSetAttribute       hipsparseSpMatSetAttribute
#define gpusparseSpMatSetStridedBatch    hipsparseSpMatSetStridedBatch
#define gpusparseSpMatSetValues          hipsparseSpMatSetValues
#define gpusparseSpSMAlg_t               hipsparseSpSMAlg_t
#define gpusparseSpSMDescr_t             hipsparseSpSMDescr_t
#define gpusparseSpSM_analysis           hipsparseSpSM_analysis
#define gpusparseSpSM_bufferSize         hipsparseSpSM_bufferSize
#define gpusparseSpSM_createDescr        hipsparseSpSM_createDescr
#define gpusparseSpSM_destroyDescr       hipsparseSpSM_destroyDescr
#define gpusparseSpSM_solve              hipsparseSpSM_solve
#define gpusparseSpSVAlg_t               hipsparseSpSVAlg_t
#define gpusparseSpSVDescr_t             hipsparseSpSVDescr_t
#define gpusparseSpSV_analysis           hipsparseSpSV_analysis
#define gpusparseSpSV_bufferSize         hipsparseSpSV_bufferSize
#define gpusparseSpSV_createDescr        hipsparseSpSV_createDescr
#define gpusparseSpSV_destroyDescr       hipsparseSpSV_destroyDescr
#define gpusparseSpSV_solve              hipsparseSpSV_solve
#define gpusparseSpVV                    hipsparseSpVV
#define gpusparseSpVV_bufferSize         hipsparseSpVV_bufferSize
#define gpusparseSpVecDescr_t            hipsparseSpVecDescr_t
#define gpusparseSpVecGet                hipsparseSpVecGet
#define gpusparseSpVecGetIndexBase       hipsparseSpVecGetIndexBase
#define gpusparseSpVecGetValues          hipsparseSpVecGetValues
#define gpusparseSpVecSetValues          hipsparseSpVecSetValues
#define gpusparseSparseToDense           hipsparseSparseToDense
#define gpusparseSparseToDenseAlg_t      hipsparseSparseToDenseAlg_t
#define gpusparseSparseToDense_bufferSize  \
        hipsparseSparseToDense_bufferSize
#define gpusparseSpruneCsr2csr           hipsparseSpruneCsr2csr
#define gpusparseSpruneCsr2csrByPercentage  \
        hipsparseSpruneCsr2csrByPercentage
#define gpusparseSpruneCsr2csrByPercentage_bufferSizeExt  \
        hipsparseSpruneCsr2csrByPercentage_bufferSizeExt
#define gpusparseSpruneCsr2csrNnz        hipsparseSpruneCsr2csrNnz
#define gpusparseSpruneCsr2csrNnzByPercentage  \
        hipsparseSpruneCsr2csrNnzByPercentage
#define gpusparseSpruneCsr2csr_bufferSizeExt  \
        hipsparseSpruneCsr2csr_bufferSizeExt
#define gpusparseSpruneDense2csr         hipsparseSpruneDense2csr
#define gpusparseSpruneDense2csrByPercentage  \
        hipsparseSpruneDense2csrByPercentage
#define gpusparseSpruneDense2csrByPercentage_bufferSizeExt  \
        hipsparseSpruneDense2csrByPercentage_bufferSizeExt
#define gpusparseSpruneDense2csrNnz      hipsparseSpruneDense2csrNnz
#define gpusparseSpruneDense2csrNnzByPercentage  \
        hipsparseSpruneDense2csrNnzByPercentage
#define gpusparseSpruneDense2csr_bufferSizeExt  \
        hipsparseSpruneDense2csr_bufferSizeExt
#define gpusparseSroti                   hipsparseSroti
#define gpusparseSsctr                   hipsparseSsctr
#define gpusparseStatus_t                hipsparseStatus_t
#define gpusparseXbsric02_zeroPivot      hipsparseXbsric02_zeroPivot
#define gpusparseXbsrilu02_zeroPivot     hipsparseXbsrilu02_zeroPivot
#define gpusparseXbsrsm2_zeroPivot       hipsparseXbsrsm2_zeroPivot
#define gpusparseXbsrsv2_zeroPivot       hipsparseXbsrsv2_zeroPivot
#define gpusparseXcoo2csr                hipsparseXcoo2csr
#define gpusparseXcoosortByColumn        hipsparseXcoosortByColumn
#define gpusparseXcoosortByRow           hipsparseXcoosortByRow
#define gpusparseXcoosort_bufferSizeExt  hipsparseXcoosort_bufferSizeExt
#define gpusparseXcscsort                hipsparseXcscsort
#define gpusparseXcscsort_bufferSizeExt  hipsparseXcscsort_bufferSizeExt
#define gpusparseXcsr2bsrNnz             hipsparseXcsr2bsrNnz
#define gpusparseXcsr2coo                hipsparseXcsr2coo
#define gpusparseXcsr2gebsrNnz           hipsparseXcsr2gebsrNnz
#define gpusparseXcsrgeam2Nnz            hipsparseXcsrgeam2Nnz
#define gpusparseXcsrgemm2Nnz            hipsparseXcsrgemm2Nnz
#define gpusparseXcsric02_zeroPivot      hipsparseXcsric02_zeroPivot
#define gpusparseXcsrilu02_zeroPivot     hipsparseXcsrilu02_zeroPivot
#define gpusparseXcsrsm2_zeroPivot       hipsparseXcsrsm2_zeroPivot
#define gpusparseXcsrsort                hipsparseXcsrsort
#define gpusparseXcsrsort_bufferSizeExt  hipsparseXcsrsort_bufferSizeExt
#define gpusparseXcsrsv2_zeroPivot       hipsparseXcsrsv2_zeroPivot
#define gpusparseXgebsr2gebsrNnz         hipsparseXgebsr2gebsrNnz
#define gpusparseZaxpyi                  hipsparseZaxpyi
#define gpusparseZbsr2csr                hipsparseZbsr2csr
#define gpusparseZbsric02                hipsparseZbsric02
#define gpusparseZbsric02_analysis       hipsparseZbsric02_analysis
#define gpusparseZbsric02_bufferSize     hipsparseZbsric02_bufferSize
#define gpusparseZbsrilu02               hipsparseZbsrilu02
#define gpusparseZbsrilu02_analysis      hipsparseZbsrilu02_analysis
#define gpusparseZbsrilu02_bufferSize    hipsparseZbsrilu02_bufferSize
#define gpusparseZbsrilu02_numericBoost  hipsparseZbsrilu02_numericBoost
#define gpusparseZbsrmm                  hipsparseZbsrmm
#define gpusparseZbsrmv                  hipsparseZbsrmv
#define gpusparseZbsrsm2_analysis        hipsparseZbsrsm2_analysis
#define gpusparseZbsrsm2_bufferSize      hipsparseZbsrsm2_bufferSize
#define gpusparseZbsrsm2_solve           hipsparseZbsrsm2_solve
#define gpusparseZbsrsv2_analysis        hipsparseZbsrsv2_analysis
#define gpusparseZbsrsv2_bufferSize      hipsparseZbsrsv2_bufferSize
#define gpusparseZbsrsv2_bufferSizeExt   hipsparseZbsrsv2_bufferSizeExt
#define gpusparseZbsrsv2_solve           hipsparseZbsrsv2_solve
#define gpusparseZbsrxmv                 hipsparseZbsrxmv
#define gpusparseZcsc2dense              hipsparseZcsc2dense
#define gpusparseZcsr2bsr                hipsparseZcsr2bsr
#define gpusparseZcsr2csr_compress       hipsparseZcsr2csr_compress
#define gpusparseZcsr2csru               hipsparseZcsr2csru
#define gpusparseZcsr2dense              hipsparseZcsr2dense
#define gpusparseZcsr2gebsr              hipsparseZcsr2gebsr
#define gpusparseZcsr2gebsr_bufferSize   hipsparseZcsr2gebsr_bufferSize
#define gpusparseZcsrcolor               hipsparseZcsrcolor
#define gpusparseZcsrgeam2               hipsparseZcsrgeam2
#define gpusparseZcsrgeam2_bufferSizeExt hipsparseZcsrgeam2_bufferSizeExt
#define gpusparseZcsrgemm2               hipsparseZcsrgemm2
#define gpusparseZcsrgemm2_bufferSizeExt hipsparseZcsrgemm2_bufferSizeExt
#define gpusparseZcsric02                hipsparseZcsric02
#define gpusparseZcsric02_analysis       hipsparseZcsric02_analysis
#define gpusparseZcsric02_bufferSize     hipsparseZcsric02_bufferSize
#define gpusparseZcsric02_bufferSizeExt  hipsparseZcsric02_bufferSizeExt
#define gpusparseZcsrilu02               hipsparseZcsrilu02
#define gpusparseZcsrilu02_analysis      hipsparseZcsrilu02_analysis
#define gpusparseZcsrilu02_bufferSize    hipsparseZcsrilu02_bufferSize
#define gpusparseZcsrilu02_bufferSizeExt hipsparseZcsrilu02_bufferSizeExt
#define gpusparseZcsrilu02_numericBoost  hipsparseZcsrilu02_numericBoost
#define gpusparseZcsrsm2_analysis        hipsparseZcsrsm2_analysis
#define gpusparseZcsrsm2_bufferSizeExt   hipsparseZcsrsm2_bufferSizeExt
#define gpusparseZcsrsm2_solve           hipsparseZcsrsm2_solve
#define gpusparseZcsrsv2_analysis        hipsparseZcsrsv2_analysis
#define gpusparseZcsrsv2_bufferSize      hipsparseZcsrsv2_bufferSize
#define gpusparseZcsrsv2_bufferSizeExt   hipsparseZcsrsv2_bufferSizeExt
#define gpusparseZcsrsv2_solve           hipsparseZcsrsv2_solve
#define gpusparseZcsru2csr               hipsparseZcsru2csr
#define gpusparseZcsru2csr_bufferSizeExt hipsparseZcsru2csr_bufferSizeExt
#define gpusparseZdense2csc              hipsparseZdense2csc
#define gpusparseZdense2csr              hipsparseZdense2csr
#define gpusparseZgebsr2csr              hipsparseZgebsr2csr
#define gpusparseZgebsr2gebsc            hipsparseZgebsr2gebsc
#define gpusparseZgebsr2gebsc_bufferSize hipsparseZgebsr2gebsc_bufferSize
#define gpusparseZgebsr2gebsr            hipsparseZgebsr2gebsr
#define gpusparseZgebsr2gebsr_bufferSize hipsparseZgebsr2gebsr_bufferSize
#define gpusparseZgemmi                  hipsparseZgemmi
#define gpusparseZgemvi                  hipsparseZgemvi
#define gpusparseZgemvi_bufferSize       hipsparseZgemvi_bufferSize
#define gpusparseZgpsvInterleavedBatch   hipsparseZgpsvInterleavedBatch
#define gpusparseZgpsvInterleavedBatch_bufferSizeExt  \
        hipsparseZgpsvInterleavedBatch_bufferSizeExt
#define gpusparseZgthr                   hipsparseZgthr
#define gpusparseZgthrz                  hipsparseZgthrz
#define gpusparseZgtsv2                  hipsparseZgtsv2
#define gpusparseZgtsv2StridedBatch      hipsparseZgtsv2StridedBatch
#define gpusparseZgtsv2StridedBatch_bufferSizeExt  \
        hipsparseZgtsv2StridedBatch_bufferSizeExt
#define gpusparseZgtsv2_bufferSizeExt    hipsparseZgtsv2_bufferSizeExt
#define gpusparseZgtsv2_nopivot          hipsparseZgtsv2_nopivot
#define gpusparseZgtsv2_nopivot_bufferSizeExt  \
        hipsparseZgtsv2_nopivot_bufferSizeExt
#define gpusparseZgtsvInterleavedBatch   hipsparseZgtsvInterleavedBatch
#define gpusparseZgtsvInterleavedBatch_bufferSizeExt  \
        hipsparseZgtsvInterleavedBatch_bufferSizeExt
#define gpusparseZnnz                    hipsparseZnnz
#define gpusparseZnnz_compress           hipsparseZnnz_compress
#define gpusparseZsctr                   hipsparseZsctr


#endif
