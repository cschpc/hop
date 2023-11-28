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

#ifndef __HOP_HOPSPARSE_CUDA_H__
#define __HOP_HOPSPARSE_CUDA_H__

#if defined(HOP_OVERRIDE_CUSPARSE_V1)
#include <cusparse.h>
#else
#include <cusparse_v2.h>
#endif

#define GPUSPARSE_ACTION_NUMERIC         CUSPARSE_ACTION_NUMERIC
#define GPUSPARSE_ACTION_SYMBOLIC        CUSPARSE_ACTION_SYMBOLIC
#define GPUSPARSE_COOMM_ALG1             CUSPARSE_COOMM_ALG1
#define GPUSPARSE_COOMM_ALG2             CUSPARSE_COOMM_ALG2
#define GPUSPARSE_COOMM_ALG3             CUSPARSE_COOMM_ALG3
#define GPUSPARSE_COOMV_ALG              CUSPARSE_COOMV_ALG
#define GPUSPARSE_CSR2CSC_ALG1           CUSPARSE_CSR2CSC_ALG1
#define GPUSPARSE_CSR2CSC_ALG2           CUSPARSE_CSR2CSC_ALG2
#define GPUSPARSE_CSR2CSC_ALG_DEFAULT    CUSPARSE_CSR2CSC_ALG_DEFAULT
#define GPUSPARSE_CSRMM_ALG1             CUSPARSE_CSRMM_ALG1
#define GPUSPARSE_CSRMV_ALG1             CUSPARSE_CSRMV_ALG1
#define GPUSPARSE_CSRMV_ALG2             CUSPARSE_CSRMV_ALG2
#define GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT  \
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT
#define GPUSPARSE_DIAG_TYPE_NON_UNIT     CUSPARSE_DIAG_TYPE_NON_UNIT
#define GPUSPARSE_DIAG_TYPE_UNIT         CUSPARSE_DIAG_TYPE_UNIT
#define GPUSPARSE_DIRECTION_COLUMN       CUSPARSE_DIRECTION_COLUMN
#define GPUSPARSE_DIRECTION_ROW          CUSPARSE_DIRECTION_ROW
#define GPUSPARSE_FILL_MODE_LOWER        CUSPARSE_FILL_MODE_LOWER
#define GPUSPARSE_FILL_MODE_UPPER        CUSPARSE_FILL_MODE_UPPER
#define GPUSPARSE_FORMAT_BLOCKED_ELL     CUSPARSE_FORMAT_BLOCKED_ELL
#define GPUSPARSE_FORMAT_COO             CUSPARSE_FORMAT_COO
#define GPUSPARSE_FORMAT_COO_AOS         CUSPARSE_FORMAT_COO_AOS
#define GPUSPARSE_FORMAT_CSC             CUSPARSE_FORMAT_CSC
#define GPUSPARSE_FORMAT_CSR             CUSPARSE_FORMAT_CSR
#define GPUSPARSE_INDEX_16U              CUSPARSE_INDEX_16U
#define GPUSPARSE_INDEX_32I              CUSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_64I              CUSPARSE_INDEX_64I
#define GPUSPARSE_INDEX_BASE_ONE         CUSPARSE_INDEX_BASE_ONE
#define GPUSPARSE_INDEX_BASE_ZERO        CUSPARSE_INDEX_BASE_ZERO
#define GPUSPARSE_MATRIX_TYPE_GENERAL    CUSPARSE_MATRIX_TYPE_GENERAL
#define GPUSPARSE_MATRIX_TYPE_HERMITIAN  CUSPARSE_MATRIX_TYPE_HERMITIAN
#define GPUSPARSE_MATRIX_TYPE_SYMMETRIC  CUSPARSE_MATRIX_TYPE_SYMMETRIC
#define GPUSPARSE_MATRIX_TYPE_TRIANGULAR CUSPARSE_MATRIX_TYPE_TRIANGULAR
#define GPUSPARSE_MM_ALG_DEFAULT         CUSPARSE_MM_ALG_DEFAULT
#define GPUSPARSE_MV_ALG_DEFAULT         CUSPARSE_MV_ALG_DEFAULT
#define GPUSPARSE_OPERATION_CONJUGATE_TRANSPOSE  \
        CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
#define GPUSPARSE_OPERATION_NON_TRANSPOSE  \
        CUSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_OPERATION_TRANSPOSE    CUSPARSE_OPERATION_TRANSPOSE
#define GPUSPARSE_ORDER_COL              CUSPARSE_ORDER_COL
#define GPUSPARSE_ORDER_ROW              CUSPARSE_ORDER_ROW
#define GPUSPARSE_POINTER_MODE_DEVICE    CUSPARSE_POINTER_MODE_DEVICE
#define GPUSPARSE_POINTER_MODE_HOST      CUSPARSE_POINTER_MODE_HOST
#define GPUSPARSE_SDDMM_ALG_DEFAULT      CUSPARSE_SDDMM_ALG_DEFAULT
#define GPUSPARSE_SOLVE_POLICY_NO_LEVEL  CUSPARSE_SOLVE_POLICY_NO_LEVEL
#define GPUSPARSE_SOLVE_POLICY_USE_LEVEL CUSPARSE_SOLVE_POLICY_USE_LEVEL
#define GPUSPARSE_SPARSETODENSE_ALG_DEFAULT  \
        CUSPARSE_SPARSETODENSE_ALG_DEFAULT
#define GPUSPARSE_SPGEMM_ALG1            CUSPARSE_SPGEMM_ALG1
#define GPUSPARSE_SPGEMM_ALG2            CUSPARSE_SPGEMM_ALG2
#define GPUSPARSE_SPGEMM_ALG3            CUSPARSE_SPGEMM_ALG3
#define GPUSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC  \
        CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC
#define GPUSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC  \
        CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC
#define GPUSPARSE_SPGEMM_DEFAULT         CUSPARSE_SPGEMM_DEFAULT
#define GPUSPARSE_SPMAT_DIAG_TYPE        CUSPARSE_SPMAT_DIAG_TYPE
#define GPUSPARSE_SPMAT_FILL_MODE        CUSPARSE_SPMAT_FILL_MODE
#define GPUSPARSE_SPMM_ALG_DEFAULT       CUSPARSE_SPMM_ALG_DEFAULT
#define GPUSPARSE_SPMM_BLOCKED_ELL_ALG1  CUSPARSE_SPMM_BLOCKED_ELL_ALG1
#define GPUSPARSE_SPMM_COO_ALG1          CUSPARSE_SPMM_COO_ALG1
#define GPUSPARSE_SPMM_COO_ALG2          CUSPARSE_SPMM_COO_ALG2
#define GPUSPARSE_SPMM_COO_ALG3          CUSPARSE_SPMM_COO_ALG3
#define GPUSPARSE_SPMM_COO_ALG4          CUSPARSE_SPMM_COO_ALG4
#define GPUSPARSE_SPMM_CSR_ALG1          CUSPARSE_SPMM_CSR_ALG1
#define GPUSPARSE_SPMM_CSR_ALG2          CUSPARSE_SPMM_CSR_ALG2
#define GPUSPARSE_SPMM_CSR_ALG3          CUSPARSE_SPMM_CSR_ALG3
#define GPUSPARSE_SPMV_ALG_DEFAULT       CUSPARSE_SPMV_ALG_DEFAULT
#define GPUSPARSE_SPMV_COO_ALG1          CUSPARSE_SPMV_COO_ALG1
#define GPUSPARSE_SPMV_COO_ALG2          CUSPARSE_SPMV_COO_ALG2
#define GPUSPARSE_SPMV_CSR_ALG1          CUSPARSE_SPMV_CSR_ALG1
#define GPUSPARSE_SPMV_CSR_ALG2          CUSPARSE_SPMV_CSR_ALG2
#define GPUSPARSE_SPSM_ALG_DEFAULT       CUSPARSE_SPSM_ALG_DEFAULT
#define GPUSPARSE_SPSV_ALG_DEFAULT       CUSPARSE_SPSV_ALG_DEFAULT
#define GPUSPARSE_STATUS_ALLOC_FAILED    CUSPARSE_STATUS_ALLOC_FAILED
#define GPUSPARSE_STATUS_ARCH_MISMATCH   CUSPARSE_STATUS_ARCH_MISMATCH
#define GPUSPARSE_STATUS_EXECUTION_FAILED  \
        CUSPARSE_STATUS_EXECUTION_FAILED
#define GPUSPARSE_STATUS_INSUFFICIENT_RESOURCES  \
        CUSPARSE_STATUS_INSUFFICIENT_RESOURCES
#define GPUSPARSE_STATUS_INTERNAL_ERROR  CUSPARSE_STATUS_INTERNAL_ERROR
#define GPUSPARSE_STATUS_INVALID_VALUE   CUSPARSE_STATUS_INVALID_VALUE
#define GPUSPARSE_STATUS_MAPPING_ERROR   CUSPARSE_STATUS_MAPPING_ERROR
#define GPUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED  \
        CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define GPUSPARSE_STATUS_NOT_INITIALIZED CUSPARSE_STATUS_NOT_INITIALIZED
#define GPUSPARSE_STATUS_NOT_SUPPORTED   CUSPARSE_STATUS_NOT_SUPPORTED
#define GPUSPARSE_STATUS_SUCCESS         CUSPARSE_STATUS_SUCCESS
#define GPUSPARSE_STATUS_ZERO_PIVOT      CUSPARSE_STATUS_ZERO_PIVOT
#define gpusparseAction_t                cusparseAction_t
#define gpusparseAxpby                   cusparseAxpby
#define gpusparseBlockedEllGet           cusparseBlockedEllGet
#define gpusparseCaxpyi                  cusparseCaxpyi
#define gpusparseCbsr2csr                cusparseCbsr2csr
#define gpusparseCbsric02                cusparseCbsric02
#define gpusparseCbsric02_analysis       cusparseCbsric02_analysis
#define gpusparseCbsric02_bufferSize     cusparseCbsric02_bufferSize
#define gpusparseCbsrilu02               cusparseCbsrilu02
#define gpusparseCbsrilu02_analysis      cusparseCbsrilu02_analysis
#define gpusparseCbsrilu02_bufferSize    cusparseCbsrilu02_bufferSize
#define gpusparseCbsrilu02_numericBoost  cusparseCbsrilu02_numericBoost
#define gpusparseCbsrmm                  cusparseCbsrmm
#define gpusparseCbsrmv                  cusparseCbsrmv
#define gpusparseCbsrsm2_analysis        cusparseCbsrsm2_analysis
#define gpusparseCbsrsm2_bufferSize      cusparseCbsrsm2_bufferSize
#define gpusparseCbsrsm2_solve           cusparseCbsrsm2_solve
#define gpusparseCbsrsv2_analysis        cusparseCbsrsv2_analysis
#define gpusparseCbsrsv2_bufferSize      cusparseCbsrsv2_bufferSize
#define gpusparseCbsrsv2_bufferSizeExt   cusparseCbsrsv2_bufferSizeExt
#define gpusparseCbsrsv2_solve           cusparseCbsrsv2_solve
#define gpusparseCbsrxmv                 cusparseCbsrxmv
#define gpusparseCcsc2dense              cusparseCcsc2dense
#define gpusparseCcsr2bsr                cusparseCcsr2bsr
#define gpusparseCcsr2csr_compress       cusparseCcsr2csr_compress
#define gpusparseCcsr2csru               cusparseCcsr2csru
#define gpusparseCcsr2dense              cusparseCcsr2dense
#define gpusparseCcsr2gebsr              cusparseCcsr2gebsr
#define gpusparseCcsr2gebsr_bufferSize   cusparseCcsr2gebsr_bufferSize
#define gpusparseCcsrcolor               cusparseCcsrcolor
#define gpusparseCcsrgeam2               cusparseCcsrgeam2
#define gpusparseCcsrgeam2_bufferSizeExt cusparseCcsrgeam2_bufferSizeExt
#define gpusparseCcsrgemm2               cusparseCcsrgemm2
#define gpusparseCcsrgemm2_bufferSizeExt cusparseCcsrgemm2_bufferSizeExt
#define gpusparseCcsric02                cusparseCcsric02
#define gpusparseCcsric02_analysis       cusparseCcsric02_analysis
#define gpusparseCcsric02_bufferSize     cusparseCcsric02_bufferSize
#define gpusparseCcsric02_bufferSizeExt  cusparseCcsric02_bufferSizeExt
#define gpusparseCcsrilu02               cusparseCcsrilu02
#define gpusparseCcsrilu02_analysis      cusparseCcsrilu02_analysis
#define gpusparseCcsrilu02_bufferSize    cusparseCcsrilu02_bufferSize
#define gpusparseCcsrilu02_bufferSizeExt cusparseCcsrilu02_bufferSizeExt
#define gpusparseCcsrilu02_numericBoost  cusparseCcsrilu02_numericBoost
#define gpusparseCcsrsm2_analysis        cusparseCcsrsm2_analysis
#define gpusparseCcsrsm2_bufferSizeExt   cusparseCcsrsm2_bufferSizeExt
#define gpusparseCcsrsm2_solve           cusparseCcsrsm2_solve
#define gpusparseCcsrsv2_analysis        cusparseCcsrsv2_analysis
#define gpusparseCcsrsv2_bufferSize      cusparseCcsrsv2_bufferSize
#define gpusparseCcsrsv2_bufferSizeExt   cusparseCcsrsv2_bufferSizeExt
#define gpusparseCcsrsv2_solve           cusparseCcsrsv2_solve
#define gpusparseCcsru2csr               cusparseCcsru2csr
#define gpusparseCcsru2csr_bufferSizeExt cusparseCcsru2csr_bufferSizeExt
#define gpusparseCdense2csc              cusparseCdense2csc
#define gpusparseCdense2csr              cusparseCdense2csr
#define gpusparseCgebsr2csr              cusparseCgebsr2csr
#define gpusparseCgebsr2gebsc            cusparseCgebsr2gebsc
#define gpusparseCgebsr2gebsc_bufferSize cusparseCgebsr2gebsc_bufferSize
#define gpusparseCgebsr2gebsr            cusparseCgebsr2gebsr
#define gpusparseCgebsr2gebsr_bufferSize cusparseCgebsr2gebsr_bufferSize
#define gpusparseCgemmi                  cusparseCgemmi
#define gpusparseCgemvi                  cusparseCgemvi
#define gpusparseCgemvi_bufferSize       cusparseCgemvi_bufferSize
#define gpusparseCgpsvInterleavedBatch   cusparseCgpsvInterleavedBatch
#define gpusparseCgpsvInterleavedBatch_bufferSizeExt  \
        cusparseCgpsvInterleavedBatch_bufferSizeExt
#define gpusparseCgthr                   cusparseCgthr
#define gpusparseCgthrz                  cusparseCgthrz
#define gpusparseCgtsv2                  cusparseCgtsv2
#define gpusparseCgtsv2StridedBatch      cusparseCgtsv2StridedBatch
#define gpusparseCgtsv2StridedBatch_bufferSizeExt  \
        cusparseCgtsv2StridedBatch_bufferSizeExt
#define gpusparseCgtsv2_bufferSizeExt    cusparseCgtsv2_bufferSizeExt
#define gpusparseCgtsv2_nopivot          cusparseCgtsv2_nopivot
#define gpusparseCgtsv2_nopivot_bufferSizeExt  \
        cusparseCgtsv2_nopivot_bufferSizeExt
#define gpusparseCgtsvInterleavedBatch   cusparseCgtsvInterleavedBatch
#define gpusparseCgtsvInterleavedBatch_bufferSizeExt  \
        cusparseCgtsvInterleavedBatch_bufferSizeExt
#define gpusparseCnnz                    cusparseCnnz
#define gpusparseCnnz_compress           cusparseCnnz_compress
#define gpusparseColorInfo_t             cusparseColorInfo_t
#define gpusparseCooAoSGet               cusparseCooAoSGet
#define gpusparseCooGet                  cusparseCooGet
#define gpusparseCooSetPointers          cusparseCooSetPointers
#define gpusparseCooSetStridedBatch      cusparseCooSetStridedBatch
#define gpusparseCopyMatDescr            cusparseCopyMatDescr
#define gpusparseCreate                  cusparseCreate
#define gpusparseCreateBlockedEll        cusparseCreateBlockedEll
#define gpusparseCreateBsric02Info       cusparseCreateBsric02Info
#define gpusparseCreateBsrilu02Info      cusparseCreateBsrilu02Info
#define gpusparseCreateBsrsm2Info        cusparseCreateBsrsm2Info
#define gpusparseCreateBsrsv2Info        cusparseCreateBsrsv2Info
#define gpusparseCreateColorInfo         cusparseCreateColorInfo
#define gpusparseCreateCoo               cusparseCreateCoo
#define gpusparseCreateCooAoS            cusparseCreateCooAoS
#define gpusparseCreateCsc               cusparseCreateCsc
#define gpusparseCreateCsr               cusparseCreateCsr
#define gpusparseCreateCsrgemm2Info      cusparseCreateCsrgemm2Info
#define gpusparseCreateCsric02Info       cusparseCreateCsric02Info
#define gpusparseCreateCsrilu02Info      cusparseCreateCsrilu02Info
#define gpusparseCreateCsrsm2Info        cusparseCreateCsrsm2Info
#define gpusparseCreateCsrsv2Info        cusparseCreateCsrsv2Info
#define gpusparseCreateCsru2csrInfo      cusparseCreateCsru2csrInfo
#define gpusparseCreateDnMat             cusparseCreateDnMat
#define gpusparseCreateDnVec             cusparseCreateDnVec
#define gpusparseCreateIdentityPermutation  \
        cusparseCreateIdentityPermutation
#define gpusparseCreateMatDescr          cusparseCreateMatDescr
#define gpusparseCreatePruneInfo         cusparseCreatePruneInfo
#define gpusparseCreateSpVec             cusparseCreateSpVec
#define gpusparseCscSetPointers          cusparseCscSetPointers
#define gpusparseCsctr                   cusparseCsctr
#define gpusparseCsr2CscAlg_t            cusparseCsr2CscAlg_t
#define gpusparseCsr2cscEx2              cusparseCsr2cscEx2
#define gpusparseCsr2cscEx2_bufferSize   cusparseCsr2cscEx2_bufferSize
#define gpusparseCsrGet                  cusparseCsrGet
#define gpusparseCsrSetPointers          cusparseCsrSetPointers
#define gpusparseCsrSetStridedBatch      cusparseCsrSetStridedBatch
#define gpusparseDaxpyi                  cusparseDaxpyi
#define gpusparseDbsr2csr                cusparseDbsr2csr
#define gpusparseDbsric02                cusparseDbsric02
#define gpusparseDbsric02_analysis       cusparseDbsric02_analysis
#define gpusparseDbsric02_bufferSize     cusparseDbsric02_bufferSize
#define gpusparseDbsrilu02               cusparseDbsrilu02
#define gpusparseDbsrilu02_analysis      cusparseDbsrilu02_analysis
#define gpusparseDbsrilu02_bufferSize    cusparseDbsrilu02_bufferSize
#define gpusparseDbsrilu02_numericBoost  cusparseDbsrilu02_numericBoost
#define gpusparseDbsrmm                  cusparseDbsrmm
#define gpusparseDbsrmv                  cusparseDbsrmv
#define gpusparseDbsrsm2_analysis        cusparseDbsrsm2_analysis
#define gpusparseDbsrsm2_bufferSize      cusparseDbsrsm2_bufferSize
#define gpusparseDbsrsm2_solve           cusparseDbsrsm2_solve
#define gpusparseDbsrsv2_analysis        cusparseDbsrsv2_analysis
#define gpusparseDbsrsv2_bufferSize      cusparseDbsrsv2_bufferSize
#define gpusparseDbsrsv2_bufferSizeExt   cusparseDbsrsv2_bufferSizeExt
#define gpusparseDbsrsv2_solve           cusparseDbsrsv2_solve
#define gpusparseDbsrxmv                 cusparseDbsrxmv
#define gpusparseDcsc2dense              cusparseDcsc2dense
#define gpusparseDcsr2bsr                cusparseDcsr2bsr
#define gpusparseDcsr2csr_compress       cusparseDcsr2csr_compress
#define gpusparseDcsr2csru               cusparseDcsr2csru
#define gpusparseDcsr2dense              cusparseDcsr2dense
#define gpusparseDcsr2gebsr              cusparseDcsr2gebsr
#define gpusparseDcsr2gebsr_bufferSize   cusparseDcsr2gebsr_bufferSize
#define gpusparseDcsrcolor               cusparseDcsrcolor
#define gpusparseDcsrgeam2               cusparseDcsrgeam2
#define gpusparseDcsrgeam2_bufferSizeExt cusparseDcsrgeam2_bufferSizeExt
#define gpusparseDcsrgemm2               cusparseDcsrgemm2
#define gpusparseDcsrgemm2_bufferSizeExt cusparseDcsrgemm2_bufferSizeExt
#define gpusparseDcsric02                cusparseDcsric02
#define gpusparseDcsric02_analysis       cusparseDcsric02_analysis
#define gpusparseDcsric02_bufferSize     cusparseDcsric02_bufferSize
#define gpusparseDcsric02_bufferSizeExt  cusparseDcsric02_bufferSizeExt
#define gpusparseDcsrilu02               cusparseDcsrilu02
#define gpusparseDcsrilu02_analysis      cusparseDcsrilu02_analysis
#define gpusparseDcsrilu02_bufferSize    cusparseDcsrilu02_bufferSize
#define gpusparseDcsrilu02_bufferSizeExt cusparseDcsrilu02_bufferSizeExt
#define gpusparseDcsrilu02_numericBoost  cusparseDcsrilu02_numericBoost
#define gpusparseDcsrsm2_analysis        cusparseDcsrsm2_analysis
#define gpusparseDcsrsm2_bufferSizeExt   cusparseDcsrsm2_bufferSizeExt
#define gpusparseDcsrsm2_solve           cusparseDcsrsm2_solve
#define gpusparseDcsrsv2_analysis        cusparseDcsrsv2_analysis
#define gpusparseDcsrsv2_bufferSize      cusparseDcsrsv2_bufferSize
#define gpusparseDcsrsv2_bufferSizeExt   cusparseDcsrsv2_bufferSizeExt
#define gpusparseDcsrsv2_solve           cusparseDcsrsv2_solve
#define gpusparseDcsru2csr               cusparseDcsru2csr
#define gpusparseDcsru2csr_bufferSizeExt cusparseDcsru2csr_bufferSizeExt
#define gpusparseDdense2csc              cusparseDdense2csc
#define gpusparseDdense2csr              cusparseDdense2csr
#define gpusparseDenseToSparseAlg_t      cusparseDenseToSparseAlg_t
#define gpusparseDenseToSparse_analysis  cusparseDenseToSparse_analysis
#define gpusparseDenseToSparse_bufferSize  \
        cusparseDenseToSparse_bufferSize
#define gpusparseDenseToSparse_convert   cusparseDenseToSparse_convert
#define gpusparseDestroy                 cusparseDestroy
#define gpusparseDestroyBsric02Info      cusparseDestroyBsric02Info
#define gpusparseDestroyBsrilu02Info     cusparseDestroyBsrilu02Info
#define gpusparseDestroyBsrsm2Info       cusparseDestroyBsrsm2Info
#define gpusparseDestroyBsrsv2Info       cusparseDestroyBsrsv2Info
#define gpusparseDestroyColorInfo        cusparseDestroyColorInfo
#define gpusparseDestroyCsrgemm2Info     cusparseDestroyCsrgemm2Info
#define gpusparseDestroyCsric02Info      cusparseDestroyCsric02Info
#define gpusparseDestroyCsrilu02Info     cusparseDestroyCsrilu02Info
#define gpusparseDestroyCsrsm2Info       cusparseDestroyCsrsm2Info
#define gpusparseDestroyCsrsv2Info       cusparseDestroyCsrsv2Info
#define gpusparseDestroyCsru2csrInfo     cusparseDestroyCsru2csrInfo
#define gpusparseDestroyDnMat            cusparseDestroyDnMat
#define gpusparseDestroyDnVec            cusparseDestroyDnVec
#define gpusparseDestroyMatDescr         cusparseDestroyMatDescr
#define gpusparseDestroyPruneInfo        cusparseDestroyPruneInfo
#define gpusparseDestroySpMat            cusparseDestroySpMat
#define gpusparseDestroySpVec            cusparseDestroySpVec
#define gpusparseDgebsr2csr              cusparseDgebsr2csr
#define gpusparseDgebsr2gebsc            cusparseDgebsr2gebsc
#define gpusparseDgebsr2gebsc_bufferSize cusparseDgebsr2gebsc_bufferSize
#define gpusparseDgebsr2gebsr            cusparseDgebsr2gebsr
#define gpusparseDgebsr2gebsr_bufferSize cusparseDgebsr2gebsr_bufferSize
#define gpusparseDgemmi                  cusparseDgemmi
#define gpusparseDgemvi                  cusparseDgemvi
#define gpusparseDgemvi_bufferSize       cusparseDgemvi_bufferSize
#define gpusparseDgpsvInterleavedBatch   cusparseDgpsvInterleavedBatch
#define gpusparseDgpsvInterleavedBatch_bufferSizeExt  \
        cusparseDgpsvInterleavedBatch_bufferSizeExt
#define gpusparseDgthr                   cusparseDgthr
#define gpusparseDgthrz                  cusparseDgthrz
#define gpusparseDgtsv2                  cusparseDgtsv2
#define gpusparseDgtsv2StridedBatch      cusparseDgtsv2StridedBatch
#define gpusparseDgtsv2StridedBatch_bufferSizeExt  \
        cusparseDgtsv2StridedBatch_bufferSizeExt
#define gpusparseDgtsv2_bufferSizeExt    cusparseDgtsv2_bufferSizeExt
#define gpusparseDgtsv2_nopivot          cusparseDgtsv2_nopivot
#define gpusparseDgtsv2_nopivot_bufferSizeExt  \
        cusparseDgtsv2_nopivot_bufferSizeExt
#define gpusparseDgtsvInterleavedBatch   cusparseDgtsvInterleavedBatch
#define gpusparseDgtsvInterleavedBatch_bufferSizeExt  \
        cusparseDgtsvInterleavedBatch_bufferSizeExt
#define gpusparseDiagType_t              cusparseDiagType_t
#define gpusparseDirection_t             cusparseDirection_t
#define gpusparseDnMatDescr_t            cusparseDnMatDescr_t
#define gpusparseDnMatGet                cusparseDnMatGet
#define gpusparseDnMatGetStridedBatch    cusparseDnMatGetStridedBatch
#define gpusparseDnMatGetValues          cusparseDnMatGetValues
#define gpusparseDnMatSetStridedBatch    cusparseDnMatSetStridedBatch
#define gpusparseDnMatSetValues          cusparseDnMatSetValues
#define gpusparseDnVecDescr_t            cusparseDnVecDescr_t
#define gpusparseDnVecGet                cusparseDnVecGet
#define gpusparseDnVecGetValues          cusparseDnVecGetValues
#define gpusparseDnVecSetValues          cusparseDnVecSetValues
#define gpusparseDnnz                    cusparseDnnz
#define gpusparseDnnz_compress           cusparseDnnz_compress
#define gpusparseDpruneCsr2csr           cusparseDpruneCsr2csr
#define gpusparseDpruneCsr2csrByPercentage  \
        cusparseDpruneCsr2csrByPercentage
#define gpusparseDpruneCsr2csrByPercentage_bufferSizeExt  \
        cusparseDpruneCsr2csrByPercentage_bufferSizeExt
#define gpusparseDpruneCsr2csrNnz        cusparseDpruneCsr2csrNnz
#define gpusparseDpruneCsr2csrNnzByPercentage  \
        cusparseDpruneCsr2csrNnzByPercentage
#define gpusparseDpruneCsr2csr_bufferSizeExt  \
        cusparseDpruneCsr2csr_bufferSizeExt
#define gpusparseDpruneDense2csr         cusparseDpruneDense2csr
#define gpusparseDpruneDense2csrByPercentage  \
        cusparseDpruneDense2csrByPercentage
#define gpusparseDpruneDense2csrByPercentage_bufferSizeExt  \
        cusparseDpruneDense2csrByPercentage_bufferSizeExt
#define gpusparseDpruneDense2csrNnz      cusparseDpruneDense2csrNnz
#define gpusparseDpruneDense2csrNnzByPercentage  \
        cusparseDpruneDense2csrNnzByPercentage
#define gpusparseDpruneDense2csr_bufferSizeExt  \
        cusparseDpruneDense2csr_bufferSizeExt
#define gpusparseDroti                   cusparseDroti
#define gpusparseDsctr                   cusparseDsctr
#define gpusparseFillMode_t              cusparseFillMode_t
#define gpusparseFormat_t                cusparseFormat_t
#define gpusparseGather                  cusparseGather
#define gpusparseGetMatDiagType          cusparseGetMatDiagType
#define gpusparseGetMatFillMode          cusparseGetMatFillMode
#define gpusparseGetMatIndexBase         cusparseGetMatIndexBase
#define gpusparseGetMatType              cusparseGetMatType
#define gpusparseGetPointerMode          cusparseGetPointerMode
#define gpusparseGetStream               cusparseGetStream
#define gpusparseGetVersion              cusparseGetVersion
#define gpusparseHandle_t                cusparseHandle_t
#define gpusparseIndexBase_t             cusparseIndexBase_t
#define gpusparseIndexType_t             cusparseIndexType_t
#define gpusparseMatDescr_t              cusparseMatDescr_t
#define gpusparseMatrixType_t            cusparseMatrixType_t
#define gpusparseOperation_t             cusparseOperation_t
#define gpusparseOrder_t                 cusparseOrder_t
#define gpusparsePointerMode_t           cusparsePointerMode_t
#define gpusparseRot                     cusparseRot
#define gpusparseSDDMM                   cusparseSDDMM
#define gpusparseSDDMMAlg_t              cusparseSDDMMAlg_t
#define gpusparseSDDMM_bufferSize        cusparseSDDMM_bufferSize
#define gpusparseSDDMM_preprocess        cusparseSDDMM_preprocess
#define gpusparseSaxpyi                  cusparseSaxpyi
#define gpusparseSbsr2csr                cusparseSbsr2csr
#define gpusparseSbsric02                cusparseSbsric02
#define gpusparseSbsric02_analysis       cusparseSbsric02_analysis
#define gpusparseSbsric02_bufferSize     cusparseSbsric02_bufferSize
#define gpusparseSbsrilu02               cusparseSbsrilu02
#define gpusparseSbsrilu02_analysis      cusparseSbsrilu02_analysis
#define gpusparseSbsrilu02_bufferSize    cusparseSbsrilu02_bufferSize
#define gpusparseSbsrilu02_numericBoost  cusparseSbsrilu02_numericBoost
#define gpusparseSbsrmm                  cusparseSbsrmm
#define gpusparseSbsrmv                  cusparseSbsrmv
#define gpusparseSbsrsm2_analysis        cusparseSbsrsm2_analysis
#define gpusparseSbsrsm2_bufferSize      cusparseSbsrsm2_bufferSize
#define gpusparseSbsrsm2_solve           cusparseSbsrsm2_solve
#define gpusparseSbsrsv2_analysis        cusparseSbsrsv2_analysis
#define gpusparseSbsrsv2_bufferSize      cusparseSbsrsv2_bufferSize
#define gpusparseSbsrsv2_bufferSizeExt   cusparseSbsrsv2_bufferSizeExt
#define gpusparseSbsrsv2_solve           cusparseSbsrsv2_solve
#define gpusparseSbsrxmv                 cusparseSbsrxmv
#define gpusparseScatter                 cusparseScatter
#define gpusparseScsc2dense              cusparseScsc2dense
#define gpusparseScsr2bsr                cusparseScsr2bsr
#define gpusparseScsr2csr_compress       cusparseScsr2csr_compress
#define gpusparseScsr2csru               cusparseScsr2csru
#define gpusparseScsr2dense              cusparseScsr2dense
#define gpusparseScsr2gebsr              cusparseScsr2gebsr
#define gpusparseScsr2gebsr_bufferSize   cusparseScsr2gebsr_bufferSize
#define gpusparseScsrcolor               cusparseScsrcolor
#define gpusparseScsrgeam2               cusparseScsrgeam2
#define gpusparseScsrgeam2_bufferSizeExt cusparseScsrgeam2_bufferSizeExt
#define gpusparseScsrgemm2               cusparseScsrgemm2
#define gpusparseScsrgemm2_bufferSizeExt cusparseScsrgemm2_bufferSizeExt
#define gpusparseScsric02                cusparseScsric02
#define gpusparseScsric02_analysis       cusparseScsric02_analysis
#define gpusparseScsric02_bufferSize     cusparseScsric02_bufferSize
#define gpusparseScsric02_bufferSizeExt  cusparseScsric02_bufferSizeExt
#define gpusparseScsrilu02               cusparseScsrilu02
#define gpusparseScsrilu02_analysis      cusparseScsrilu02_analysis
#define gpusparseScsrilu02_bufferSize    cusparseScsrilu02_bufferSize
#define gpusparseScsrilu02_bufferSizeExt cusparseScsrilu02_bufferSizeExt
#define gpusparseScsrilu02_numericBoost  cusparseScsrilu02_numericBoost
#define gpusparseScsrsm2_analysis        cusparseScsrsm2_analysis
#define gpusparseScsrsm2_bufferSizeExt   cusparseScsrsm2_bufferSizeExt
#define gpusparseScsrsm2_solve           cusparseScsrsm2_solve
#define gpusparseScsrsv2_analysis        cusparseScsrsv2_analysis
#define gpusparseScsrsv2_bufferSize      cusparseScsrsv2_bufferSize
#define gpusparseScsrsv2_bufferSizeExt   cusparseScsrsv2_bufferSizeExt
#define gpusparseScsrsv2_solve           cusparseScsrsv2_solve
#define gpusparseScsru2csr               cusparseScsru2csr
#define gpusparseScsru2csr_bufferSizeExt cusparseScsru2csr_bufferSizeExt
#define gpusparseSdense2csc              cusparseSdense2csc
#define gpusparseSdense2csr              cusparseSdense2csr
#define gpusparseSetMatDiagType          cusparseSetMatDiagType
#define gpusparseSetMatFillMode          cusparseSetMatFillMode
#define gpusparseSetMatIndexBase         cusparseSetMatIndexBase
#define gpusparseSetMatType              cusparseSetMatType
#define gpusparseSetPointerMode          cusparseSetPointerMode
#define gpusparseSetStream               cusparseSetStream
#define gpusparseSgebsr2csr              cusparseSgebsr2csr
#define gpusparseSgebsr2gebsc            cusparseSgebsr2gebsc
#define gpusparseSgebsr2gebsc_bufferSize cusparseSgebsr2gebsc_bufferSize
#define gpusparseSgebsr2gebsr            cusparseSgebsr2gebsr
#define gpusparseSgebsr2gebsr_bufferSize cusparseSgebsr2gebsr_bufferSize
#define gpusparseSgemmi                  cusparseSgemmi
#define gpusparseSgemvi                  cusparseSgemvi
#define gpusparseSgemvi_bufferSize       cusparseSgemvi_bufferSize
#define gpusparseSgpsvInterleavedBatch   cusparseSgpsvInterleavedBatch
#define gpusparseSgpsvInterleavedBatch_bufferSizeExt  \
        cusparseSgpsvInterleavedBatch_bufferSizeExt
#define gpusparseSgthr                   cusparseSgthr
#define gpusparseSgthrz                  cusparseSgthrz
#define gpusparseSgtsv2                  cusparseSgtsv2
#define gpusparseSgtsv2StridedBatch      cusparseSgtsv2StridedBatch
#define gpusparseSgtsv2StridedBatch_bufferSizeExt  \
        cusparseSgtsv2StridedBatch_bufferSizeExt
#define gpusparseSgtsv2_bufferSizeExt    cusparseSgtsv2_bufferSizeExt
#define gpusparseSgtsv2_nopivot          cusparseSgtsv2_nopivot
#define gpusparseSgtsv2_nopivot_bufferSizeExt  \
        cusparseSgtsv2_nopivot_bufferSizeExt
#define gpusparseSgtsvInterleavedBatch   cusparseSgtsvInterleavedBatch
#define gpusparseSgtsvInterleavedBatch_bufferSizeExt  \
        cusparseSgtsvInterleavedBatch_bufferSizeExt
#define gpusparseSnnz                    cusparseSnnz
#define gpusparseSnnz_compress           cusparseSnnz_compress
#define gpusparseSolvePolicy_t           cusparseSolvePolicy_t
#define gpusparseSpGEMMAlg_t             cusparseSpGEMMAlg_t
#define gpusparseSpGEMMDescr_t           cusparseSpGEMMDescr_t
#define gpusparseSpGEMM_compute          cusparseSpGEMM_compute
#define gpusparseSpGEMM_copy             cusparseSpGEMM_copy
#define gpusparseSpGEMM_createDescr      cusparseSpGEMM_createDescr
#define gpusparseSpGEMM_destroyDescr     cusparseSpGEMM_destroyDescr
#define gpusparseSpGEMM_workEstimation   cusparseSpGEMM_workEstimation
#define gpusparseSpMM                    cusparseSpMM
#define gpusparseSpMMAlg_t               cusparseSpMMAlg_t
#define gpusparseSpMM_bufferSize         cusparseSpMM_bufferSize
#define gpusparseSpMM_preprocess         cusparseSpMM_preprocess
#define gpusparseSpMV                    cusparseSpMV
#define gpusparseSpMVAlg_t               cusparseSpMVAlg_t
#define gpusparseSpMV_bufferSize         cusparseSpMV_bufferSize
#define gpusparseSpMatAttribute_t        cusparseSpMatAttribute_t
#define gpusparseSpMatDescr_t            cusparseSpMatDescr_t
#define gpusparseSpMatGetAttribute       cusparseSpMatGetAttribute
#define gpusparseSpMatGetFormat          cusparseSpMatGetFormat
#define gpusparseSpMatGetIndexBase       cusparseSpMatGetIndexBase
#define gpusparseSpMatGetSize            cusparseSpMatGetSize
#define gpusparseSpMatGetStridedBatch    cusparseSpMatGetStridedBatch
#define gpusparseSpMatGetValues          cusparseSpMatGetValues
#define gpusparseSpMatSetAttribute       cusparseSpMatSetAttribute
#define gpusparseSpMatSetStridedBatch    cusparseSpMatSetStridedBatch
#define gpusparseSpMatSetValues          cusparseSpMatSetValues
#define gpusparseSpSMAlg_t               cusparseSpSMAlg_t
#define gpusparseSpSMDescr_t             cusparseSpSMDescr_t
#define gpusparseSpSM_analysis           cusparseSpSM_analysis
#define gpusparseSpSM_bufferSize         cusparseSpSM_bufferSize
#define gpusparseSpSM_createDescr        cusparseSpSM_createDescr
#define gpusparseSpSM_destroyDescr       cusparseSpSM_destroyDescr
#define gpusparseSpSM_solve              cusparseSpSM_solve
#define gpusparseSpSVAlg_t               cusparseSpSVAlg_t
#define gpusparseSpSVDescr_t             cusparseSpSVDescr_t
#define gpusparseSpSV_analysis           cusparseSpSV_analysis
#define gpusparseSpSV_bufferSize         cusparseSpSV_bufferSize
#define gpusparseSpSV_createDescr        cusparseSpSV_createDescr
#define gpusparseSpSV_destroyDescr       cusparseSpSV_destroyDescr
#define gpusparseSpSV_solve              cusparseSpSV_solve
#define gpusparseSpVV                    cusparseSpVV
#define gpusparseSpVV_bufferSize         cusparseSpVV_bufferSize
#define gpusparseSpVecDescr_t            cusparseSpVecDescr_t
#define gpusparseSpVecGet                cusparseSpVecGet
#define gpusparseSpVecGetIndexBase       cusparseSpVecGetIndexBase
#define gpusparseSpVecGetValues          cusparseSpVecGetValues
#define gpusparseSpVecSetValues          cusparseSpVecSetValues
#define gpusparseSparseToDense           cusparseSparseToDense
#define gpusparseSparseToDenseAlg_t      cusparseSparseToDenseAlg_t
#define gpusparseSparseToDense_bufferSize  \
        cusparseSparseToDense_bufferSize
#define gpusparseSpruneCsr2csr           cusparseSpruneCsr2csr
#define gpusparseSpruneCsr2csrByPercentage  \
        cusparseSpruneCsr2csrByPercentage
#define gpusparseSpruneCsr2csrByPercentage_bufferSizeExt  \
        cusparseSpruneCsr2csrByPercentage_bufferSizeExt
#define gpusparseSpruneCsr2csrNnz        cusparseSpruneCsr2csrNnz
#define gpusparseSpruneCsr2csrNnzByPercentage  \
        cusparseSpruneCsr2csrNnzByPercentage
#define gpusparseSpruneCsr2csr_bufferSizeExt  \
        cusparseSpruneCsr2csr_bufferSizeExt
#define gpusparseSpruneDense2csr         cusparseSpruneDense2csr
#define gpusparseSpruneDense2csrByPercentage  \
        cusparseSpruneDense2csrByPercentage
#define gpusparseSpruneDense2csrByPercentage_bufferSizeExt  \
        cusparseSpruneDense2csrByPercentage_bufferSizeExt
#define gpusparseSpruneDense2csrNnz      cusparseSpruneDense2csrNnz
#define gpusparseSpruneDense2csrNnzByPercentage  \
        cusparseSpruneDense2csrNnzByPercentage
#define gpusparseSpruneDense2csr_bufferSizeExt  \
        cusparseSpruneDense2csr_bufferSizeExt
#define gpusparseSroti                   cusparseSroti
#define gpusparseSsctr                   cusparseSsctr
#define gpusparseStatus_t                cusparseStatus_t
#define gpusparseXbsric02_zeroPivot      cusparseXbsric02_zeroPivot
#define gpusparseXbsrilu02_zeroPivot     cusparseXbsrilu02_zeroPivot
#define gpusparseXbsrsm2_zeroPivot       cusparseXbsrsm2_zeroPivot
#define gpusparseXbsrsv2_zeroPivot       cusparseXbsrsv2_zeroPivot
#define gpusparseXcoo2csr                cusparseXcoo2csr
#define gpusparseXcoosortByColumn        cusparseXcoosortByColumn
#define gpusparseXcoosortByRow           cusparseXcoosortByRow
#define gpusparseXcoosort_bufferSizeExt  cusparseXcoosort_bufferSizeExt
#define gpusparseXcscsort                cusparseXcscsort
#define gpusparseXcscsort_bufferSizeExt  cusparseXcscsort_bufferSizeExt
#define gpusparseXcsr2bsrNnz             cusparseXcsr2bsrNnz
#define gpusparseXcsr2coo                cusparseXcsr2coo
#define gpusparseXcsr2gebsrNnz           cusparseXcsr2gebsrNnz
#define gpusparseXcsrgeam2Nnz            cusparseXcsrgeam2Nnz
#define gpusparseXcsrgemm2Nnz            cusparseXcsrgemm2Nnz
#define gpusparseXcsric02_zeroPivot      cusparseXcsric02_zeroPivot
#define gpusparseXcsrilu02_zeroPivot     cusparseXcsrilu02_zeroPivot
#define gpusparseXcsrsm2_zeroPivot       cusparseXcsrsm2_zeroPivot
#define gpusparseXcsrsort                cusparseXcsrsort
#define gpusparseXcsrsort_bufferSizeExt  cusparseXcsrsort_bufferSizeExt
#define gpusparseXcsrsv2_zeroPivot       cusparseXcsrsv2_zeroPivot
#define gpusparseXgebsr2gebsrNnz         cusparseXgebsr2gebsrNnz
#define gpusparseZaxpyi                  cusparseZaxpyi
#define gpusparseZbsr2csr                cusparseZbsr2csr
#define gpusparseZbsric02                cusparseZbsric02
#define gpusparseZbsric02_analysis       cusparseZbsric02_analysis
#define gpusparseZbsric02_bufferSize     cusparseZbsric02_bufferSize
#define gpusparseZbsrilu02               cusparseZbsrilu02
#define gpusparseZbsrilu02_analysis      cusparseZbsrilu02_analysis
#define gpusparseZbsrilu02_bufferSize    cusparseZbsrilu02_bufferSize
#define gpusparseZbsrilu02_numericBoost  cusparseZbsrilu02_numericBoost
#define gpusparseZbsrmm                  cusparseZbsrmm
#define gpusparseZbsrmv                  cusparseZbsrmv
#define gpusparseZbsrsm2_analysis        cusparseZbsrsm2_analysis
#define gpusparseZbsrsm2_bufferSize      cusparseZbsrsm2_bufferSize
#define gpusparseZbsrsm2_solve           cusparseZbsrsm2_solve
#define gpusparseZbsrsv2_analysis        cusparseZbsrsv2_analysis
#define gpusparseZbsrsv2_bufferSize      cusparseZbsrsv2_bufferSize
#define gpusparseZbsrsv2_bufferSizeExt   cusparseZbsrsv2_bufferSizeExt
#define gpusparseZbsrsv2_solve           cusparseZbsrsv2_solve
#define gpusparseZbsrxmv                 cusparseZbsrxmv
#define gpusparseZcsc2dense              cusparseZcsc2dense
#define gpusparseZcsr2bsr                cusparseZcsr2bsr
#define gpusparseZcsr2csr_compress       cusparseZcsr2csr_compress
#define gpusparseZcsr2csru               cusparseZcsr2csru
#define gpusparseZcsr2dense              cusparseZcsr2dense
#define gpusparseZcsr2gebsr              cusparseZcsr2gebsr
#define gpusparseZcsr2gebsr_bufferSize   cusparseZcsr2gebsr_bufferSize
#define gpusparseZcsrcolor               cusparseZcsrcolor
#define gpusparseZcsrgeam2               cusparseZcsrgeam2
#define gpusparseZcsrgeam2_bufferSizeExt cusparseZcsrgeam2_bufferSizeExt
#define gpusparseZcsrgemm2               cusparseZcsrgemm2
#define gpusparseZcsrgemm2_bufferSizeExt cusparseZcsrgemm2_bufferSizeExt
#define gpusparseZcsric02                cusparseZcsric02
#define gpusparseZcsric02_analysis       cusparseZcsric02_analysis
#define gpusparseZcsric02_bufferSize     cusparseZcsric02_bufferSize
#define gpusparseZcsric02_bufferSizeExt  cusparseZcsric02_bufferSizeExt
#define gpusparseZcsrilu02               cusparseZcsrilu02
#define gpusparseZcsrilu02_analysis      cusparseZcsrilu02_analysis
#define gpusparseZcsrilu02_bufferSize    cusparseZcsrilu02_bufferSize
#define gpusparseZcsrilu02_bufferSizeExt cusparseZcsrilu02_bufferSizeExt
#define gpusparseZcsrilu02_numericBoost  cusparseZcsrilu02_numericBoost
#define gpusparseZcsrsm2_analysis        cusparseZcsrsm2_analysis
#define gpusparseZcsrsm2_bufferSizeExt   cusparseZcsrsm2_bufferSizeExt
#define gpusparseZcsrsm2_solve           cusparseZcsrsm2_solve
#define gpusparseZcsrsv2_analysis        cusparseZcsrsv2_analysis
#define gpusparseZcsrsv2_bufferSize      cusparseZcsrsv2_bufferSize
#define gpusparseZcsrsv2_bufferSizeExt   cusparseZcsrsv2_bufferSizeExt
#define gpusparseZcsrsv2_solve           cusparseZcsrsv2_solve
#define gpusparseZcsru2csr               cusparseZcsru2csr
#define gpusparseZcsru2csr_bufferSizeExt cusparseZcsru2csr_bufferSizeExt
#define gpusparseZdense2csc              cusparseZdense2csc
#define gpusparseZdense2csr              cusparseZdense2csr
#define gpusparseZgebsr2csr              cusparseZgebsr2csr
#define gpusparseZgebsr2gebsc            cusparseZgebsr2gebsc
#define gpusparseZgebsr2gebsc_bufferSize cusparseZgebsr2gebsc_bufferSize
#define gpusparseZgebsr2gebsr            cusparseZgebsr2gebsr
#define gpusparseZgebsr2gebsr_bufferSize cusparseZgebsr2gebsr_bufferSize
#define gpusparseZgemmi                  cusparseZgemmi
#define gpusparseZgemvi                  cusparseZgemvi
#define gpusparseZgemvi_bufferSize       cusparseZgemvi_bufferSize
#define gpusparseZgpsvInterleavedBatch   cusparseZgpsvInterleavedBatch
#define gpusparseZgpsvInterleavedBatch_bufferSizeExt  \
        cusparseZgpsvInterleavedBatch_bufferSizeExt
#define gpusparseZgthr                   cusparseZgthr
#define gpusparseZgthrz                  cusparseZgthrz
#define gpusparseZgtsv2                  cusparseZgtsv2
#define gpusparseZgtsv2StridedBatch      cusparseZgtsv2StridedBatch
#define gpusparseZgtsv2StridedBatch_bufferSizeExt  \
        cusparseZgtsv2StridedBatch_bufferSizeExt
#define gpusparseZgtsv2_bufferSizeExt    cusparseZgtsv2_bufferSizeExt
#define gpusparseZgtsv2_nopivot          cusparseZgtsv2_nopivot
#define gpusparseZgtsv2_nopivot_bufferSizeExt  \
        cusparseZgtsv2_nopivot_bufferSizeExt
#define gpusparseZgtsvInterleavedBatch   cusparseZgtsvInterleavedBatch
#define gpusparseZgtsvInterleavedBatch_bufferSizeExt  \
        cusparseZgtsvInterleavedBatch_bufferSizeExt
#define gpusparseZnnz                    cusparseZnnz
#define gpusparseZnnz_compress           cusparseZnnz_compress
#define gpusparseZsctr                   cusparseZsctr


#endif
