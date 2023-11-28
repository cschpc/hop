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

#ifndef __HOP_SOURCE_CUDA_CUSPARSE_H__
#define __HOP_SOURCE_CUDA_CUSPARSE_H__

#define HOP_SOURCE_CUDA

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <library_types.h>

#define CUSPARSE_ACTION_NUMERIC          GPUSPARSE_ACTION_NUMERIC
#define CUSPARSE_ACTION_SYMBOLIC         GPUSPARSE_ACTION_SYMBOLIC
#define CUSPARSE_COOMM_ALG1              GPUSPARSE_COOMM_ALG1
#define CUSPARSE_COOMM_ALG2              GPUSPARSE_COOMM_ALG2
#define CUSPARSE_COOMM_ALG3              GPUSPARSE_COOMM_ALG3
#define CUSPARSE_COOMV_ALG               GPUSPARSE_COOMV_ALG
#define CUSPARSE_CSR2CSC_ALG1            GPUSPARSE_CSR2CSC_ALG1
#define CUSPARSE_CSR2CSC_ALG2            GPUSPARSE_CSR2CSC_ALG2
#define CUSPARSE_CSR2CSC_ALG_DEFAULT     GPUSPARSE_CSR2CSC_ALG_DEFAULT
#define CUSPARSE_CSRMM_ALG1              GPUSPARSE_CSRMM_ALG1
#define CUSPARSE_CSRMV_ALG1              GPUSPARSE_CSRMV_ALG1
#define CUSPARSE_CSRMV_ALG2              GPUSPARSE_CSRMV_ALG2
#define CUSPARSE_DENSETOSPARSE_ALG_DEFAULT  \
        GPUSPARSE_DENSETOSPARSE_ALG_DEFAULT
#define CUSPARSE_DIAG_TYPE_NON_UNIT      GPUSPARSE_DIAG_TYPE_NON_UNIT
#define CUSPARSE_DIAG_TYPE_UNIT          GPUSPARSE_DIAG_TYPE_UNIT
#define CUSPARSE_DIRECTION_COLUMN        GPUSPARSE_DIRECTION_COLUMN
#define CUSPARSE_DIRECTION_ROW           GPUSPARSE_DIRECTION_ROW
#define CUSPARSE_FILL_MODE_LOWER         GPUSPARSE_FILL_MODE_LOWER
#define CUSPARSE_FILL_MODE_UPPER         GPUSPARSE_FILL_MODE_UPPER
#define CUSPARSE_FORMAT_BLOCKED_ELL      GPUSPARSE_FORMAT_BLOCKED_ELL
#define CUSPARSE_FORMAT_COO              GPUSPARSE_FORMAT_COO
#define CUSPARSE_FORMAT_COO_AOS          GPUSPARSE_FORMAT_COO_AOS
#define CUSPARSE_FORMAT_CSC              GPUSPARSE_FORMAT_CSC
#define CUSPARSE_FORMAT_CSR              GPUSPARSE_FORMAT_CSR
#define CUSPARSE_INDEX_16U               GPUSPARSE_INDEX_16U
#define CUSPARSE_INDEX_32I               GPUSPARSE_INDEX_32I
#define CUSPARSE_INDEX_64I               GPUSPARSE_INDEX_64I
#define CUSPARSE_INDEX_BASE_ONE          GPUSPARSE_INDEX_BASE_ONE
#define CUSPARSE_INDEX_BASE_ZERO         GPUSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_MATRIX_TYPE_GENERAL     GPUSPARSE_MATRIX_TYPE_GENERAL
#define CUSPARSE_MATRIX_TYPE_HERMITIAN   GPUSPARSE_MATRIX_TYPE_HERMITIAN
#define CUSPARSE_MATRIX_TYPE_SYMMETRIC   GPUSPARSE_MATRIX_TYPE_SYMMETRIC
#define CUSPARSE_MATRIX_TYPE_TRIANGULAR  GPUSPARSE_MATRIX_TYPE_TRIANGULAR
#define CUSPARSE_MM_ALG_DEFAULT          GPUSPARSE_MM_ALG_DEFAULT
#define CUSPARSE_MV_ALG_DEFAULT          GPUSPARSE_MV_ALG_DEFAULT
#define CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE  \
        GPUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
#define CUSPARSE_OPERATION_NON_TRANSPOSE GPUSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE     GPUSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_ORDER_COL               GPUSPARSE_ORDER_COL
#define CUSPARSE_ORDER_ROW               GPUSPARSE_ORDER_ROW
#define CUSPARSE_POINTER_MODE_DEVICE     GPUSPARSE_POINTER_MODE_DEVICE
#define CUSPARSE_POINTER_MODE_HOST       GPUSPARSE_POINTER_MODE_HOST
#define CUSPARSE_SDDMM_ALG_DEFAULT       GPUSPARSE_SDDMM_ALG_DEFAULT
#define CUSPARSE_SOLVE_POLICY_NO_LEVEL   GPUSPARSE_SOLVE_POLICY_NO_LEVEL
#define CUSPARSE_SOLVE_POLICY_USE_LEVEL  GPUSPARSE_SOLVE_POLICY_USE_LEVEL
#define CUSPARSE_SPARSETODENSE_ALG_DEFAULT  \
        GPUSPARSE_SPARSETODENSE_ALG_DEFAULT
#define CUSPARSE_SPGEMM_ALG1             GPUSPARSE_SPGEMM_ALG1
#define CUSPARSE_SPGEMM_ALG2             GPUSPARSE_SPGEMM_ALG2
#define CUSPARSE_SPGEMM_ALG3             GPUSPARSE_SPGEMM_ALG3
#define CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC  \
        GPUSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC
#define CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC  \
        GPUSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC
#define CUSPARSE_SPGEMM_DEFAULT          GPUSPARSE_SPGEMM_DEFAULT
#define CUSPARSE_SPMAT_DIAG_TYPE         GPUSPARSE_SPMAT_DIAG_TYPE
#define CUSPARSE_SPMAT_FILL_MODE         GPUSPARSE_SPMAT_FILL_MODE
#define CUSPARSE_SPMM_ALG_DEFAULT        GPUSPARSE_SPMM_ALG_DEFAULT
#define CUSPARSE_SPMM_BLOCKED_ELL_ALG1   GPUSPARSE_SPMM_BLOCKED_ELL_ALG1
#define CUSPARSE_SPMM_COO_ALG1           GPUSPARSE_SPMM_COO_ALG1
#define CUSPARSE_SPMM_COO_ALG2           GPUSPARSE_SPMM_COO_ALG2
#define CUSPARSE_SPMM_COO_ALG3           GPUSPARSE_SPMM_COO_ALG3
#define CUSPARSE_SPMM_COO_ALG4           GPUSPARSE_SPMM_COO_ALG4
#define CUSPARSE_SPMM_CSR_ALG1           GPUSPARSE_SPMM_CSR_ALG1
#define CUSPARSE_SPMM_CSR_ALG2           GPUSPARSE_SPMM_CSR_ALG2
#define CUSPARSE_SPMM_CSR_ALG3           GPUSPARSE_SPMM_CSR_ALG3
#define CUSPARSE_SPMV_ALG_DEFAULT        GPUSPARSE_SPMV_ALG_DEFAULT
#define CUSPARSE_SPMV_COO_ALG1           GPUSPARSE_SPMV_COO_ALG1
#define CUSPARSE_SPMV_COO_ALG2           GPUSPARSE_SPMV_COO_ALG2
#define CUSPARSE_SPMV_CSR_ALG1           GPUSPARSE_SPMV_CSR_ALG1
#define CUSPARSE_SPMV_CSR_ALG2           GPUSPARSE_SPMV_CSR_ALG2
#define CUSPARSE_SPSM_ALG_DEFAULT        GPUSPARSE_SPSM_ALG_DEFAULT
#define CUSPARSE_SPSV_ALG_DEFAULT        GPUSPARSE_SPSV_ALG_DEFAULT
#define CUSPARSE_STATUS_ALLOC_FAILED     GPUSPARSE_STATUS_ALLOC_FAILED
#define CUSPARSE_STATUS_ARCH_MISMATCH    GPUSPARSE_STATUS_ARCH_MISMATCH
#define CUSPARSE_STATUS_EXECUTION_FAILED GPUSPARSE_STATUS_EXECUTION_FAILED
#define CUSPARSE_STATUS_INSUFFICIENT_RESOURCES  \
        GPUSPARSE_STATUS_INSUFFICIENT_RESOURCES
#define CUSPARSE_STATUS_INTERNAL_ERROR   GPUSPARSE_STATUS_INTERNAL_ERROR
#define CUSPARSE_STATUS_INVALID_VALUE    GPUSPARSE_STATUS_INVALID_VALUE
#define CUSPARSE_STATUS_MAPPING_ERROR    GPUSPARSE_STATUS_MAPPING_ERROR
#define CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED  \
        GPUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED
#define CUSPARSE_STATUS_NOT_INITIALIZED  GPUSPARSE_STATUS_NOT_INITIALIZED
#define CUSPARSE_STATUS_NOT_SUPPORTED    GPUSPARSE_STATUS_NOT_SUPPORTED
#define CUSPARSE_STATUS_SUCCESS          GPUSPARSE_STATUS_SUCCESS
#define CUSPARSE_STATUS_ZERO_PIVOT       GPUSPARSE_STATUS_ZERO_PIVOT
#define cusparseAction_t                 gpusparseAction_t
#define cusparseAxpby                    gpusparseAxpby
#define cusparseBlockedEllGet            gpusparseBlockedEllGet
#define cusparseCaxpyi                   gpusparseCaxpyi
#define cusparseCbsr2csr                 gpusparseCbsr2csr
#define cusparseCbsric02                 gpusparseCbsric02
#define cusparseCbsric02_analysis        gpusparseCbsric02_analysis
#define cusparseCbsric02_bufferSize      gpusparseCbsric02_bufferSize
#define cusparseCbsrilu02                gpusparseCbsrilu02
#define cusparseCbsrilu02_analysis       gpusparseCbsrilu02_analysis
#define cusparseCbsrilu02_bufferSize     gpusparseCbsrilu02_bufferSize
#define cusparseCbsrilu02_numericBoost   gpusparseCbsrilu02_numericBoost
#define cusparseCbsrmm                   gpusparseCbsrmm
#define cusparseCbsrmv                   gpusparseCbsrmv
#define cusparseCbsrsm2_analysis         gpusparseCbsrsm2_analysis
#define cusparseCbsrsm2_bufferSize       gpusparseCbsrsm2_bufferSize
#define cusparseCbsrsm2_solve            gpusparseCbsrsm2_solve
#define cusparseCbsrsv2_analysis         gpusparseCbsrsv2_analysis
#define cusparseCbsrsv2_bufferSize       gpusparseCbsrsv2_bufferSize
#define cusparseCbsrsv2_bufferSizeExt    gpusparseCbsrsv2_bufferSizeExt
#define cusparseCbsrsv2_solve            gpusparseCbsrsv2_solve
#define cusparseCbsrxmv                  gpusparseCbsrxmv
#define cusparseCcsc2dense               gpusparseCcsc2dense
#define cusparseCcsr2bsr                 gpusparseCcsr2bsr
#define cusparseCcsr2csr_compress        gpusparseCcsr2csr_compress
#define cusparseCcsr2csru                gpusparseCcsr2csru
#define cusparseCcsr2dense               gpusparseCcsr2dense
#define cusparseCcsr2gebsr               gpusparseCcsr2gebsr
#define cusparseCcsr2gebsr_bufferSize    gpusparseCcsr2gebsr_bufferSize
#define cusparseCcsrcolor                gpusparseCcsrcolor
#define cusparseCcsrgeam2                gpusparseCcsrgeam2
#define cusparseCcsrgeam2_bufferSizeExt  gpusparseCcsrgeam2_bufferSizeExt
#define cusparseCcsrgemm2                gpusparseCcsrgemm2
#define cusparseCcsrgemm2_bufferSizeExt  gpusparseCcsrgemm2_bufferSizeExt
#define cusparseCcsric02                 gpusparseCcsric02
#define cusparseCcsric02_analysis        gpusparseCcsric02_analysis
#define cusparseCcsric02_bufferSize      gpusparseCcsric02_bufferSize
#define cusparseCcsric02_bufferSizeExt   gpusparseCcsric02_bufferSizeExt
#define cusparseCcsrilu02                gpusparseCcsrilu02
#define cusparseCcsrilu02_analysis       gpusparseCcsrilu02_analysis
#define cusparseCcsrilu02_bufferSize     gpusparseCcsrilu02_bufferSize
#define cusparseCcsrilu02_bufferSizeExt  gpusparseCcsrilu02_bufferSizeExt
#define cusparseCcsrilu02_numericBoost   gpusparseCcsrilu02_numericBoost
#define cusparseCcsrsm2_analysis         gpusparseCcsrsm2_analysis
#define cusparseCcsrsm2_bufferSizeExt    gpusparseCcsrsm2_bufferSizeExt
#define cusparseCcsrsm2_solve            gpusparseCcsrsm2_solve
#define cusparseCcsrsv2_analysis         gpusparseCcsrsv2_analysis
#define cusparseCcsrsv2_bufferSize       gpusparseCcsrsv2_bufferSize
#define cusparseCcsrsv2_bufferSizeExt    gpusparseCcsrsv2_bufferSizeExt
#define cusparseCcsrsv2_solve            gpusparseCcsrsv2_solve
#define cusparseCcsru2csr                gpusparseCcsru2csr
#define cusparseCcsru2csr_bufferSizeExt  gpusparseCcsru2csr_bufferSizeExt
#define cusparseCdense2csc               gpusparseCdense2csc
#define cusparseCdense2csr               gpusparseCdense2csr
#define cusparseCgebsr2csr               gpusparseCgebsr2csr
#define cusparseCgebsr2gebsc             gpusparseCgebsr2gebsc
#define cusparseCgebsr2gebsc_bufferSize  gpusparseCgebsr2gebsc_bufferSize
#define cusparseCgebsr2gebsr             gpusparseCgebsr2gebsr
#define cusparseCgebsr2gebsr_bufferSize  gpusparseCgebsr2gebsr_bufferSize
#define cusparseCgemmi                   gpusparseCgemmi
#define cusparseCgemvi                   gpusparseCgemvi
#define cusparseCgemvi_bufferSize        gpusparseCgemvi_bufferSize
#define cusparseCgpsvInterleavedBatch    gpusparseCgpsvInterleavedBatch
#define cusparseCgpsvInterleavedBatch_bufferSizeExt  \
        gpusparseCgpsvInterleavedBatch_bufferSizeExt
#define cusparseCgthr                    gpusparseCgthr
#define cusparseCgthrz                   gpusparseCgthrz
#define cusparseCgtsv2                   gpusparseCgtsv2
#define cusparseCgtsv2StridedBatch       gpusparseCgtsv2StridedBatch
#define cusparseCgtsv2StridedBatch_bufferSizeExt  \
        gpusparseCgtsv2StridedBatch_bufferSizeExt
#define cusparseCgtsv2_bufferSizeExt     gpusparseCgtsv2_bufferSizeExt
#define cusparseCgtsv2_nopivot           gpusparseCgtsv2_nopivot
#define cusparseCgtsv2_nopivot_bufferSizeExt  \
        gpusparseCgtsv2_nopivot_bufferSizeExt
#define cusparseCgtsvInterleavedBatch    gpusparseCgtsvInterleavedBatch
#define cusparseCgtsvInterleavedBatch_bufferSizeExt  \
        gpusparseCgtsvInterleavedBatch_bufferSizeExt
#define cusparseCnnz                     gpusparseCnnz
#define cusparseCnnz_compress            gpusparseCnnz_compress
#define cusparseColorInfo_t              gpusparseColorInfo_t
#define cusparseCooAoSGet                gpusparseCooAoSGet
#define cusparseCooGet                   gpusparseCooGet
#define cusparseCooSetPointers           gpusparseCooSetPointers
#define cusparseCooSetStridedBatch       gpusparseCooSetStridedBatch
#define cusparseCopyMatDescr             gpusparseCopyMatDescr
#define cusparseCreate                   gpusparseCreate
#define cusparseCreateBlockedEll         gpusparseCreateBlockedEll
#define cusparseCreateBsric02Info        gpusparseCreateBsric02Info
#define cusparseCreateBsrilu02Info       gpusparseCreateBsrilu02Info
#define cusparseCreateBsrsm2Info         gpusparseCreateBsrsm2Info
#define cusparseCreateBsrsv2Info         gpusparseCreateBsrsv2Info
#define cusparseCreateColorInfo          gpusparseCreateColorInfo
#define cusparseCreateCoo                gpusparseCreateCoo
#define cusparseCreateCooAoS             gpusparseCreateCooAoS
#define cusparseCreateCsc                gpusparseCreateCsc
#define cusparseCreateCsr                gpusparseCreateCsr
#define cusparseCreateCsrgemm2Info       gpusparseCreateCsrgemm2Info
#define cusparseCreateCsric02Info        gpusparseCreateCsric02Info
#define cusparseCreateCsrilu02Info       gpusparseCreateCsrilu02Info
#define cusparseCreateCsrsm2Info         gpusparseCreateCsrsm2Info
#define cusparseCreateCsrsv2Info         gpusparseCreateCsrsv2Info
#define cusparseCreateCsru2csrInfo       gpusparseCreateCsru2csrInfo
#define cusparseCreateDnMat              gpusparseCreateDnMat
#define cusparseCreateDnVec              gpusparseCreateDnVec
#define cusparseCreateIdentityPermutation  \
        gpusparseCreateIdentityPermutation
#define cusparseCreateMatDescr           gpusparseCreateMatDescr
#define cusparseCreatePruneInfo          gpusparseCreatePruneInfo
#define cusparseCreateSpVec              gpusparseCreateSpVec
#define cusparseCscSetPointers           gpusparseCscSetPointers
#define cusparseCsctr                    gpusparseCsctr
#define cusparseCsr2CscAlg_t             gpusparseCsr2CscAlg_t
#define cusparseCsr2cscEx2               gpusparseCsr2cscEx2
#define cusparseCsr2cscEx2_bufferSize    gpusparseCsr2cscEx2_bufferSize
#define cusparseCsrGet                   gpusparseCsrGet
#define cusparseCsrSetPointers           gpusparseCsrSetPointers
#define cusparseCsrSetStridedBatch       gpusparseCsrSetStridedBatch
#define cusparseDaxpyi                   gpusparseDaxpyi
#define cusparseDbsr2csr                 gpusparseDbsr2csr
#define cusparseDbsric02                 gpusparseDbsric02
#define cusparseDbsric02_analysis        gpusparseDbsric02_analysis
#define cusparseDbsric02_bufferSize      gpusparseDbsric02_bufferSize
#define cusparseDbsrilu02                gpusparseDbsrilu02
#define cusparseDbsrilu02_analysis       gpusparseDbsrilu02_analysis
#define cusparseDbsrilu02_bufferSize     gpusparseDbsrilu02_bufferSize
#define cusparseDbsrilu02_numericBoost   gpusparseDbsrilu02_numericBoost
#define cusparseDbsrmm                   gpusparseDbsrmm
#define cusparseDbsrmv                   gpusparseDbsrmv
#define cusparseDbsrsm2_analysis         gpusparseDbsrsm2_analysis
#define cusparseDbsrsm2_bufferSize       gpusparseDbsrsm2_bufferSize
#define cusparseDbsrsm2_solve            gpusparseDbsrsm2_solve
#define cusparseDbsrsv2_analysis         gpusparseDbsrsv2_analysis
#define cusparseDbsrsv2_bufferSize       gpusparseDbsrsv2_bufferSize
#define cusparseDbsrsv2_bufferSizeExt    gpusparseDbsrsv2_bufferSizeExt
#define cusparseDbsrsv2_solve            gpusparseDbsrsv2_solve
#define cusparseDbsrxmv                  gpusparseDbsrxmv
#define cusparseDcsc2dense               gpusparseDcsc2dense
#define cusparseDcsr2bsr                 gpusparseDcsr2bsr
#define cusparseDcsr2csr_compress        gpusparseDcsr2csr_compress
#define cusparseDcsr2csru                gpusparseDcsr2csru
#define cusparseDcsr2dense               gpusparseDcsr2dense
#define cusparseDcsr2gebsr               gpusparseDcsr2gebsr
#define cusparseDcsr2gebsr_bufferSize    gpusparseDcsr2gebsr_bufferSize
#define cusparseDcsrcolor                gpusparseDcsrcolor
#define cusparseDcsrgeam2                gpusparseDcsrgeam2
#define cusparseDcsrgeam2_bufferSizeExt  gpusparseDcsrgeam2_bufferSizeExt
#define cusparseDcsrgemm2                gpusparseDcsrgemm2
#define cusparseDcsrgemm2_bufferSizeExt  gpusparseDcsrgemm2_bufferSizeExt
#define cusparseDcsric02                 gpusparseDcsric02
#define cusparseDcsric02_analysis        gpusparseDcsric02_analysis
#define cusparseDcsric02_bufferSize      gpusparseDcsric02_bufferSize
#define cusparseDcsric02_bufferSizeExt   gpusparseDcsric02_bufferSizeExt
#define cusparseDcsrilu02                gpusparseDcsrilu02
#define cusparseDcsrilu02_analysis       gpusparseDcsrilu02_analysis
#define cusparseDcsrilu02_bufferSize     gpusparseDcsrilu02_bufferSize
#define cusparseDcsrilu02_bufferSizeExt  gpusparseDcsrilu02_bufferSizeExt
#define cusparseDcsrilu02_numericBoost   gpusparseDcsrilu02_numericBoost
#define cusparseDcsrsm2_analysis         gpusparseDcsrsm2_analysis
#define cusparseDcsrsm2_bufferSizeExt    gpusparseDcsrsm2_bufferSizeExt
#define cusparseDcsrsm2_solve            gpusparseDcsrsm2_solve
#define cusparseDcsrsv2_analysis         gpusparseDcsrsv2_analysis
#define cusparseDcsrsv2_bufferSize       gpusparseDcsrsv2_bufferSize
#define cusparseDcsrsv2_bufferSizeExt    gpusparseDcsrsv2_bufferSizeExt
#define cusparseDcsrsv2_solve            gpusparseDcsrsv2_solve
#define cusparseDcsru2csr                gpusparseDcsru2csr
#define cusparseDcsru2csr_bufferSizeExt  gpusparseDcsru2csr_bufferSizeExt
#define cusparseDdense2csc               gpusparseDdense2csc
#define cusparseDdense2csr               gpusparseDdense2csr
#define cusparseDenseToSparseAlg_t       gpusparseDenseToSparseAlg_t
#define cusparseDenseToSparse_analysis   gpusparseDenseToSparse_analysis
#define cusparseDenseToSparse_bufferSize gpusparseDenseToSparse_bufferSize
#define cusparseDenseToSparse_convert    gpusparseDenseToSparse_convert
#define cusparseDestroy                  gpusparseDestroy
#define cusparseDestroyBsric02Info       gpusparseDestroyBsric02Info
#define cusparseDestroyBsrilu02Info      gpusparseDestroyBsrilu02Info
#define cusparseDestroyBsrsm2Info        gpusparseDestroyBsrsm2Info
#define cusparseDestroyBsrsv2Info        gpusparseDestroyBsrsv2Info
#define cusparseDestroyColorInfo         gpusparseDestroyColorInfo
#define cusparseDestroyCsrgemm2Info      gpusparseDestroyCsrgemm2Info
#define cusparseDestroyCsric02Info       gpusparseDestroyCsric02Info
#define cusparseDestroyCsrilu02Info      gpusparseDestroyCsrilu02Info
#define cusparseDestroyCsrsm2Info        gpusparseDestroyCsrsm2Info
#define cusparseDestroyCsrsv2Info        gpusparseDestroyCsrsv2Info
#define cusparseDestroyCsru2csrInfo      gpusparseDestroyCsru2csrInfo
#define cusparseDestroyDnMat             gpusparseDestroyDnMat
#define cusparseDestroyDnVec             gpusparseDestroyDnVec
#define cusparseDestroyMatDescr          gpusparseDestroyMatDescr
#define cusparseDestroyPruneInfo         gpusparseDestroyPruneInfo
#define cusparseDestroySpMat             gpusparseDestroySpMat
#define cusparseDestroySpVec             gpusparseDestroySpVec
#define cusparseDgebsr2csr               gpusparseDgebsr2csr
#define cusparseDgebsr2gebsc             gpusparseDgebsr2gebsc
#define cusparseDgebsr2gebsc_bufferSize  gpusparseDgebsr2gebsc_bufferSize
#define cusparseDgebsr2gebsr             gpusparseDgebsr2gebsr
#define cusparseDgebsr2gebsr_bufferSize  gpusparseDgebsr2gebsr_bufferSize
#define cusparseDgemmi                   gpusparseDgemmi
#define cusparseDgemvi                   gpusparseDgemvi
#define cusparseDgemvi_bufferSize        gpusparseDgemvi_bufferSize
#define cusparseDgpsvInterleavedBatch    gpusparseDgpsvInterleavedBatch
#define cusparseDgpsvInterleavedBatch_bufferSizeExt  \
        gpusparseDgpsvInterleavedBatch_bufferSizeExt
#define cusparseDgthr                    gpusparseDgthr
#define cusparseDgthrz                   gpusparseDgthrz
#define cusparseDgtsv2                   gpusparseDgtsv2
#define cusparseDgtsv2StridedBatch       gpusparseDgtsv2StridedBatch
#define cusparseDgtsv2StridedBatch_bufferSizeExt  \
        gpusparseDgtsv2StridedBatch_bufferSizeExt
#define cusparseDgtsv2_bufferSizeExt     gpusparseDgtsv2_bufferSizeExt
#define cusparseDgtsv2_nopivot           gpusparseDgtsv2_nopivot
#define cusparseDgtsv2_nopivot_bufferSizeExt  \
        gpusparseDgtsv2_nopivot_bufferSizeExt
#define cusparseDgtsvInterleavedBatch    gpusparseDgtsvInterleavedBatch
#define cusparseDgtsvInterleavedBatch_bufferSizeExt  \
        gpusparseDgtsvInterleavedBatch_bufferSizeExt
#define cusparseDiagType_t               gpusparseDiagType_t
#define cusparseDirection_t              gpusparseDirection_t
#define cusparseDnMatDescr_t             gpusparseDnMatDescr_t
#define cusparseDnMatGet                 gpusparseDnMatGet
#define cusparseDnMatGetStridedBatch     gpusparseDnMatGetStridedBatch
#define cusparseDnMatGetValues           gpusparseDnMatGetValues
#define cusparseDnMatSetStridedBatch     gpusparseDnMatSetStridedBatch
#define cusparseDnMatSetValues           gpusparseDnMatSetValues
#define cusparseDnVecDescr_t             gpusparseDnVecDescr_t
#define cusparseDnVecGet                 gpusparseDnVecGet
#define cusparseDnVecGetValues           gpusparseDnVecGetValues
#define cusparseDnVecSetValues           gpusparseDnVecSetValues
#define cusparseDnnz                     gpusparseDnnz
#define cusparseDnnz_compress            gpusparseDnnz_compress
#define cusparseDpruneCsr2csr            gpusparseDpruneCsr2csr
#define cusparseDpruneCsr2csrByPercentage  \
        gpusparseDpruneCsr2csrByPercentage
#define cusparseDpruneCsr2csrByPercentage_bufferSizeExt  \
        gpusparseDpruneCsr2csrByPercentage_bufferSizeExt
#define cusparseDpruneCsr2csrNnz         gpusparseDpruneCsr2csrNnz
#define cusparseDpruneCsr2csrNnzByPercentage  \
        gpusparseDpruneCsr2csrNnzByPercentage
#define cusparseDpruneCsr2csr_bufferSizeExt  \
        gpusparseDpruneCsr2csr_bufferSizeExt
#define cusparseDpruneDense2csr          gpusparseDpruneDense2csr
#define cusparseDpruneDense2csrByPercentage  \
        gpusparseDpruneDense2csrByPercentage
#define cusparseDpruneDense2csrByPercentage_bufferSizeExt  \
        gpusparseDpruneDense2csrByPercentage_bufferSizeExt
#define cusparseDpruneDense2csrNnz       gpusparseDpruneDense2csrNnz
#define cusparseDpruneDense2csrNnzByPercentage  \
        gpusparseDpruneDense2csrNnzByPercentage
#define cusparseDpruneDense2csr_bufferSizeExt  \
        gpusparseDpruneDense2csr_bufferSizeExt
#define cusparseDroti                    gpusparseDroti
#define cusparseDsctr                    gpusparseDsctr
#define cusparseFillMode_t               gpusparseFillMode_t
#define cusparseFormat_t                 gpusparseFormat_t
#define cusparseGather                   gpusparseGather
#define cusparseGetMatDiagType           gpusparseGetMatDiagType
#define cusparseGetMatFillMode           gpusparseGetMatFillMode
#define cusparseGetMatIndexBase          gpusparseGetMatIndexBase
#define cusparseGetMatType               gpusparseGetMatType
#define cusparseGetPointerMode           gpusparseGetPointerMode
#define cusparseGetStream                gpusparseGetStream
#define cusparseGetVersion               gpusparseGetVersion
#define cusparseHandle_t                 gpusparseHandle_t
#define cusparseIndexBase_t              gpusparseIndexBase_t
#define cusparseIndexType_t              gpusparseIndexType_t
#define cusparseMatDescr_t               gpusparseMatDescr_t
#define cusparseMatrixType_t             gpusparseMatrixType_t
#define cusparseOperation_t              gpusparseOperation_t
#define cusparseOrder_t                  gpusparseOrder_t
#define cusparsePointerMode_t            gpusparsePointerMode_t
#define cusparseRot                      gpusparseRot
#define cusparseSDDMM                    gpusparseSDDMM
#define cusparseSDDMMAlg_t               gpusparseSDDMMAlg_t
#define cusparseSDDMM_bufferSize         gpusparseSDDMM_bufferSize
#define cusparseSDDMM_preprocess         gpusparseSDDMM_preprocess
#define cusparseSaxpyi                   gpusparseSaxpyi
#define cusparseSbsr2csr                 gpusparseSbsr2csr
#define cusparseSbsric02                 gpusparseSbsric02
#define cusparseSbsric02_analysis        gpusparseSbsric02_analysis
#define cusparseSbsric02_bufferSize      gpusparseSbsric02_bufferSize
#define cusparseSbsrilu02                gpusparseSbsrilu02
#define cusparseSbsrilu02_analysis       gpusparseSbsrilu02_analysis
#define cusparseSbsrilu02_bufferSize     gpusparseSbsrilu02_bufferSize
#define cusparseSbsrilu02_numericBoost   gpusparseSbsrilu02_numericBoost
#define cusparseSbsrmm                   gpusparseSbsrmm
#define cusparseSbsrmv                   gpusparseSbsrmv
#define cusparseSbsrsm2_analysis         gpusparseSbsrsm2_analysis
#define cusparseSbsrsm2_bufferSize       gpusparseSbsrsm2_bufferSize
#define cusparseSbsrsm2_solve            gpusparseSbsrsm2_solve
#define cusparseSbsrsv2_analysis         gpusparseSbsrsv2_analysis
#define cusparseSbsrsv2_bufferSize       gpusparseSbsrsv2_bufferSize
#define cusparseSbsrsv2_bufferSizeExt    gpusparseSbsrsv2_bufferSizeExt
#define cusparseSbsrsv2_solve            gpusparseSbsrsv2_solve
#define cusparseSbsrxmv                  gpusparseSbsrxmv
#define cusparseScatter                  gpusparseScatter
#define cusparseScsc2dense               gpusparseScsc2dense
#define cusparseScsr2bsr                 gpusparseScsr2bsr
#define cusparseScsr2csr_compress        gpusparseScsr2csr_compress
#define cusparseScsr2csru                gpusparseScsr2csru
#define cusparseScsr2dense               gpusparseScsr2dense
#define cusparseScsr2gebsr               gpusparseScsr2gebsr
#define cusparseScsr2gebsr_bufferSize    gpusparseScsr2gebsr_bufferSize
#define cusparseScsrcolor                gpusparseScsrcolor
#define cusparseScsrgeam2                gpusparseScsrgeam2
#define cusparseScsrgeam2_bufferSizeExt  gpusparseScsrgeam2_bufferSizeExt
#define cusparseScsrgemm2                gpusparseScsrgemm2
#define cusparseScsrgemm2_bufferSizeExt  gpusparseScsrgemm2_bufferSizeExt
#define cusparseScsric02                 gpusparseScsric02
#define cusparseScsric02_analysis        gpusparseScsric02_analysis
#define cusparseScsric02_bufferSize      gpusparseScsric02_bufferSize
#define cusparseScsric02_bufferSizeExt   gpusparseScsric02_bufferSizeExt
#define cusparseScsrilu02                gpusparseScsrilu02
#define cusparseScsrilu02_analysis       gpusparseScsrilu02_analysis
#define cusparseScsrilu02_bufferSize     gpusparseScsrilu02_bufferSize
#define cusparseScsrilu02_bufferSizeExt  gpusparseScsrilu02_bufferSizeExt
#define cusparseScsrilu02_numericBoost   gpusparseScsrilu02_numericBoost
#define cusparseScsrsm2_analysis         gpusparseScsrsm2_analysis
#define cusparseScsrsm2_bufferSizeExt    gpusparseScsrsm2_bufferSizeExt
#define cusparseScsrsm2_solve            gpusparseScsrsm2_solve
#define cusparseScsrsv2_analysis         gpusparseScsrsv2_analysis
#define cusparseScsrsv2_bufferSize       gpusparseScsrsv2_bufferSize
#define cusparseScsrsv2_bufferSizeExt    gpusparseScsrsv2_bufferSizeExt
#define cusparseScsrsv2_solve            gpusparseScsrsv2_solve
#define cusparseScsru2csr                gpusparseScsru2csr
#define cusparseScsru2csr_bufferSizeExt  gpusparseScsru2csr_bufferSizeExt
#define cusparseSdense2csc               gpusparseSdense2csc
#define cusparseSdense2csr               gpusparseSdense2csr
#define cusparseSetMatDiagType           gpusparseSetMatDiagType
#define cusparseSetMatFillMode           gpusparseSetMatFillMode
#define cusparseSetMatIndexBase          gpusparseSetMatIndexBase
#define cusparseSetMatType               gpusparseSetMatType
#define cusparseSetPointerMode           gpusparseSetPointerMode
#define cusparseSetStream                gpusparseSetStream
#define cusparseSgebsr2csr               gpusparseSgebsr2csr
#define cusparseSgebsr2gebsc             gpusparseSgebsr2gebsc
#define cusparseSgebsr2gebsc_bufferSize  gpusparseSgebsr2gebsc_bufferSize
#define cusparseSgebsr2gebsr             gpusparseSgebsr2gebsr
#define cusparseSgebsr2gebsr_bufferSize  gpusparseSgebsr2gebsr_bufferSize
#define cusparseSgemmi                   gpusparseSgemmi
#define cusparseSgemvi                   gpusparseSgemvi
#define cusparseSgemvi_bufferSize        gpusparseSgemvi_bufferSize
#define cusparseSgpsvInterleavedBatch    gpusparseSgpsvInterleavedBatch
#define cusparseSgpsvInterleavedBatch_bufferSizeExt  \
        gpusparseSgpsvInterleavedBatch_bufferSizeExt
#define cusparseSgthr                    gpusparseSgthr
#define cusparseSgthrz                   gpusparseSgthrz
#define cusparseSgtsv2                   gpusparseSgtsv2
#define cusparseSgtsv2StridedBatch       gpusparseSgtsv2StridedBatch
#define cusparseSgtsv2StridedBatch_bufferSizeExt  \
        gpusparseSgtsv2StridedBatch_bufferSizeExt
#define cusparseSgtsv2_bufferSizeExt     gpusparseSgtsv2_bufferSizeExt
#define cusparseSgtsv2_nopivot           gpusparseSgtsv2_nopivot
#define cusparseSgtsv2_nopivot_bufferSizeExt  \
        gpusparseSgtsv2_nopivot_bufferSizeExt
#define cusparseSgtsvInterleavedBatch    gpusparseSgtsvInterleavedBatch
#define cusparseSgtsvInterleavedBatch_bufferSizeExt  \
        gpusparseSgtsvInterleavedBatch_bufferSizeExt
#define cusparseSnnz                     gpusparseSnnz
#define cusparseSnnz_compress            gpusparseSnnz_compress
#define cusparseSolvePolicy_t            gpusparseSolvePolicy_t
#define cusparseSpGEMMAlg_t              gpusparseSpGEMMAlg_t
#define cusparseSpGEMMDescr_t            gpusparseSpGEMMDescr_t
#define cusparseSpGEMM_compute           gpusparseSpGEMM_compute
#define cusparseSpGEMM_copy              gpusparseSpGEMM_copy
#define cusparseSpGEMM_createDescr       gpusparseSpGEMM_createDescr
#define cusparseSpGEMM_destroyDescr      gpusparseSpGEMM_destroyDescr
#define cusparseSpGEMM_workEstimation    gpusparseSpGEMM_workEstimation
#define cusparseSpMM                     gpusparseSpMM
#define cusparseSpMMAlg_t                gpusparseSpMMAlg_t
#define cusparseSpMM_bufferSize          gpusparseSpMM_bufferSize
#define cusparseSpMM_preprocess          gpusparseSpMM_preprocess
#define cusparseSpMV                     gpusparseSpMV
#define cusparseSpMVAlg_t                gpusparseSpMVAlg_t
#define cusparseSpMV_bufferSize          gpusparseSpMV_bufferSize
#define cusparseSpMatAttribute_t         gpusparseSpMatAttribute_t
#define cusparseSpMatDescr_t             gpusparseSpMatDescr_t
#define cusparseSpMatGetAttribute        gpusparseSpMatGetAttribute
#define cusparseSpMatGetFormat           gpusparseSpMatGetFormat
#define cusparseSpMatGetIndexBase        gpusparseSpMatGetIndexBase
#define cusparseSpMatGetSize             gpusparseSpMatGetSize
#define cusparseSpMatGetStridedBatch     gpusparseSpMatGetStridedBatch
#define cusparseSpMatGetValues           gpusparseSpMatGetValues
#define cusparseSpMatSetAttribute        gpusparseSpMatSetAttribute
#define cusparseSpMatSetStridedBatch     gpusparseSpMatSetStridedBatch
#define cusparseSpMatSetValues           gpusparseSpMatSetValues
#define cusparseSpSMAlg_t                gpusparseSpSMAlg_t
#define cusparseSpSMDescr_t              gpusparseSpSMDescr_t
#define cusparseSpSM_analysis            gpusparseSpSM_analysis
#define cusparseSpSM_bufferSize          gpusparseSpSM_bufferSize
#define cusparseSpSM_createDescr         gpusparseSpSM_createDescr
#define cusparseSpSM_destroyDescr        gpusparseSpSM_destroyDescr
#define cusparseSpSM_solve               gpusparseSpSM_solve
#define cusparseSpSVAlg_t                gpusparseSpSVAlg_t
#define cusparseSpSVDescr_t              gpusparseSpSVDescr_t
#define cusparseSpSV_analysis            gpusparseSpSV_analysis
#define cusparseSpSV_bufferSize          gpusparseSpSV_bufferSize
#define cusparseSpSV_createDescr         gpusparseSpSV_createDescr
#define cusparseSpSV_destroyDescr        gpusparseSpSV_destroyDescr
#define cusparseSpSV_solve               gpusparseSpSV_solve
#define cusparseSpVV                     gpusparseSpVV
#define cusparseSpVV_bufferSize          gpusparseSpVV_bufferSize
#define cusparseSpVecDescr_t             gpusparseSpVecDescr_t
#define cusparseSpVecGet                 gpusparseSpVecGet
#define cusparseSpVecGetIndexBase        gpusparseSpVecGetIndexBase
#define cusparseSpVecGetValues           gpusparseSpVecGetValues
#define cusparseSpVecSetValues           gpusparseSpVecSetValues
#define cusparseSparseToDense            gpusparseSparseToDense
#define cusparseSparseToDenseAlg_t       gpusparseSparseToDenseAlg_t
#define cusparseSparseToDense_bufferSize gpusparseSparseToDense_bufferSize
#define cusparseSpruneCsr2csr            gpusparseSpruneCsr2csr
#define cusparseSpruneCsr2csrByPercentage  \
        gpusparseSpruneCsr2csrByPercentage
#define cusparseSpruneCsr2csrByPercentage_bufferSizeExt  \
        gpusparseSpruneCsr2csrByPercentage_bufferSizeExt
#define cusparseSpruneCsr2csrNnz         gpusparseSpruneCsr2csrNnz
#define cusparseSpruneCsr2csrNnzByPercentage  \
        gpusparseSpruneCsr2csrNnzByPercentage
#define cusparseSpruneCsr2csr_bufferSizeExt  \
        gpusparseSpruneCsr2csr_bufferSizeExt
#define cusparseSpruneDense2csr          gpusparseSpruneDense2csr
#define cusparseSpruneDense2csrByPercentage  \
        gpusparseSpruneDense2csrByPercentage
#define cusparseSpruneDense2csrByPercentage_bufferSizeExt  \
        gpusparseSpruneDense2csrByPercentage_bufferSizeExt
#define cusparseSpruneDense2csrNnz       gpusparseSpruneDense2csrNnz
#define cusparseSpruneDense2csrNnzByPercentage  \
        gpusparseSpruneDense2csrNnzByPercentage
#define cusparseSpruneDense2csr_bufferSizeExt  \
        gpusparseSpruneDense2csr_bufferSizeExt
#define cusparseSroti                    gpusparseSroti
#define cusparseSsctr                    gpusparseSsctr
#define cusparseStatus_t                 gpusparseStatus_t
#define cusparseXbsric02_zeroPivot       gpusparseXbsric02_zeroPivot
#define cusparseXbsrilu02_zeroPivot      gpusparseXbsrilu02_zeroPivot
#define cusparseXbsrsm2_zeroPivot        gpusparseXbsrsm2_zeroPivot
#define cusparseXbsrsv2_zeroPivot        gpusparseXbsrsv2_zeroPivot
#define cusparseXcoo2csr                 gpusparseXcoo2csr
#define cusparseXcoosortByColumn         gpusparseXcoosortByColumn
#define cusparseXcoosortByRow            gpusparseXcoosortByRow
#define cusparseXcoosort_bufferSizeExt   gpusparseXcoosort_bufferSizeExt
#define cusparseXcscsort                 gpusparseXcscsort
#define cusparseXcscsort_bufferSizeExt   gpusparseXcscsort_bufferSizeExt
#define cusparseXcsr2bsrNnz              gpusparseXcsr2bsrNnz
#define cusparseXcsr2coo                 gpusparseXcsr2coo
#define cusparseXcsr2gebsrNnz            gpusparseXcsr2gebsrNnz
#define cusparseXcsrgeam2Nnz             gpusparseXcsrgeam2Nnz
#define cusparseXcsrgemm2Nnz             gpusparseXcsrgemm2Nnz
#define cusparseXcsric02_zeroPivot       gpusparseXcsric02_zeroPivot
#define cusparseXcsrilu02_zeroPivot      gpusparseXcsrilu02_zeroPivot
#define cusparseXcsrsm2_zeroPivot        gpusparseXcsrsm2_zeroPivot
#define cusparseXcsrsort                 gpusparseXcsrsort
#define cusparseXcsrsort_bufferSizeExt   gpusparseXcsrsort_bufferSizeExt
#define cusparseXcsrsv2_zeroPivot        gpusparseXcsrsv2_zeroPivot
#define cusparseXgebsr2gebsrNnz          gpusparseXgebsr2gebsrNnz
#define cusparseZaxpyi                   gpusparseZaxpyi
#define cusparseZbsr2csr                 gpusparseZbsr2csr
#define cusparseZbsric02                 gpusparseZbsric02
#define cusparseZbsric02_analysis        gpusparseZbsric02_analysis
#define cusparseZbsric02_bufferSize      gpusparseZbsric02_bufferSize
#define cusparseZbsrilu02                gpusparseZbsrilu02
#define cusparseZbsrilu02_analysis       gpusparseZbsrilu02_analysis
#define cusparseZbsrilu02_bufferSize     gpusparseZbsrilu02_bufferSize
#define cusparseZbsrilu02_numericBoost   gpusparseZbsrilu02_numericBoost
#define cusparseZbsrmm                   gpusparseZbsrmm
#define cusparseZbsrmv                   gpusparseZbsrmv
#define cusparseZbsrsm2_analysis         gpusparseZbsrsm2_analysis
#define cusparseZbsrsm2_bufferSize       gpusparseZbsrsm2_bufferSize
#define cusparseZbsrsm2_solve            gpusparseZbsrsm2_solve
#define cusparseZbsrsv2_analysis         gpusparseZbsrsv2_analysis
#define cusparseZbsrsv2_bufferSize       gpusparseZbsrsv2_bufferSize
#define cusparseZbsrsv2_bufferSizeExt    gpusparseZbsrsv2_bufferSizeExt
#define cusparseZbsrsv2_solve            gpusparseZbsrsv2_solve
#define cusparseZbsrxmv                  gpusparseZbsrxmv
#define cusparseZcsc2dense               gpusparseZcsc2dense
#define cusparseZcsr2bsr                 gpusparseZcsr2bsr
#define cusparseZcsr2csr_compress        gpusparseZcsr2csr_compress
#define cusparseZcsr2csru                gpusparseZcsr2csru
#define cusparseZcsr2dense               gpusparseZcsr2dense
#define cusparseZcsr2gebsr               gpusparseZcsr2gebsr
#define cusparseZcsr2gebsr_bufferSize    gpusparseZcsr2gebsr_bufferSize
#define cusparseZcsrcolor                gpusparseZcsrcolor
#define cusparseZcsrgeam2                gpusparseZcsrgeam2
#define cusparseZcsrgeam2_bufferSizeExt  gpusparseZcsrgeam2_bufferSizeExt
#define cusparseZcsrgemm2                gpusparseZcsrgemm2
#define cusparseZcsrgemm2_bufferSizeExt  gpusparseZcsrgemm2_bufferSizeExt
#define cusparseZcsric02                 gpusparseZcsric02
#define cusparseZcsric02_analysis        gpusparseZcsric02_analysis
#define cusparseZcsric02_bufferSize      gpusparseZcsric02_bufferSize
#define cusparseZcsric02_bufferSizeExt   gpusparseZcsric02_bufferSizeExt
#define cusparseZcsrilu02                gpusparseZcsrilu02
#define cusparseZcsrilu02_analysis       gpusparseZcsrilu02_analysis
#define cusparseZcsrilu02_bufferSize     gpusparseZcsrilu02_bufferSize
#define cusparseZcsrilu02_bufferSizeExt  gpusparseZcsrilu02_bufferSizeExt
#define cusparseZcsrilu02_numericBoost   gpusparseZcsrilu02_numericBoost
#define cusparseZcsrsm2_analysis         gpusparseZcsrsm2_analysis
#define cusparseZcsrsm2_bufferSizeExt    gpusparseZcsrsm2_bufferSizeExt
#define cusparseZcsrsm2_solve            gpusparseZcsrsm2_solve
#define cusparseZcsrsv2_analysis         gpusparseZcsrsv2_analysis
#define cusparseZcsrsv2_bufferSize       gpusparseZcsrsv2_bufferSize
#define cusparseZcsrsv2_bufferSizeExt    gpusparseZcsrsv2_bufferSizeExt
#define cusparseZcsrsv2_solve            gpusparseZcsrsv2_solve
#define cusparseZcsru2csr                gpusparseZcsru2csr
#define cusparseZcsru2csr_bufferSizeExt  gpusparseZcsru2csr_bufferSizeExt
#define cusparseZdense2csc               gpusparseZdense2csc
#define cusparseZdense2csr               gpusparseZdense2csr
#define cusparseZgebsr2csr               gpusparseZgebsr2csr
#define cusparseZgebsr2gebsc             gpusparseZgebsr2gebsc
#define cusparseZgebsr2gebsc_bufferSize  gpusparseZgebsr2gebsc_bufferSize
#define cusparseZgebsr2gebsr             gpusparseZgebsr2gebsr
#define cusparseZgebsr2gebsr_bufferSize  gpusparseZgebsr2gebsr_bufferSize
#define cusparseZgemmi                   gpusparseZgemmi
#define cusparseZgemvi                   gpusparseZgemvi
#define cusparseZgemvi_bufferSize        gpusparseZgemvi_bufferSize
#define cusparseZgpsvInterleavedBatch    gpusparseZgpsvInterleavedBatch
#define cusparseZgpsvInterleavedBatch_bufferSizeExt  \
        gpusparseZgpsvInterleavedBatch_bufferSizeExt
#define cusparseZgthr                    gpusparseZgthr
#define cusparseZgthrz                   gpusparseZgthrz
#define cusparseZgtsv2                   gpusparseZgtsv2
#define cusparseZgtsv2StridedBatch       gpusparseZgtsv2StridedBatch
#define cusparseZgtsv2StridedBatch_bufferSizeExt  \
        gpusparseZgtsv2StridedBatch_bufferSizeExt
#define cusparseZgtsv2_bufferSizeExt     gpusparseZgtsv2_bufferSizeExt
#define cusparseZgtsv2_nopivot           gpusparseZgtsv2_nopivot
#define cusparseZgtsv2_nopivot_bufferSizeExt  \
        gpusparseZgtsv2_nopivot_bufferSizeExt
#define cusparseZgtsvInterleavedBatch    gpusparseZgtsvInterleavedBatch
#define cusparseZgtsvInterleavedBatch_bufferSizeExt  \
        gpusparseZgtsvInterleavedBatch_bufferSizeExt
#define cusparseZnnz                     gpusparseZnnz
#define cusparseZnnz_compress            gpusparseZnnz_compress
#define cusparseZsctr                    gpusparseZsctr

#include <hop/hopsparse.h>

#endif
