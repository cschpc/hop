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

#ifndef __HOPBLAS_CUDA_H__
#define __HOPBLAS_CUDA_H__

#include <cublas.h>

#define gpublasStatus_t           cublasStatus_t
#define gpublasHandle_t           cublasHandle_t
#define gpublasOperation_t        cublasOperation_t

#define gpublasCreate             cublasCreate
#define gpublasDestroy            hipblasDestroy
#define gpublasSetStream          cublasSetStream
#define gpuublasSetVector         cublasSetVector
#define gpublasGetMatrixAsync     cublasGetMatrixAsync
#define gpublasSetMatrixAsync     cublasSetMatrixAsync
#define gpublasGetPointerMode     cublasGetPointerMode
#define gpublasSetPointerMode     cublasSetPointerMode
#define gpublasDsyrk              cublasDsyrk
#define gpublasDsyr2k             cublasDsyr2k
#define gpublasDscal              cublasDscal
#define gpublasZscal              cublasZscal
#define gpublasDgemm              cublasDgemm
#define gpublasZgemm              cublasZgemm
#define gpublasDgemv              cublasDgemv
#define gpublasZgemv              cublasZgemv
#define gpublasDaxpy              cublasDaxpy
#define gpublasZaxpy              cublasZaxpy
#define gpublasZherk              cublasZherk
#define gpublasZher2k             cublasZher2k
#define gpublasDdot               cublasDdot
#define gpublasZdotc              cublasZdotc
#define gpublasZdotu              cublasZdotu

#define GPUBLAS_OP_N                     CUBLAS_OP_N
#define GPUBLAS_OP_T                     CUBLAS_OP_T
#define GPUBLAS_OP_C                     CUBLAS_OP_C
#define GPUBLAS_FILL_MODE_UPPER          CUBLAS_FILL_MODE_UPPER
#define GPUBLAS_SIDE_LEFT                CUBLAS_SIDE_LEFT
#define GPUBLAS_SIDE_RIGHT               CUBLAS_SIDE_RIGHT
#define GPUBLAS_STATUS_SUCCESS           CUBLAS_STATUS_SUCCESS
#define GPUBLAS_STATUS_NOT_INITIALIZED   CUBLAS_STATUS_NOT_INITIALIZED
#define GPUBLAS_STATUS_ALLOC_FAILED      CUBLAS_STATUS_ALLOC_FAILED
#define GPUBLAS_STATUS_INVALID_VALUE     CUBLAS_STATUS_INVALID_VALUE
#define GPUBLAS_STATUS_ARCH_MISMATCH     CUBLAS_STATUS_ARCH_MISMATCH
#define GPUBLAS_STATUS_MAPPING_ERROR     CUBLAS_STATUS_MAPPING_ERROR
#define GPUBLAS_STATUS_EXECUTION_FAILED  CUBLAS_STATUS_EXECUTION_FAILED
#define GPUBLAS_STATUS_INTERNAL_ERROR    CUBLAS_STATUS_INTERNAL_ERROR

#endif
