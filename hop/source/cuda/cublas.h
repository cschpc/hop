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

#ifndef __HOP_SOURCE_CUBLAS_H__
#define __HOP_SOURCE_CUBLAS_H__

#define HOP_SOURCE_CUDA

#define cublasStatus_t           gpublasStatus_t
#define cublasHandle_t           gpublasHandle_t
#define cublasOperation_t        gpublasOperation_t

#define cublasCreate             gpublasCreate
#define cublasDestroy            gpublasDestroy
#define cublasSetStream          gpublasSetStream
#define cublasGetMatrixAsync     gpublasGetMatrixAsync
#define cublasSetMatrixAsync     gpublasSetMatrixAsync
#define cublasDsyrk              gpublasDsyrk
#define cublasDsyr2k             gpublasDsyr2k
#define cublasDscal              gpublasDscal
#define cublasZscal              gpublasZscal
#define cublasDgemm              gpublasDgemm
#define cublasZgemm              gpublasZgemm
#define cublasDgemv              gpublasDgemv
#define cublasZgemv              gpublasZgemv
#define cublasDaxpy              gpublasDaxpy
#define cublasZaxpy              gpublasZaxpy
#define cublasZherk              gpublasZherk
#define cublasZher2k             gpublasZher2k
#define cublasDdot               gpublasDdot
#define cublasZdotc              gpublasZdotc
#define cublasZdotu              gpublasZdotu

#define CUBLAS_OP_N                     GPUBLAS_OP_N
#define CUBLAS_OP_T                     GPUBLAS_OP_T
#define CUBLAS_OP_C                     GPUBLAS_OP_C
#define CUBLAS_FILL_MODE_UPPER          GPUBLAS_FILL_MODE_UPPER
#define CUBLAS_STATUS_SUCCESS           GPUBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED   GPUBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED      GPUBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE     GPUBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH     GPUBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR     GPUBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED  GPUBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR    GPUBLAS_STATUS_INTERNAL_ERROR

#include <hop/hopblas.h>

#endif
