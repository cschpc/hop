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

#ifndef __HOPBLAS_HIP_H__
#define __HOPBLAS_HIP_H__

#include <hipblas/hipblas.h>

#define gpublasStatus_t           hipblasStatus_t
#define gpublasHandle_t           hipblasHandle_t
#define gpublasOperation_t        hipblasOperation_t

#define gpublasCreate             hipblasCreate
#define gpublasDestroy            hipblasDestroy
#define gpublasSetStream          hipblasSetStream
#define gpuublasSetVector         hipblasSetVector
#define gpublasGetMatrixAsync     hipblasGetMatrixAsync
#define gpublasSetMatrixAsync     hipblasSetMatrixAsync
#define gpublasGetPointerMode     hipblasGetPointerMode
#define gpublasSetPointerMode     hipblasSetPointerMode
#define gpublasDsyrk              hipblasDsyrk
#define gpublasDsyr2k             hipblasDsyr2k
#define gpublasDscal              hipblasDscal
#define gpublasZscal              hipblasZscal
#define gpublasDgemm              hipblasDgemm
#define gpublasZgemm              hipblasZgemm
#define gpublasDgemv              hipblasDgemv
#define gpublasZgemv              hipblasZgemv
#define gpublasDaxpy              hipblasDaxpy
#define gpublasZaxpy              hipblasZaxpy
#define gpublasZherk              hipblasZherk
#define gpublasZher2k             hipblasZher2k
#define gpublasDdot               hipblasDdot
#define gpublasZdotc              hipblasZdotc
#define gpublasZdotu              hipblasZdotu

#define GPUBLAS_OP_N                     HIPBLAS_OP_N
#define GPUBLAS_OP_T                     HIPBLAS_OP_T
#define GPUBLAS_OP_C                     HIPBLAS_OP_C
#define GPUBLAS_FILL_MODE_UPPER          HIPBLAS_FILL_MODE_UPPER
#define GPUBLAS_SIDE_LEFT                HIPBLAS_SIDE_LEFT
#define GPUBLAS_SIDE_RIGHT               HIPBLAS_SIDE_RIGHT
#define GPUBLAS_STATUS_SUCCESS           HIPBLAS_STATUS_SUCCESS
#define GPUBLAS_STATUS_NOT_INITIALIZED   HIPBLAS_STATUS_NOT_INITIALIZED
#define GPUBLAS_STATUS_ALLOC_FAILED      HIPBLAS_STATUS_ALLOC_FAILED
#define GPUBLAS_STATUS_INVALID_VALUE     HIPBLAS_STATUS_INVALID_VALUE
#define GPUBLAS_STATUS_ARCH_MISMATCH     HIPBLAS_STATUS_ARCH_MISMATCH
#define GPUBLAS_STATUS_MAPPING_ERROR     HIPBLAS_STATUS_MAPPING_ERROR
#define GPUBLAS_STATUS_EXECUTION_FAILED  HIPBLAS_STATUS_EXECUTION_FAILED
#define GPUBLAS_STATUS_INTERNAL_ERROR    HIPBLAS_STATUS_INTERNAL_ERROR

#endif
