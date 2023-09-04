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

#ifndef __HOP_SOURCE_HIPBLAS_H__
#define __HOP_SOURCE_HIPBLAS_H__

#define HOP_SOURCE_HIP

#define hipblasStatus_t          gpublasStatus_t
#define hipblasHandle_t          gpublasHandle_t
#define hipblasOperation_t       gpublasOperation_t

#define hipblasCreate            gpublasCreate
#define hipblasDestroy           gpublasDestroy
#define hipblasSetStream         gpublasSetStream
#define hipblasSetVector         gpublasSetVector
#define hipblasGetMatrixAsync    gpublasGetMatrixAsync
#define hipblasSetMatrixAsync    gpublasSetMatrixAsync
#define hipblasGetPointerMode    gpublasGetPointerMode
#define hipblasSetPointerMode    gpublasSetPointerMode
#define hipblasDsyrk             gpublasDsyrk
#define hipblasDsyr2k            gpublasDsyr2k
#define hipblasDscal             gpublasDscal
#define hipblasZscal             gpublasZscal
#define hipblasDgemm             gpublasDgemm
#define hipblasZgemm             gpublasZgemm
#define hipblasDgemv             gpublasDgemv
#define hipblasZgemv             gpublasZgemv
#define hipblasDaxpy             gpublasDaxpy
#define hipblasZaxpy             gpublasZaxpy
#define hipblasZherk             gpublasZherk
#define hipblasZher2k            gpublasZher2k
#define hipblasDdot              gpublasDdot
#define hipblasZdotc             gpublasZdotc
#define hipblasZdotu             gpublasZdotu

#define HIPBLAS_OP_N                     GPUBLAS_OP_N
#define HIPBLAS_OP_T                     GPUBLAS_OP_T
#define HIPBLAS_OP_C                     GPUBLAS_OP_C
#define HIPBLAS_FILL_MODE_UPPER          GPUBLAS_FILL_MODE_UPPER
#define HIPBLAS_SIDE_LEFT                GPUBLAS_SIDE_LEFT
#define HIPBLAS_SIDE_RIGHT               GPUBLAS_SIDE_RIGHT
#define HIPBLAS_STATUS_SUCCESS           GPUBLAS_STATUS_SUCCESS
#define HIPBLAS_STATUS_NOT_INITIALIZED   GPUBLAS_STATUS_NOT_INITIALIZED
#define HIPBLAS_STATUS_ALLOC_FAILED      GPUBLAS_STATUS_ALLOC_FAILED
#define HIPBLAS_STATUS_INVALID_VALUE     GPUBLAS_STATUS_INVALID_VALUE
#define HIPBLAS_STATUS_ARCH_MISMATCH     GPUBLAS_STATUS_ARCH_MISMATCH
#define HIPBLAS_STATUS_MAPPING_ERROR     GPUBLAS_STATUS_MAPPING_ERROR
#define HIPBLAS_STATUS_EXECUTION_FAILED  GPUBLAS_STATUS_EXECUTION_FAILED
#define HIPBLAS_STATUS_INTERNAL_ERROR    GPUBLAS_STATUS_INTERNAL_ERROR

#include <hop/hopblas.h>

#endif
