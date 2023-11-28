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

#ifndef __HOP_SOURCE_CUDA_LIBRARY_TYPES_H__
#define __HOP_SOURCE_CUDA_LIBRARY_TYPES_H__

#define HOP_SOURCE_CUDA

#define CUDA_C_16BF                      GPUBLAS_C_16B
#define CUDA_C_16F                       GPUBLAS_C_16F
#define CUDA_C_32F                       GPUBLAS_C_32F
#define CUDA_C_32I                       GPUBLAS_C_32I
#define CUDA_C_32U                       GPUBLAS_C_32U
#define CUDA_C_64F                       GPUBLAS_C_64F
#define CUDA_C_8I                        GPUBLAS_C_8I
#define CUDA_C_8U                        GPUBLAS_C_8U
#define CUDA_R_16BF                      GPUBLAS_R_16B
#define CUDA_R_16F                       GPUBLAS_R_16F
#define CUDA_R_32F                       GPUBLAS_R_32F
#define CUDA_R_32I                       GPUBLAS_R_32I
#define CUDA_R_32U                       GPUBLAS_R_32U
#define CUDA_R_64F                       GPUBLAS_R_64F
#define CUDA_R_8I                        GPUBLAS_R_8I
#define CUDA_R_8U                        GPUBLAS_R_8U
#define cudaDataType                     gpublasDatatype_t
#define cudaDataType_t                   gpublasDatatype_t

#if !defined(HOP_OVERRIDE_LIBRARY_TYPES)
#include <hop/hopblas.h>
#endif

#endif
