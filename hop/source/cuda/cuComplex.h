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

#ifndef __HOP_SOURCE_CUDA_CUCOMPLEX_H__
#define __HOP_SOURCE_CUDA_CUCOMPLEX_H__

#if !defined(HOP_SOURCE_CUDA)
#define HOP_SOURCE_CUDA
#endif

#define cuCabs                           gpuCabs
#define cuCabsf                          gpuCabsf
#define cuCadd                           gpuCadd
#define cuCaddf                          gpuCaddf
#define cuCdiv                           gpuCdiv
#define cuCdivf                          gpuCdivf
#define cuCfma                           gpuCfma
#define cuCfmaf                          gpuCfmaf
#define cuCimag                          gpuCimag
#define cuCimagf                         gpuCimagf
#define cuCmul                           gpuCmul
#define cuCmulf                          gpuCmulf
#define cuComplex                        gpuComplex
#define cuComplexDoubleToFloat           gpuComplexDoubleToFloat
#define cuComplexFloatToDouble           gpuComplexFloatToDouble
#define cuConj                           gpuConj
#define cuConjf                          gpuConjf
#define cuCreal                          gpuCreal
#define cuCrealf                         gpuCrealf
#define cuCsub                           gpuCsub
#define cuCsubf                          gpuCsubf
#define cuDoubleComplex                  gpuDoubleComplex
#define cuFloatComplex                   gpuFloatComplex
#define make_cuComplex                   make_gpuComplex
#define make_cuDoubleComplex             make_gpuDoubleComplex
#define make_cuFloatComplex              make_gpuFloatComplex

#include <hop/hop_complex.h>

#endif
