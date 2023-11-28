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

#ifndef __HOP_SOURCE_HIP_HIP_COMPLEX_H__
#define __HOP_SOURCE_HIP_HIP_COMPLEX_H__

#define HOP_SOURCE_HIP

#define hipCabs                          gpuCabs
#define hipCabsf                         gpuCabsf
#define hipCadd                          gpuCadd
#define hipCaddf                         gpuCaddf
#define hipCdiv                          gpuCdiv
#define hipCdivf                         gpuCdivf
#define hipCfma                          gpuCfma
#define hipCfmaf                         gpuCfmaf
#define hipCimag                         gpuCimag
#define hipCimagf                        gpuCimagf
#define hipCmul                          gpuCmul
#define hipCmulf                         gpuCmulf
#define hipComplex                       gpuComplex
#define hipComplexDoubleToFloat          gpuComplexDoubleToFloat
#define hipComplexFloatToDouble          gpuComplexFloatToDouble
#define hipConj                          gpuConj
#define hipConjf                         gpuConjf
#define hipCreal                         gpuCreal
#define hipCrealf                        gpuCrealf
#define hipCsub                          gpuCsub
#define hipCsubf                         gpuCsubf
#define hipDoubleComplex                 gpuDoubleComplex
#define hipFloatComplex                  gpuFloatComplex
#define make_hipComplex                  make_gpuComplex
#define make_hipDoubleComplex            make_gpuDoubleComplex
#define make_hipFloatComplex             make_gpuFloatComplex

#include <hop/hop_complex.h>

#endif
