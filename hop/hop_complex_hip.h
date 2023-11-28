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

#ifndef __HOP_HOP_COMPLEX_HIP_H__
#define __HOP_HOP_COMPLEX_HIP_H__

#include <hip/hip_complex.h>

#define gpuCabs                          hipCabs
#define gpuCabsf                         hipCabsf
#define gpuCadd                          hipCadd
#define gpuCaddf                         hipCaddf
#define gpuCdiv                          hipCdiv
#define gpuCdivf                         hipCdivf
#define gpuCfma                          hipCfma
#define gpuCfmaf                         hipCfmaf
#define gpuCimag                         hipCimag
#define gpuCimagf                        hipCimagf
#define gpuCmul                          hipCmul
#define gpuCmulf                         hipCmulf
#define gpuComplex                       hipComplex
#define gpuComplexDoubleToFloat          hipComplexDoubleToFloat
#define gpuComplexFloatToDouble          hipComplexFloatToDouble
#define gpuConj                          hipConj
#define gpuConjf                         hipConjf
#define gpuCreal                         hipCreal
#define gpuCrealf                        hipCrealf
#define gpuCsub                          hipCsub
#define gpuCsubf                         hipCsubf
#define gpuDoubleComplex                 hipDoubleComplex
#define gpuFloatComplex                  hipFloatComplex
#define make_gpuComplex                  make_hipComplex
#define make_gpuDoubleComplex            make_hipDoubleComplex
#define make_gpuFloatComplex             make_hipFloatComplex


#endif
