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

#ifndef __HOPFFT_HIP_H__
#define __HOPFFT_HIP_H__

#define gpufftDoubleComplex      hipfftDoubleComplex
#define gpufftDoubleReal         hipfftDoubleReal
#define gpufftHandle             hipfftHandle
#define gpufftExecD2Z            hipfftExecD2Z
#define gpufftExecZ2D            hipfftExecZ2D
#define gpufftDestroy            hipfftDestroy
#define gpufftPlan2d             hipfftPlan2d

#define GPUFFT_D2Z               HIPFFT_D2Z
#define GPUFFT_Z2D               HIPFFT_Z2D

#include <hipfft.h>

#endif
