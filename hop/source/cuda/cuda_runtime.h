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

#ifndef __HOP_SOURCE_CUDA_CUDA_RUNTIME_H__
#define __HOP_SOURCE_CUDA_CUDA_RUNTIME_H__

#if !defined(HOP_SOURCE_CUDA)
#define HOP_SOURCE_CUDA
#endif

#include <cuda_runtime_api.h>
#include <library_types.h>

#define cudaOccupancyMaxPotentialBlockSize  \
        gpuOccupancyMaxPotentialBlockSize
#define cudaOccupancyMaxPotentialBlockSizeVariableSMem  \
        gpuOccupancyMaxPotentialBlockSizeVariableSMem
#define cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags  \
        gpuOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
#define cudaOccupancyMaxPotentialBlockSizeWithFlags  \
        gpuOccupancyMaxPotentialBlockSizeWithFlags

#include <hop/hop_runtime.h>

#endif
