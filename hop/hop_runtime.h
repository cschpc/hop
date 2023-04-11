#if defined(HOP_SOURCE_HIP)
#include "source/hip/hip/hip_runtime.h"

#elif defined(HOP_SOURCE_CUDA)
#include "source/cuda/cuda_runtime.h"

#endif

#if defined(HOP_TARGET_HIP)
#include "hop_runtime_hip.h"

#elif defined(HOP_TARGET_CUDA)
#include "hop_runtime_cuda.h"

#else
#error HOP target undefined (cf. HOP_TARGET_*)

#endif
