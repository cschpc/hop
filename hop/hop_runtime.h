#if defined(HOP_TARGET_HIP)
#include "hop_runtime_hip.h"

#elif defined(HOP_TARGET_CUDA)
#include "hop_runtime_cuda.h"

#else
#error HOP target undefined (cf. HOP_TARGET_*)

#endif
