//
// GPU SVD Implementation (HIP/ROCm)
//

#include "idmrg/gpu/gpu_svd.h"

#ifdef IDMRG_USE_HIP

namespace idmrg {
namespace gpu {

// SVD implementations are in the header as template specializations
// This file is for HIP-specific kernel implementations if needed

} // namespace gpu
} // namespace idmrg

#endif // IDMRG_USE_HIP
