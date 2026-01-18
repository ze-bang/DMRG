//
// GPU Tensor Implementation (HIP/ROCm)
//

#include "idmrg/gpu/gpu_tensor.h"

#ifdef IDMRG_USE_HIP

namespace idmrg {
namespace gpu {

// Explicit template instantiations
template class GPUTensor<double>;
template class GPUTensor<float>;
template class GPUMatrix<double>;
template class GPUMatrix<float>;

} // namespace gpu
} // namespace idmrg

#endif // IDMRG_USE_HIP
