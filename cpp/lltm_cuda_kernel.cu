#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


// template <typename scalar_t>
// __global__ void lltm_cuda_forward_kernel(torch::PackedTensorAccessor32<scalar_t
//             ,2,torch::RestrictPtrTraits> output_gate){
//   int x = blockIdx.x;
//   int y = threadIdx.x;
//   output_gate[x][y] = output_gate[x][y] + 1;
//
//
// }

template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cel){
  int x = blockIdx.x;
  int y = threadIdx.x;
  candidate_cel[x][y]=1;
}


torch::Tensor gather_reduce_cuda(torch::Tensor a,torch::Tensor b){

  auto shape_a = a.sizes();
  auto shape_b = b.sizes();
  auto out = torch::zeros(shape_a, torch::kDouble);

  auto s1 = shape_a[0];
  auto s2 = shape_a[1];
  auto s3 = shape_a[0];
  auto s4 = shape_b[1];

  for(int i=0;i<s1;i++){
    for(int j=0;j<s2;j++){
      out.index_put_({i,j},out.index({i,j})+1);
    }
  }

  // AT_DISPATCH_FLOATING_TYPES(torch::ScalarType::Double, "lltm_forward_cuda", [&]{ lltm_cuda_forward_kernel<scalar_t><<<10, 10 >>> (out.data<scalar_t>());});
  AT_DISPATCH_FLOATING_TYPES(torch::ScalarType::Double, "gather_reduce_cuda", [&]{
     lltm_cuda_forward_kernel<scalar_t><<<s1, s2 >>> (out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());});

  // AT_DISPATCH_FLOATING_TYPES(torch::ScalarType::Double, "gather_reduce_cuda", ([&] {
  //   lltm_cuda_forward_kernel<scalar_t><<<10, 10 >>>(out.data<scalar_t>);
  // }));
  // return out;
}
