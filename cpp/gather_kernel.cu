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
__global__ void kernel(torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> a,
                torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> b,
              torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> out,
            int k){
  int x = blockIdx.x;
  int y = threadIdx.x;
  // assert(a[x][y]!=0);
  float s = 0;
  for(int i=0;i<k;i++){
    s = s + a[b[x][i]][y];
  }
  out[x][y] = s;
}


torch::Tensor gather_reduce_cuda(torch::Tensor a,torch::Tensor b){

  auto shape_a = a.sizes();
  auto shape_b = b.sizes();
  auto out = torch::zeros(shape_a, torch::kFloat).to(torch::kCUDA);

  auto s1 = shape_a[0];
  auto s2 = shape_a[1];
  int s3 = shape_b[0];
  int s4 = shape_b[1];


  // AT_DISPATCH_FLOATING_TYPES(torch::ScalarType::Double, "lltm_forward_cuda", [&]{ lltm_cuda_forward_kernel<scalar_t><<<10, 10 >>> (out.data<scalar_t>());});
  AT_DISPATCH_FLOATING_TYPES(torch::ScalarType::Float, "gather_reduce_cuda", [&]{
     kernel<scalar_t><<<s1, s2 >>> (
       a.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        b.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
        out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),s4);});


  // AT_DISPATCH_FLOATING_TYPES(torch::ScalarType::Double, "gather_reduce_cuda", ([&] {
  //   lltm_cuda_forward_kernel<scalar_t><<<10, 10 >>>(out.data<scalar_t>);
  // }));
  return out;
}
