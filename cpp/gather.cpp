

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>



torch::Tensor gather_reduce(torch::Tensor a,torch::Tensor b){

  auto shape_a = a.sizes();
  auto shape_b = b.sizes();
  auto out = torch::zeros(shape_a, torch::kFloat);

  auto s1 = shape_a[0];
  auto s2 = shape_a[1];

  auto s3 = shape_b[0];
  auto s4 = shape_b[1];
  assert(s1 == s3);

  for(int i=0;i<s1;i++){
    for(int k=0;k<s4;k++){
      for(int j=0;j<s2;j++){
        out.index_put_({i,j},out.index({i,j}) + a.index({b.index({i,k}),j}));
      }
    }
  }
  return out;
}


PYBIND11_MODULE(c_kernel, m) {
  m.def("cpu_gather", &gather_reduce, "gather_reduce");
}
