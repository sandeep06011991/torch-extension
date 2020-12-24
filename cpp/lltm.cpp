

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <omp.h>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

torch::Tensor gather_reduce(torch::Tensor a,torch::Tensor b){

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
  return out;
}


PYBIND11_MODULE(lltm_cpp, m) {
  m.def("forward", &gather_reduce, "LLTM forward");
}
