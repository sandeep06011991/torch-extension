import sys
sys.path.append("../cpp")
import torch, time
import custom_kernels

# s1,s2 - shape of A matrix
# s3 - size of gather matrix.
def naive(s1,s2,s3,device):
  a = torch.rand((s1,s2)).to(device)
  b = torch.randint(0,s1,(s1,s3)).to(device)
  s = torch.zeros((s1,s2)).to(device)
  ones = torch.ones((s1,s2)).to(device).int()
  if(device == 'cuda'):
      torch.cuda.synchronize(device)
  s_time = time.time()
  for i in range(s3):
      b_i = b[:,i].reshape(s1,1) * ones
      s = s + torch.gather(a,0,b_i)
  if(device == 'cuda'):
      torch.cuda.synchronize(device)
  e_time = time.time()
  # validation script
  for i in range(s1):
      t = torch.zeros((s1,s2)).to(device)
      for k in range(s3):
          t = t + a[b[i][k],:]
      assert(torch.all(torch.eq(t,s[i,:])))
  print("Success !!")
  print("total time {}".format(e_time - s_time))

def custom_cpp_operator(s1,s2,s3):
    a = torch.rand((s1,s2))
    # a = torch.ones((s1,s2))
    # b = torch.ones((s1,s3)).int()
    b = torch.randint(0,s1,(s1,s3))
    s_time = time.time()
    s = custom_kernels.cpu_gather(a,b)
    e_time = time.time()
    # validation scriptself
    for i in range(s1):
        t = torch.zeros((s2)).float()
        for k in range(s3):
            t = t + a[b[i][k],:]
        assert(torch.all(torch.eq(t,s[i,:])))
    print("Success !!")
    print("total time for custom cpu operator {}".format(e_time - s_time))

def custom_cuda_operator(s1,s2,s3):
    a = torch.rand((s1,s2)).to('cuda')
    b = torch.randint(0,s1,(s1,s3)).to('cuda')
    s_time = time.time()
    s = custom_kernels.gpu_gather(a,b)
    torch.cuda.synchronize()
    e_time = time.time()
    s = s.to('cpu')
    # validation scriptself
    for i in range(s1):
        t = torch.zeros((s2)).float()
        for k in range(s3):
            t = t + a[b[i][k],:]
        assert(torch.all(torch.eq(t,s[i,:])))
    print("Success !!")
    print("total time for custom gpu operator {}".format(e_time - s_time))


# naive(1000,100,1000,'cuda')
# naive(1000,100,1000,'cpu')
custom_cpp_operator(5,5,5)
