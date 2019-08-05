import torch
# would be xi^2/sum(xi), not robust
A=torch.Tensor([[[[1, 2, 3, 4],
          [1, 2, 3, 4],
          [1, 2, 3, 4],
          [1, 2, 3, 4]],

         [[1, 2, 3, 4],
          [1, 2, 3, 4],
          [1, 2, 3, 4],
          [1, 2, 3, 4]]]])
print(A.size())
all = torch.sum(A,dim=-1)
all = torch.sum(all,dim=-1)
all = all.unsqueeze(-1).unsqueeze(-1)
print(all.size())
print(all)
print(A/all)