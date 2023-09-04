import torch
from lietorch import SO3

phi = torch.randn(8000, 3, device='cuda', requires_grad=True)
R = SO3.exp(phi)

# relative rotation matrix, SO3 ^ {8000 x 8000}
dR = R[:,None].inv() * R[None,:]

# 8000x8000 matrix of angles
ang = dR.log().norm(dim=-1)

# backpropogation in tangent space
loss = ang.sum()
loss.backward()
print(loss)