import sys

import torch
from safetensors.torch import save_file

print(sys.argv)
sd = torch.load(sys.argv[1])
print(sd.keys())
save_file({k: sd[k].float() for k in sd.keys()}, sys.argv[2])
