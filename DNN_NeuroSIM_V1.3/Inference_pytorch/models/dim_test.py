import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

if __name__ == "__main__":
    # a = F.conv2d(3,5,stride=(1,0),kernel_size=1)

    b = torch.arange(1.0*4*768).reshape(1,4,768)
    fileter = nn.Parameter(torch.arange(1.0*768*768).reshape(768,768))
    c = F.linear(b,fileter)
    print(c)