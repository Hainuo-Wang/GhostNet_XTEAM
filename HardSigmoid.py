import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


if __name__ == '__main__':
    input = list(range(-10, 10))
    input = torch.tensor(input)
    opt = hard_sigmoid(input)
    print(opt)
    plt.plot(input, opt)
    plt.show()
