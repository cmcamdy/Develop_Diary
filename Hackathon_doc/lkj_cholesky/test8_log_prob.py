import torch
from torch.distributions import LKJCholesky





lkj = LKJCholesky(dim = 3, concentration=1.0, validate_args=False)
# sample = torch.tensor( [[ 1.        ,  0.        ,  0.        ],
#             [-0.40908989,  0.91249406,  0.        ],
#             [ 0.39595827,  0.28080371,  0.87428045]])
sample = torch.tensor([[1.        , 0.        , 0.        ],
                        [0.56376445, 0.82593566, 0.        ],
                        [0.00589490, 0.76600456, 0.64280802],
                        [1.        , 1.        , 1.        ]])
print(lkj.log_prob(sample))