import torch
from torch.distributions import LKJCholesky

# lkj = LKJCholesky(dim = 3, concentration=torch.tensor([0.58801454, 0.6991087 ]), validate_args=False)
# lkj = LKJCholesky(dim = 3, concentration=torch.tensor([1.0]), validate_args=False)
lkj = LKJCholesky(dim = 3, concentration=torch.tensor([1.0, 1.5]), validate_args=False)

# sample = torch.tensor([[1.        , 0.        , 0.        ],
#                         [0.70339406, 0.71080011, 0.        ],
#                         [0.15145791, 0.78267860, 0.60371739]])


sample = torch.tensor([[[ 1.        ,  0.        ,  0.        ],
                        [-0.25003403,  0.96823704,  0.        ],
                        [ 0.07735625,  0.58252847,  0.80912089]],

                        [[ 1.        ,  0.        ,  0.        ],
                        [-0.53712130,  0.84350502,  0.        ],
                        [ 0.92916626,  0.00134969,  0.36965963]]])

res = lkj.log_prob(sample)
print(res)
