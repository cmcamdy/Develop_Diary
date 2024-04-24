import torch
from torch.distributions import LKJCholesky





lkj = LKJCholesky(dim = 3, concentration=torch.tensor([0.58801454, 0.6991087 ]), validate_args=False)

# sample = torch.tensor([[1.        , 0.        , 0.        ],
#                         [0.56376445, 0.82593566, 0.        ],
#                         [0.00589490, 0.76600456, 0.64280802],
#                         [1.        , 1.        , 1.        ]])

sample = torch.tensor([[[ 1.        ,  0.        ,  0.        ],
                        [-0.25003403,  0.96823704,  0.        ],
                        [ 0.07735625,  0.58252847,  0.80912089]],

                        [[ 1.        ,  0.        ,  0.        ],
                        [-0.53712130,  0.84350502,  0.        ],
                        [ 0.92916626,  0.00134969,  0.36965963]]])
# [-1.57624304, -0.98245108]
# [-0.03227834, -0.17018943]
print(lkj.log_prob(sample))