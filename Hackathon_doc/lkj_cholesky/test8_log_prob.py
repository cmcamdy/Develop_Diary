
import paddle
import lkj_cholesky


# sample_paddle = paddle.to_tensor([[1.        , 0.        , 0.        ],
#                                     [0.70339406, 0.71080011, 0.        ],
#                                     [0.15145791, 0.78267860, 0.60371739]])

sample_paddle = paddle.to_tensor([[[ 1.        ,  0.        ,  0.        ],
                                [-0.25003403,  0.96823704,  0.        ],
                                [ 0.07735625,  0.58252847,  0.80912089]],

                                [[ 1.        ,  0.        ,  0.        ],
                                [-0.53712130,  0.84350502,  0.        ],
                                [ 0.92916626,  0.00134969,  0.36965963]]])

# [-1.57624304, -0.98245108]
# [-0.03227834, -0.17018943]

# lkj_paddle = lkj_cholesky.LKJCholesky(3, 1.0)
lkj_paddle = lkj_cholesky.LKJCholesky(3, [1.0, 1.5])
res_paddle = lkj_paddle.log_prob(sample_paddle)

print(res_paddle)