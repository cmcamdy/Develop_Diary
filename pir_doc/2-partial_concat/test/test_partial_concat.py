import numpy as np


def np_partial_concat(inputs, start, length):
    assert len(inputs[0].shape) == 2
    size = inputs[0].shape[1]
    assert start >= -size and start < size

    if start < 0:
        start += size
    if length < 0:
        length = size - start
    assert size >= start + length

    elems = []
    for elem in inputs:
        assert elem.shape == inputs[0].shape
        elems.append(elem[:, start : start + length])
    res = np.concatenate(elems, axis=1)
    return np.concatenate(elems, axis=1)



arr1 = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [5, 6, 7, 8]])


print(arr1.shape)
start = -5
length = -1
result1 = np_partial_concat([arr1, arr1, arr1], start, length)
print("Result 1:")
print(result1.shape)

'''
首先，每个数据的个数
batch 层面，concat前后不变
但是colnum层面，长度的计算公式为：num_var * length
'''