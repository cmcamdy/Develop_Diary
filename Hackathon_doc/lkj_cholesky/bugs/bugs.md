## AttributeError: 'Variable' object has no attribute 'fill_'
- 静态图出现的错误
```
======================================================================
ERROR: test_sample_shape (__main__.TestLKJCholeskyShape0.3-onion.CPUPlace)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_distribution_lkj_cholesky_static.py", line 148, in test_sample_shape
    fetch_list=self._paddle_lkj_cholesky.sample(case.get('input')),
  File "/home/cmcandy/code/PD/Develop_Diary/Hackathon_doc/lkj_cholesky/lkj_cholesky.py", line 257, in sample
    return self._onion(sample_shape)
  File "/home/cmcandy/code/PD/Develop_Diary/Hackathon_doc/lkj_cholesky/lkj_cholesky.py", line 199, in _onion
    u_hypersphere[..., 0, :].fill_(0.0)
AttributeError: 'Variable' object has no attribute 'fill_'

======================================================================
ERROR: test_sample_shape (__main__.TestLKJCholeskyShape0.3-onion.CUDAPlace)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_distribution_lkj_cholesky_static.py", line 148, in test_sample_shape
    fetch_list=self._paddle_lkj_cholesky.sample(case.get('input')),
  File "/home/cmcandy/code/PD/Develop_Diary/Hackathon_doc/lkj_cholesky/lkj_cholesky.py", line 257, in sample
    return self._onion(sample_shape)
  File "/home/cmcandy/code/PD/Develop_Diary/Hackathon_doc/lkj_cholesky/lkj_cholesky.py", line 199, in _onion
    u_hypersphere[..., 0, :].fill_(0.0)
AttributeError: 'Variable' object has no attribute 'fill_'

----------------------------------------------------------------------
Ran 2 tests in 0.157s


```