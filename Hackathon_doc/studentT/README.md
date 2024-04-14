
## 调研


学生 t 分布的概率密度函数（probability density function, PDF）可以用以下公式表示：

$$
f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi} \cdot \Gamma(\frac{\nu}{2})} \cdot \frac{1}{\left(1 + \frac{x^2}{\nu}\right)^{\frac{\nu+1}{2}}}
$$

其中，

- $x$ 是随机变量的取值；
- $\nu$ 是自由度（degrees of freedom）；
- $\Gamma(z)$ 是伽马函数（Gamma function），它是阶乘在实数和复数上的推广。

伽马函数的定义如下：

$$
\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt
$$

对于正整数 $n$，伽马函数满足以下关系：

$$
\Gamma(n) = (n - 1)!
$$

在实际应用中，我们通常使用标准化的学生 t 分布，即均值为 0，方差为 $\frac{\nu}{\nu - 2}$（$\nu > 2$ 时成立）。此时，概率密度函数的公式可以简化为：

$$
f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi} \cdot \Gamma(\frac{\nu}{2})} \cdot \frac{1}{\left(1 + \frac{x^2}{\nu}\right)^{\frac{\nu+1}{2}}}
$$

这里的 $x$ 是标准化后的随机变量取值。





### pytorch：
- [代码](https://github.com/pytorch/pytorch/blob/9bb54c7f3c048fdf2e5294e9e49e3642f1de98d8/torch/distributions/studentT.py#L12)
- [文档](https://pytorch.org/docs/stable/distributions.html#torch.distributions.studentT.StudentT)

<!-- ### numpy
- [代码](https://github.com/numpy/numpy/blob/ab7649fe2ed8f0f0260322d66631b8dfab57deff/numpy/random/mtrand.pyx#L4080)
- [文档](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html#) -->


### scipy

- [代码](https://github.com/scipy/scipy/blob/v1.13.0/scipy/stats/_multivariate.py#L289-L851)
- [文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)