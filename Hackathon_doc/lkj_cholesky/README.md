


pytorch 

https://github.com/pytorch/pytorch/blob/9bb54c7f3c048fdf2e5294e9e49e3642f1de98d8/torch/distributions/lkj_cholesky.py


https://mc-stan.org/docs/2_25/functions-reference/lkj-correlation.html


## onion

1. $Y \sim \text{Beta}(\alpha, \beta)$

2. $U_{\text{normal}}$ 是一个下三角矩阵，
$$
U_{\text{normal}} = 
\begin{cases} 
    \mathcal{N}(0,1), & \text{if } i > j \\
    0, & \text{if } i \leq j \\
\end{cases}
$$
3. 将这个下三角矩阵的每一行归一化到单位超球面上，得到 $U_{\text{hypersphere}}$，
其中
 $U_{\text{hypersphere},i} = \frac{U_{\text{normal},i}}{||U_{\text{normal},i}||}$
4. 用零填充 $U_{\text{hypersphere}}$ 的第一行。

5. 计算 $ W $ 矩阵，它是 $U_{\text{hypersphere}}$ 和 $\sqrt{Y}$ 的哈达玛积（即元素相乘）。$W = \sqrt{Y} \cdot U_{\text{hypersphere}}$

$$
W_{i,j} = 
\begin{cases} 
\sqrt{Y} \cdot \frac{U_{i,j}}{||U_{i,*}||}, & \text{if } i > j \\
\sqrt{1 - \sum_{k < i} W_{i,k}^2}, & \text{if } i = j \\
0, & \text{if } i < j
\end{cases}
$$

其中 $ i $ 和 $ j $ 是矩阵的行索引和列索引，$ U_{i,*} $ 表示 $ U $ 矩阵的第 $ i $ 行。这个过程生成的 $ W $ 矩阵是一个随机正交矩阵，它的行向量是彼此正交的，并且都有单位长度。



## cvine

### 步骤1: 部分相关系数的生成

对于每一对变量，我们首先需要生成它们之间的部分相关系数。这可以通过从Beta分布中采样获得：

$$
\beta_{ij} \sim \text{Beta}(\alpha, \beta), \quad \text{for } i < j
$$

然后，将Beta分布的采样结果转换到$[-1, 1]$区间以获取部分相关系数：

$$
\rho_{ij} = 2\beta_{ij} - 1
$$

### 步骤2: 构造下三角矩阵

将这些部分相关系数填充到一个下三角矩阵中，其中$i < j$的元素对应于变量$i$和变量$j$之间的部分相关系数：

$$
R = 
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
r_{21} & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
r_{n1} & r_{n2} & \cdots & 1
\end{bmatrix}
$$

### 步骤3: 计算累积乘积的平方根

对于矩阵$R$中的每个元素，计算其累积乘积的平方根，并进行必要的填充：

$$
z_{ij} = \sqrt{\prod_{k=i+1}^{j-1} (1 - r_{kj}^2)}, \quad \text{for } i > j
$$

这里，$z_{ij}$表示在考虑到变量$i$和变量$j$之间的直接依赖关系时，间接通过其他变量的影响被考虑进来。

### 步骤4: 最终矩阵的构造

- $out_{ij} =  z_{ij} * r_{ij}$
设$O_{i,j}$为最终构造的矩阵中的元素，其中$i$和$j$分别表示矩阵的行和列。则：

$$
O_{i,j} = 
\begin{cases} 
    r_{i,j} \cdot z_{ij}, & \text{if } i > j \\
    \sqrt{1 - \sum_{k=1}^{i-1} r_{k,i}^2 \cdot \prod_{m=k+1}^{i-1} (1 - r_{m,i}^2)}, & \text{if } i = j \\
    0, & \text{if } i < j
\end{cases}
$$
这里，$\rho_{i,j}$代表变量$i$和变量$j$之间的部分相关系数。当$i < j$时，$\rho_{i,j}$乘以从$i+1$到$j-1$的所有$\rho_{k,j}$的补数（即$1 - \rho_{k,j}^2$）的乘积的平方根，这反映了通过其他变量间接影响$i$和$j$之间关系的调整。当$i = j$时，对角线元素通过计算1减去所有小于当前行的$\rho_{k,i}$的平方乘以它们对应的从$k+1$到$i-1$的所有$\rho_{m,i}$的补数的乘积的平方根，以确保矩阵的正定性。当$i > j$时，矩阵的下三角部分被设置为0，因为我们只考虑上三角矩阵（包含对角线）来表示变量之间的依赖结构。

请注意，这个公式是对C-型Vine构造过程的一个简化表示，实际应用中可能需要根据具体情况进行调整和细化。