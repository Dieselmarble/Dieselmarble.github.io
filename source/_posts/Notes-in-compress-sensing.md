---
title: Notes in compress sensing
date: 2018-11-02 15:41:38
tags:
categories: Blog
---

## 毕业设计知识储备

压缩传感理论主要包括信号的稀疏表示、编码测量和重构算法等三个方面。信号的稀疏表示就是将信号投影到正交变换基时，绝大部分变换系数的绝对值很小，所得到的变换向量是稀疏或者近似稀疏的，以将其看作原始信号的一种简洁表达，这是压缩传感的先验条件，即信号必须在某种变换下可以稀疏表示。 通常变换基可以根据信号本身的特点灵活选取， 常用的有离散余弦变换基、快速傅里叶变换基、离散小波变换基、Curvelet基、Gabor 基 以及冗余字典等。 在编码测量中， 首先选择稳定的投影矩阵，为了确保信号的线性投影能够保持信号的原始结构， 投影矩阵必须满足约束等距性 (Restricted isometry property, RIP)条件， 然后通过原始信号与测量矩阵的乘积获得原始信号的线性投影测量。最后，运用重构算法由测量值及投影矩阵重构原始信号。信号重构过程一般转换为一个最小L0范数的优化问题，求解方法主要有最小L1 范数法、匹配追踪系列算法、最小全变分方法、迭代阈值算法等。

- [Compress sensing(压缩感知)](http://www.math.umu.se/digitalAssets/115/115905_csfornasierrauhut.pdf)
- Restricted Isometry Property, RIP (有限等距性质)

![](/uploads/rip.png)

<!-- <img src="rip.png" width="300" hegiht="90" align=center/> -->

> CandesE, Tao T. Decoding by linear programming. IEEE Transactions on InformationTheory, 2005,59(8):4203-4215

从能量角度来说，RIP实际上保证了压缩观测前后的能量变化范围，下限不能为零，上限不能超过原信号的两倍（RIP不等式中取δ=1的极限时）。这里的下限不能为零是很好理解的，当信号能量为零时表示原信号的所有项全部为零，零向量中自然是没有信息的；上限怎么理解呢？为什么压缩观测后的能量不能超过原信号的两倍呢？