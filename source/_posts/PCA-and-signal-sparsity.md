---
title: PCA and signal sparsity
date: 2018-10-31 20:44:13
tags:
categories: Blog
---

I just realised that PCA has somewhat of relationship to signal sparsity.

In PCA, we calculate the eigen-space based on eigenvalue decomposition of data covariance matrix 

$$ S = AA^T $$ 

where $ A = (x - x)(x - x)^T $

This equivalent(similar, strictly speaking) to performing SVD on A.

In SVD, we can get rid of small singular values due to signal sparsity.

Hence we can truncate the eigen values in PCA.

![image.png](/uploads/egval.png)