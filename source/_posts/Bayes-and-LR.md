---
title: 贝叶斯和逻辑回归的异同分析
date: 2018-11-02 15:40:42
tags:
categories: Blog
---

An idea popped up in my mind about the relationship between Logistic Regression and Bayes' theorem.

## 1. Bayes’ theorem

Bayes's theorem solve a specific class of problem called  [Inverse probability](https://en.wikipedia.org/wiki/Probability_distribution)

- Statement of theorem (from Wikipedia):

![](/uploads/bayes.png)

<!-- <img src="bayes.png" width="300" hegiht="90" align=center/> -->

想象A是类别，B是特征。

![](/uploads/conditionP.png)

因为求条件概率太麻烦，设想有n个$X_i$变量，那么条件概率展开有$2*2^{n}种$！

因此我们假设$C_i$ 是 i.i.d，从而引出朴素贝叶斯。

### 朴素贝叶斯算法 Naive Bayes Algorithm

朴素贝叶斯算法是一个基于贝叶斯法则的分类算法，它假设$X$的各个属性$X_1,X_2...,X_n$ 在给定* $Y$的前提下是条件无关的

$$ P(Y = y_k|X_1 ...X_n) = P(Y = y_k)P(X_1 ...X_n|Y = y_k) = \sum_j P(Y = y_j)P(X_1 ...X_n|Y = y_j)$$

我们寻找的是概率最大的$y_k$ 因此可以用

$$ Y ← argmax y_k P(Y = y_k)\prod_{i}  P(X_i |Y = y_k) $$

为什么去掉了贝叶斯公式中分母？因为后验概率中分母是确定的！

补充一个有趣的公式：

$$ P(C_i|A,B) = \frac{P(C_i,B|A)}{\sum_i P(C_i, B|A)} $$



### 更加简单地理解贝叶斯公式 

我们蒙着眼睛，伸手从箱子取球：由于球的大小形状和数量都一样，所以我们认为取的球来自A箱和B箱的概率都是1/2。

但是我摸出来以后，我瞄了一眼，发现：这是白球。然后我就断定：这个球一定来自A箱子。

摸出来的球来自A箱的概率由1/2变成了1。这是为什么呢？就是因为有后验概率是不一样的，摸出来球的颜色会对一开始的概率产生影响。

## 2. Logistic regression

逻辑回归基于线性回归，然而它是分类器

Given a hypothesis based on sigmoid function:

$$ h_\theta (x)= \frac{1}{1+e^{-z}} = \frac{1}{1+e^{-\theta^T x}}$$

Where $X$ is a column vector formed with features. $\theta$ is the projection matrix. $\theta^T x = \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$

可见Logistic Regression算法是将线性函数的结果映射到了sigmoid函数中。

Sigmoid 函数的值域在（0，1）区间。

现在假设有两个事件 X, Y。

Y 基于 X 发生的概率为：

$$ P(y=1|x) = \frac{1}{1+e^{-\theta^Tx}} $$

在X 条件下Y的概率为：

$$ P(y=0|x) = 1- P(y=1|x) = 1- \frac{1}{1+e^{-\theta^Tx}} = \frac{1}{1+e^{\theta^Tx}} $$

事件发生于不发生的概率比（odds）为：

$$ \frac{P(y=1|x)}{P(y=0|x)} = \frac{p}{1-p} = e^{\theta^T x}$$

对odds取对数函数：

$$ ln(\frac{P}{1-P}) = \theta^T x $$

那么那么如何求出分类器的参数$\theta$呢。

对于 projection matrix $\theta$ 的求解，我们用到了最大拟然估计（MLE）就是利用已知的样本结果信息，反推最具有可能（最大概率）导致这些样本结果出现的模型参数值。

$$ L(\theta) = \prod_{i=1}^{m} h_\theta (x_i) )^{y_i} (1- h_\theta (x_i)^{1-y_i} $$

assuming are observations are independent

### 两者的内在联系

那么贝叶斯和逻辑回归有着什么样的内在联系呢？

> http://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf 给出了详尽的解释。

首先，

- Logistic regression和Naive bayes都是对特征的线性表达

- Logistic regression和Naive bayes建模的都是条件概率

不得不说有的时候找到合适的文献非常重要。。。

To summarize, Logistic Regression directly estimates the parameters of P(Y|X), whereas Naive Bayes directly estimates parameters for P(Y) and P(X|Y). We often call the former a discriminative classifier, and the latter a generative classifier.

The two algorithms also differ in interesting ways:

- When the GNB modeling assumptions do not hold, Logistic Regression and GNB typically learn different classifier functions. In this case, the asymptotic (as the number of training examples approach infinity) classification accuracy for Logistic Regression is often better than the asymptotic accuracy of GNB. Although Logistic Regression is consistent with the Naive Bayes assumption that the input features Xi are conditionally independent given Y, it is not rigidly tied to this assumption as is Naive Bayes. Given data that disobeys this assumption, the conditional likelihood maximization algorithm for Logistic Regression will adjust its parameters to maximize the fit to (the conditional likelihood of) the data, even if the resulting parameters are inconsistent with the Naive Bayes parameter estimates.
- GNB and Logistic Regression converge toward their asymptotic accuracies at different rates. As Ng & Jordan (2002) show, GNB parameter estimates converge toward their asymptotic values in order logn examples, where n is the dimension of X. In contrast, Logistic Regression parameter estimates converge more slowly, requiring order n examples. The authors also show that in several data sets Logistic Regression outperforms GNB when many training examples are available, but GNB outperforms Logistic Regression when training data is scarce.
- Naive Bayes is a learning algorithm with greater bias, but lower variance, than Logistic Regression. If this bias is appropriate given the actual data, Naive Bayes will be preferred. Otherwise, Logistic Regression will be preferred.

### 有用的补充

[最大拟然估计](https://zhuanlan.zhihu.com/p/26614750)

