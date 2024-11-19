---
tags: []
parent: ""
collections:
    - RL+LLMs
$version: 46
$libraryID: 1
$itemKey: NXKJ7N9A

---
#### 一、动机 (Motivation)

##### 1. 要解决的问题

奖励函数难以设计：在一些复杂任务中，要设计一个合理有效的奖励函数十分困难，而在强化学习中奖励函数至关重要

##### 2. 方法

用human feedback数据拟合奖励函数，然后训练RL模型

##### 3. 本文设计

本文的feedback非常简单，不需要专家标注，普通人看2段1\~2秒的视频，然后比较，挑选出自己认为更好的那个，将该比较数据作为feedback

#### 二、框架 (Framework)

##### 1. 设定

$agent$的策略用$\pi$来表示：$\pi:O \rightarrow A$，拟合的奖励函数为：$\widehat{r}:O \times A \rightarrow R$，都使用神经网络进行拟合

##### 2. 步骤

所有的神经网络用下面三步来进行更新：

*   策略$\pi$和环境交互，得到很多条$trajectory:\{\tau^1,\tau^2,...,\tau^i\}$，然后使用传统的RL算法更新$\pi$的参数，目的是最大化预测出的奖励$r_t=\widehat{r}\{o_t,a_t\}$

*   从轨迹集合$\{\tau^1,\tau^2,...,\tau^i\}$中筛选出多对轨迹片段$(\sigma^1,\sigma^2)$，发给标注人员进行比较，挑选出自己觉得更好的那个轨迹片段

*   利用标注结果更新拟合的奖励函数$\widehat{r}$

每一条标注过的feedback数据可以看作三元组：$(\sigma^1,\sigma^2,\mu),\mu\in\{1,2\}$，$\mu$表示更好的那个轨迹片段

##### 3. 如何使用feedback来拟合奖励函数

这里的奖励函数其实就是feedback二分类器，作者使用Bradley-Terry模型的方法：

*   Bradley-Terry模型：

    *   每个竞争者被赋予一个实力参数，这些 参数用于预测在任意两个竞争者之间的比较结果。模型的基本假设是，一个竞争者击败另一个竞争者的概率与这两个竞争者的实力参数的比值有关

    *   如果我们有两个竞争者A和B，他们的实力参数为$p_A,p_B$，那么A击败B的概率可以表示为：$P(A \succ B)=\frac{p_A}{p_A + p_B}$

在本文中，片段1的实力参数为片段1的所有奖励总和的指数值，片段2的实力参数为片段2的所有奖励总和的指数值，故片段1比片段2好的拟合概率为：

$$
\widehat{P}(\sigma^1 \succ \sigma^2)=\frac{exp\sum\widehat{r}(o^1_t,a^1_t)}{exp\sum\widehat{r}(o^1_t,a^1_t) + exp\sum\widehat{r}(o^2_t,a^2_t)}
$$

我们选择能够最小化这些预测值和真实人类反馈之间交叉熵损失的奖励函数$\widehat{r}$，即使用交叉熵函数对$\widehat{r}$的神经网络计算损失进行更新：

$$
loss(\widehat{r})=-\sum_{(\sigma^1,\sigma^2,\mu)\in D} \mu(1)\log{\widehat{P}[\sigma^1\succ\sigma^2]}+\mu(2)\log{\widehat{P}[\sigma^2\succ\sigma^1]}
$$

在真实实现的过程中，作者使用了一些技巧：

*   奖励函数集成，即同时训练多个奖励函数，reward值是所有预测值的平均

*   保留$1/e$部分的数据作为验证集

*   假设10%的错误率，也就是有0.1的概率标注人员是乱选的

*   轨迹片段不只是随即筛选的，而是先随机筛选一大堆，然后用多个拟合奖励函数对轨迹片段进行预测得到多个reward，将方差大的那些轨迹挑选出来，交给人来比较
