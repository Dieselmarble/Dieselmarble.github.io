Indoor Scene Reconstruction with monocular RGB camera

Li Zhongxuan

### Recall from Multi-view Stereo

**Multi-view stereo** (MVS) algorithms produce reconstructions that rival laser range scanner accuracy. However, stereo algorithms require textured surfaces, and therefore work poorly for many architectural scenes (e.g., building interiors with textureless, painted walls). **Manhattan-world Assumption**, also named as Manhattan-world Stereo,  states that all surfaces in the world are aligned with three dominant directions, typically corresponding to the X, Y, and Z axes. 1) it is remarkably robust to lack of texture, and able to model flat painted walls, and 2) it produces remarkably clean, simple models as outputs. By exploiting the vanishing and dominant lines, one can extract structured planes (e.g. walls) even from a single image.

>  Distinctive Image Features from Scale-Invariant Keypoints (David G. Lowe)



![image-20211011141620356](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211011141620356.png)

Manhattan world stereo can be summarized as follows, 

> Oriented points reconstructed by MVS. 通过多视角立体视觉重建的有向点
>
> Dominant axes extracted from points. 从点中获取的主要轴
>
> Point density on d1. peaks. d1方向的点密集度，峰值
>
> Plane hypotheses generated from peaks. 从峰值生成的平面假设
>
> Reconstruction by labeling hypotheses to pixels. (MRF) 通过标注像素的假设来进行重建（马尔科夫随机场）



### Extend the problem to monocular 3D reconstruction

Monocular Planar Reconstruction has the following three main groups of methods:

1. **Geometry-based methods** extract line segments, vanishing points, and super-pixels from an image. Assume an environment is composed of a flat ground and vertical walls, and use a Conditional Random Field (CRF) model to label the detected primitives. 

2. **Appearance-based methods** infer geometric properties of an image based on its appearance. First detect convex/concave edges, occlusion boundaries, superpixels, and their orientations, then formulate the grouping problem as a binary quadratic program under the Manhattan world assumption.

3. Several **CNN-based methods** have been proposed to directly predict global 3D plane structures. Yang et al. propose a deep neural network that learns to infer plane parameters and assign plane IDs

   > Real-time 3D Scene Layout from a Single Image Using Convolutional (Shichao Yang , Daniel Maturana and Sebastian Scherer)



![image-20211011110823598](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211011110823598.png)



### Review on the paper "Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding"

Zehao Yu's method is developed on the Appearance-based methods, composed of two stages. Differs from other methods, which first predict plane parameter and then associate each pixel with a particular plane parameter. In contrast, this parper groups pixels into plane instances and then estimate the parameter for each plane instance. In the first stage, a CNN is trained to obtain planar/non-planar segmentations and pixel embeddings. In the second stage, another neural network is trained to predict pixel level parameters. Finally an instance-aware pooling layer with the instance segmentation map from the first stage is combined with second stage to produce the final plane parameters.

> Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding (Zehao Yu et al.)

![image-20211011104225067](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211011104225067.png)





Traditional **Instance Segmentation** approaches classify the objects in the bounding box and segment the foreground objects within each proposal. In this paper, the author proposed a method that maps each pixel into an embedding space where pixels belonging to same instance have similar embeddings. Then, they use a simple cluster technique to generate instance segmentation results. In short words, it treats each plane in an image as an instance, and utilize the idea of associative embedding to detect plane instances.  Furthermore, the paper propose two additional innovations, i) an efficient mean shift algorithm to cluster plane instances, and ii) an end-to-end trainable network to jointly predict plane instance segmentation and plane parameters.



The paper employs ResNet-101 pretrained on ImageNet as an encoder. Three decodes are added to the original network, i.e., plane segmentation decoder, plane embedding decoder, and plane parameter decoder. These decoders predict planar/non-planar segmentation map for each pixel. correspondingly, the **three loss functions** states as follows,
$$
L_S = −(1 − w) \sum_{i \in F} log p_i − w \sum_{i\in B} log(1 − p_i)
$$
$F$ and $B$ are the set of foreground and background pixels, respectively. $P_i$ is the probability that $i-th$ pixel belongs to foreground, $w$​ is the fore/background pixel number ratios. 

Discriminative loss defines a "pull" loss and a "push" loss.  The “pull” loss pulls each embedding to the mean embedding of the corresponding instance (i.e. the instance center), whereas the “push” loss pushes the instance centers away from each other.
$$
L_{pull} = \frac1 C \sum_{c=1}^C \frac1 N_c \sum_{i=1}^{N_c} max(||\mu_c - x_i|| - \delta_v, 0), \\
L_{push} = \frac1 {C(C-1)} \sum_{C_A}\sum_{C_B}max(\delta_d - ||\mu_{C_A} - \mu_{C_B}||, 0)
$$
where $C$ is the clusters of plane instances, $\mu$ is the mean embedding of the clusters, and $\delta$ is the margin of "pull" and "push" losses.
$$
L_{PP} = \frac1 N \sum_{i=1}^N ||n_i - n_i^{*}||,
$$
where $n_i$ is the predicted plane parameter(e.g. depth) and $n_i^*$ is the ground truth plane parameter for $i$-th pixel.



The last loss function enforces the instance-level parameter to be consistent with the scene geometry. Where $n_j$ is the instance level parameter, $Q_i$ is the  3D point at pixel i inferred from ground truth map.
$$
L_{IP} = \frac 1{N\tilde{C}}\sum_{j=1}^\tilde{C} \sum_{i=1}^N S_{ij} · ||n_j^TQ_i -1||,
$$
The total loss is stated as,
$$
L = L_S + L_E + L_{PP} + L_{IP}
$$
It can be noted that this paper uses **bin mean shifting** as the embedded pixels grouping method. It finds "similar" regions instead of pointing out what each region is, in terms of categories, specifically. Mean shift algorithm has a local region of interest embounding embedded pixels given by the encoder. During each iteration, a new mass of center is calculated after taking into account the shifting direction of encircled pixel vectors. 

### Evaluation Results 

![image-20211011111037728](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211011111037728.png)







We evaluate _Planar 3D Reconstruction via Associative Embedding_ on several self-acquired pictures in the office.

**Top-View floor**

![corner](D:\point-cloud-matching\results\corner.png)

![corner_result](D:\point-cloud-matching\results\corner_result.png)



**Desk**

![desk](C:\Users\l00538412\Desktop\desk.png)

![desk_result](D:\point-cloud-matching\results\desk_result.png)



**Floor**

![floor](D:\point-cloud-matching\results\floor.png)

![floor_result](D:\point-cloud-matching\results\floor_result.png)

It can be seen that this algorithm can well detects the planar regions of the indoor scene, with the speed of approximately 120FPS on a Tesla V100 GPU.

**Point cloud recovered**



![image-20211012115437264](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211012115437264.png)



![image-20211012115554807](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211012115554807.png)



### Other Methods

Inspired by the various instance segmentation methods. We can advise that planar detection can be further refined by CRF/MRF(Conditional Random Filed, Markov Random Field). Similar to *CRFasRNN* and *DenseCRF*. A conditional random field clustering step is cascaded after the encoding step of CNN.

> Conditional Random Fields as Recurrent Neural Networks (Shuai Zheng et, al)

![image-20211011175138862](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211011175138862.png)



### How to apply monocular 3D reconstruction to indoor roadside navigation 

For tasks such as navigation, height of the occupied area above the ground is not important. Distance to the closest occupied points on the floor are sufficient to determine open space on the floor.

![image-20211011150215870](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211011150215870.png)



![img](https://img2018.cnblogs.com/blog/104032/201906/104032-20190606163125032-1863322540.png)



![img](https://img2018.cnblogs.com/blog/104032/201906/104032-20190606163143682-557841463.png)

 View translation is required for converting the front view RGB image and estimated depth map to a bird view graph. Homography is the most conventional method to complete such a task, but ii lacks of height information in the Y-axis. Distortion will be induced by plainly apply homography transformation.



The equation below transform the world axis to resolution axis. From our estimated depth map, we can obtain 3D points in resolution axis and covert them to point clouds in world axis.
$$
Z_c \begin{bmatrix}
u \\
v \\
1
\end{bmatrix}
=\begin{bmatrix}
f_x & 0 & u_0 & 0\\
0 & f_y & v_0 & 0\\
0 & 0 & 1 & 0
\end{bmatrix}
\times 
\begin{bmatrix}
R & T \\
0 & 1
\end{bmatrix}
\times
\begin{bmatrix}
X_w\\
Y_w \\
Z_w \\
1
\end{bmatrix}
$$


### Point cloud to grid map

After extracting the flat floor region, we project point cloud to XY raster plane. By calculating the height difference (i.e. Max - Min in Z direction), we can filter out obstacles in the grid.





### Transformations

![Basic set of 2D planar transformations](https://s2.ax1x.com/2019/05/29/VuEtUA.png)

### Rodrigues函数



### Perspective Transform & Homography Matrix

These are the same thing. This can be confusing because in OpenCV there are two different functions for each: `getPerspectiveTransform` and `findHomography`. When using `getPerspectiveTransform` you must provide a known *good* set of points. That is, the two sets of 4 points must correspond to each other in the same order and each point must be co-planar. `findHomography` loosens this restriction a bit. You can pass in 2 sets of >=4 points you suspect suspect to be “good” and it will use algorithms like [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) to find the best perspective transform between them.



Project objects to its top by applying a shifting matrix tvsecs 

```python
# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h),None)
corners2 = cv2.cornerSubPix(gray,corners, (9,9), (-1,-1), criteria)
imgpoints2, _ = cv2.projectPoints(objp, np.asarray([0, 0, 0], dtype=np.float32), np.asarray([0, 0, 700], dtype=np.float32), mtx, dist)
tmp2 =  np.reshape(corners2[::-1], [-1,2])
trans2,_ = cv2.findHomography(tmp2, tmp1)
dat = cv2.warpPerspective(img, trans2, (1280, 720))
cv2.imshow("perspect transformed", dat)
```

Transform rvsecs to a 3X3 rotation matrix

```python
_, rvecs, tvecs = getCameraParameters(rgb_filename)
rot_mat, _ = cv2.Rodrigues(rvecs[0])
```



- 使用语义图像分割来估计可用空间。
- 基于可用空间估计创建占据栅格。
- 在鸟瞰图上可视化占据栅格。
- 基于占据栅格创建车辆代价地图。
- 检查世界坐标系中的位置被占据还是自由的。



通过世界地图转换后的栅格地图逐个网络与图片坐标系每一个栅格进行距离计算，判断是否小于机器人分辨率大小得到障碍物地图。

栅格数据就是将空间分割成有规律的网格，每一个网格称为一个单元，并在各单元上赋予相应的属性值来表示实体的一种数据形式。每一个单元（像素）的位置由它的行列号定义，所表示的实体位置隐含在栅格行列位置中，数据组织中的每个数据表示地物或现象的非几何属性或指向其属性的指针。在实际的路径搜索问题，将搜寻的区域（一般为二维地图，形状不定）分割为有规律的栅格，每个栅格称为一个单元，每个栅格的位置由其行和列决定，这样就简化搜索区域为 2 维数组，数组的每一项代表一个栅格，它的状态就是可走 (walkable) 和不可走 (unwalkable)，所谓的可走可以理解为该栅格没有障碍物——算法可拓展到该栅格，不可走可以理解为该栅格有障碍物——算法不能拓展到该栅格



原则（1）为力求整数个栅格能够覆盖区域、障碍物和目标物, 当搜索的区域确定时，我们要想到——栅格越小，搜索的区域就需要用更多栅格表示，那么搜索算法的问题空间就越大，计算时间就越长，因此不难得出原则（2）栅格的尺寸应尽可能的大



我们不能对搜索区域中的每一个栅格都进行判定，那样计算量很大，只需要对障碍物附近的栅格进行判断，如何确定附近栅格呢？——取障碍物矩形边界四个角的栅格行列范围：[_x_min,_x_max]、[_y_min,_y_max]，即得到行、列的最大最小值，在这2个范围内使用2个for循环来遍历障碍物附近所有的栅格。



Occupancy Grid Map 

• Maps the environment as an array of cells. – Cell sizes range from 5 to 50 cm. 

• Each cell holds a probability value – that the cell is occupied. 

• Useful for combining different sensor scans, and even different sensor modalities. – Sonar, laser, IR, bump, etc. 

• No assumption about type of features. – Static world, but with frequent updates.

openCV坐标系



总结一下，现有基于CNN做深度估计的算法本质上还都是通过overfit场景中的某种信息来进行深度估计，整体来说这些方法其实并没有考虑到任何几何的限制，所以在不同场景下的泛化能力也比较弱。如何能结合geometry方法和learning方法的优势是一个老生常谈的话题。一方面这个问题可以被拆分成两部分，一部分通过几何的办法来完成不同位置相对深度的估计，另一部分通过一些场景先验或者数据驱动的办法来预测出绝对尺度，然后再融合这两者得到一个可靠鲁棒的深度估计。



![image-20211027151637144](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211027151637144.png)



PlanarReconstruction

First train a CNN to obtain planar/non-planar segmentation map and pixel embeddings. 

Then mask the pixel embeddings with the segmentation map and group the masked pixel embeddings by an efficient mean shift clustering algorithm to form plane instances.

![image-20211009144817775](C:\Users\l00538412\AppData\Roaming\Typora\typora-user-images\image-20211009144817775.png)

Network architecture. In the first stage, the network takes a single RGB image as input, and predicts a planar/nonplanar segmentation mask and pixel-level embeddings. Then, an efficient mean shift clustering algorithm is applied to generate plane instances. In the second stage, we estimate parameter of each plane by considering both pixel-level and instance-level geometric consistencies

**Efficient Mean Shift Clustering**

we propose a fast variant of the mean shift clustering algorithm. Instead of shifting all pixels in embedding space, we only shift a small number of anchors in embedding space and assign each pixel to the nearest anchor. 



After the algorithm converges, we merge nearby anchors to form clusters C˜, where each cluster c˜ corresponds to a plane instance. Specifically, we consider two anchors belongs to the same cluster if their distance less than bandwidth b. The center of this cluster is the mean of anchors belonging to this cluster



Although the fact stated in the argument may somewhat indicate the potential demand from customers of buying the wool fabric. The author acquire the conclusion too hastily that the their future sales and profits are likely to boom. There are several critical reasons that then author must accommodate before he or she can induct any further conclusions.

Firstly, The author claim that due to the fact that itself and its competitor have suspended the selling of fabric wool for five years, customer needs will be willing to buy such wool cloth. However,  the evidence that the ever-intermitted supplying does not suffice lead to the promised future.  The customer preference may have greatly varied during the past five years, which does not guarantee that something populated five years ago would deserve the commensurate level of popularity today. Perhaps, consumers are more concerned about environmental friendly materials than ever before,  and hence would not accept wool fabric that harms animals. Until the author gives out further evidence that substantiate the argument, I remain unconvinced that the customer demands will rise.

Furthermore, the argument based on the incomplete evidence that due to most types of clothing has increased in each the five years, customer should be willing to pay more for overcoats than before. Although it is indeed possible that the the average price of goods is like to rise on annually basis, as a consequence of economic inflation. But such an assumption needs solid evidence that the macro economic  condition will at lease remains the same in the future. Because the author has not provided such an evidence, I cannot make a certain deduction that the overcoat's price will continue to increase. In addition, in the light of the rising price, customers may not be willing to pay for such a high price. Maybe, the improper expensive price prohibits the customer from buying overcoats, therefore does not aggregate to the company's profit. 

Last but not least, the author does not provide information about the cost of wool fabrics today. Profits is total revenue deducted by total cost. Without any information about cost, we cannot assure that the sartorian company's profit will increase. Even if the customers have high willingness to buy wool cloth, and the company's cost remains as low as five years ago. The author still need to proffer a detailed survey showing that the customer will buy Sartorian's product, in particular. Until evidence showing that Sartorian overperforms its rivals in the market, we cannot hastily conclude that the profit will go up.

To conclude, the author need to give further evidence regarding the following three aspects, the competing environment, the wool ingredient cost, and the future economic conditions. I remain unconvinced about the argument that Sartotian's profit is likely to rise.





It is certainly true that the profitability is the most rudimentary responsibility of a corporation. From common sense, that without the ability to keep a healthy cash flow and retain profit, a company would likely to bankrupt. But it would be too hasitly and dogmatically to claim the the only responsibility of corporation is to make as much money as possible. I remain rather neutral about the two claims. A company should take larger social responsibilities after ensuring its financial susbtainability.

First of all, profitability is a necessary but not sufficient condition for a coporation to promoting well-being of surroundings.  Microsoft, as an example, was once world's most profitable company, who also played a critical role  in funding the most cutting-edge technologies and innovations. Furthermore, the huge tech leads in the USA took part in promoting woman's right as well as banishing child labors. A large amount of money instilled in these movements are from retained earning of these company. Hence we can argue that healthy profiability helps the company to better promtes social well-being.

Granted that profitablity undergirds the well-being. Overemphasis on profiability will result in undeserable effects. "Blood diamond", a epithet given by medias to the child-labor in Africa, illustrates that at least some corporation, put there own interest above the basic human moral standard. In addition, monopolies in 1920s, while extremely profitable, take litter social responsibility. Works have to work overtime persistently, but receive miniscule payments. Over-pursuing the profit eventually result in confrontations between labors and capitalists, more serverly, leads to the great recession in 1930.

In sum, if we overemphasize the importance of profitability, and place it as the only responsibility of modern enterprise. It would be detrimental to the overall well-being of human as a whole. But we must be apostle for corporation to make money upon which the it has the spare effort to help the vulnerable groups and improve technology innovations. Governments medias and industrial associations can play a vital role in interacting with corporations to operating better and with benevolence.

真正的机器人应该是怎么样的范式？我不想盲从浪潮，我相信真正的人工智能，应该是超越人，但包括人的智能，应该是从我们对物理世界的规律，能量转化的利用和对现实世界的抽象的数学中涌现出来。

Some other works, convert relational database model to graph database model. This approach usually uses naive converting process, which put all tuple in each relation as a node and the foreign key as an edge between two nodes





Dear Professor Wang,

I have a xyz degree from Imperial College London (2015-2019) and 3 years of industry experience as a research engineer at Huawei. I am passionate about robotics especially reasoning

 and have done so and so work/reading. 

I went through your publication list and absolutely loved the following papers. I liked so and so in paper1 (and go on) I think the application of this idea benefits the industry in so and so way.

I will love researching (name this) topic further with your guidance. I am also planning to apply for a PhD at (university name) and hoping to get an admit for (group name). 

I am attaching my resume for your reference. You may find more information about me on my personal webpage(https://dieselmarble.github.io/)

Please let me know if you think I could be a good fit in your amazing research team. I am happy to 

Regards,

Zhongxuan Li





DexMV: Imitation Learning for Dexterous Manipulation from Human Videos [ Xiaolong Wang, University of California San Diego]

Imitation learning for dexterous manipulation, converts a sequence of hand object poses into robot demonstrations. 

Task Space Vectors (TSV) defined from finger tips position to the hand palm, define a loss function to map the human hand pose to robotic hand.

state-only imitation learning: learn inverse model to predicts missing actions in demonstrations.

state-action imitation learning: Generative Adversarial Imitation Learning (GAIL) & Demo Augmented Policy Gradient (DAPG)

 

From One Hand to Multiple Hands: Imitation Learning for Dexterous Manipulation from Single-Camera Teleoperation [Xiaolong Wang, University of California San Diego]

Construct a customized robot hand for each user in the physical simulator, which is a manipulator resembling the same kinematics structure and shape of the operator’s hand, no need for human-robot hand retargeting. We only need to collect the trajectories once to generate imitation data for all these specified robots. Note these robot hands can even have different morphology compared to the human hand (e.g., Allegro Hand only has four fingers).



State-Only Imitation Learning for Dexterous Manipulation[ Xiaolong Wang, University of California San Diego

propose a simple and effective method that we call SOIL (for StateOnly Imitation Learning). Our method involves an agent that has a behavior policy and an inverse dynamics model. The agent uses the policy to interact with the world and **learns an inverse model from past interactions by self-supervision**. train the inverse dynamics model and the policy network jointly in an iterative manner.state-action demonstrations: demo augmented policy gradient (DAPG), which augments the vanilla policy gradient with an auxiliary imitation term



Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations(https://arxiv.org/pdf/1709.10087.pdf)

