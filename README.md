# Team Member :
- 謝忱
- 張鈞
- 周秉儒

## 2019 Spring Final Project -- Multi-Source Domain Adaption
A pytorch implementation of [Moment Matching for Multi-Source domain adaption](https://arxiv.org/pdf/1812.01754.pdf)
<p align="center">
  <img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/introduction.png">
</p>


## Environment
- pytorch : 3.5+
- torch : 1.0
- numpy : 1.16.2
- pandas : 0.24.0
- torchvision : 0.2.2
- matplotlib
- sklearn 

## Dataset
In this project, we use four dataset in DomainNet(quickdraw, real, infograph, sketch)
<p align="center">
  <img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/dataset_img.png">
</p>

## Download Model ( ⚠️***VERY IMPORTANT***⚠️ )
```
bash ./get_model.sh
```

## Train
For trainging : < **Last one will be Target domain, other will be Source domain** >
```
python3 train.py $1 $2 $3 $4

For example:
python3 train.py sketch infograph quickdraw real
```
Source domain (sketch, infograph, quickdraw) ---> Target domain (real)

- **$1~$3** are source domain 
- **$4** is target domain


## Evaluation
For eval : < **Last one will be Target domain, other will be Source domain** >
```
python3 eval.py $1 $2 $3 $4

For example:
python3 eval.py sketch infograph quickdraw real
```
Source domain (sketch, infograph, quickdraw) ---> Target domain (real)

- **$1~$3** are source domain
- **$4** is target domain


## Predict
For predict :
```
bash ./predict.sh $1 $2

For example:
bash ./predict.sh ./dataset_public/test/ real
```
- **$1** : Image Path 
- **$2** : Target domain

## Result

|              | inf, real, skt --> qdr | inf, real, qdr --> skt  | inf, qdr, skt --> real | qdr, skt, real --> inf |
| :----------: | :--------------------: | :---------------------: | :--------------------: | :--------------------: |
|     MSDA     |        7.82 %          |        30.09 %          |        14.26 %         |        54.54 %         |
 

## Visualization
<p align="center">TSNE PLOT (LR=3e-4, Batch size = 256)</p>
<p float="left">
<img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/tsne_info.png" width=200 height=200 ><img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/tsne_real.png" width=200 height=200><img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/tsne_quickdraw.png" width=200 height=200><img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/tsne_sketch.png" width=200 height=200>
</p>  

<p align="center">Grad CAM PLOT</p>
<p float="left">
<img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/info.jpg" width=200 height=200>
<img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/real.jpg" width=200 height=200>
<img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/quick.png" width=200 height=200>
<img src="https://github.com/dlcv-spring-2019/final-DLCV_fadacai/blob/master/image/sketch.jpg" width=200 height=200>
</p>



# Reference
[Moment Matching for Multi-Source Domain Adaptation](https://arxiv.org/pdf/1812.01754.pdf)  
[Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1712.02560.pdf)  
[Multi-Source Domain Adaptation with Mixture of Experts](https://arxiv.org/pdf/1809.02256.pdf)  
[最大分类器差异的领域自适应](https://zhuanlan.zhihu.com/p/52085426)  


