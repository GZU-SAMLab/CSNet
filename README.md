# CSNet

**Count-Supervised Network (CSNet)**  can complete the counting of wheat ears with only quantitative supervision.

paper: CSNet: A Count-supervised Network via Multiscale MLP-Mixer for Wheat Ear Counting

## The Overview of CSNet
![](methodoverview.png pic_center =30x30)


## About Data
We use the global wheat Head Detection ([dataset](http://www.global-wheat.com/gwhd.html)) for training, where the quantity labels are obtained by summing the target boxes in the dataset.

## Code Structure
`train.py` To train the model. 

`demo.py` To predict an image. 

`model/model_vgg16.py` The structure of the network and the backbone is vgg16. 

`model/model_MobileNetV2.py` The structure of the network and the backbone is MobileNetV2. 

`model/backbone_vgg16.py` The structure of the first ten layers of Vgg16. 

`tool/gwhdcoco_count.py`  To convert the coco label in the GWHD dataset to count label.

`tool/mergecoco.py`  To combine multiple Json files of coco label into one.

## Training
```shell
python train.py --batch_size=16 --epoch=1000 --lr=1e-4 --device="cuda" 
```

## Mode Weight


## Result
![](result.png)

