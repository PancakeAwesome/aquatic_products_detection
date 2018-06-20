# aquatic_products_detection
This is a object detection module in a environment protection project.We use faster RCNN model to detect the aquatic products under the water so that we can accuqire water qualitiy.Specificly, we firstly receive videos from camera in water and then extract a series of frames from it.Then we can detect these frames by faster RCNN model.The origin paper can be found [here](https://arxiv.org/abs/1506.01497). For more detail about the paper and code, see this [blog][1]

[1]:http://pancakeawesome.ink/%E5%BC%84%E6%87%82%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B(Faster-R-CNN)?%E7%9C%8B%E8%BF%99%E7%AF%87%E5%B0%B1%E5%A4%9F%E4%BA%86!.html
***
# setup
- requirements: tensorflow1.3, opencv-python
- if you have a gpu device, build the library by
```shell
cd lib
chmod +x make.sh
./make.sh
```
# demo
- put your images in data/demo, the results will be saved in data/results, and run demo in the root 
```shell
python ..//tools/demo.py
```
***
# training
## prepare data
- First, download the pre-trained model of VGG16 net and put it in data/pretrain/VGG_imagenet.npy. 
- Second, put your own train data and test data in /data,and modify the data path in ./train_net.py 
```
## train 
Simplely run
```shell
python train_net.py
```
- you can modify some hyper parameters in ctpn/text.yml, or just used the parameters I set.

