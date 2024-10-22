# Deployment

```shell
cd test && mkdir build && cd build && cmake .. && make && ./testRTDETRv2
```

# Training

## *dataset*

```
http://db.hustlangya.fun/%E6%95%B0%E6%8D%AE%E9%9B%86/AutoAim-ObjDet
```

## *pretrained model*

```
http://db.hustlangya.fun/%E6%A8%A1%E5%9E%8B%E5%BA%93/AutoAim-rtdetr
```

## *command*

```shell
cd rtdetrv2_pytorch
pip install -r requirements.txt
bash run.sh
```

# View our log


### *2024/10/19*

```shell
## txt
vim rtdetrv2_pytorch/output/241019/log.txt
## tensorboard
tensorboard --logdir=rtdetrv2_pytorch/output/241019/events.out.tfevents.1729333525.langya-radar.122513.0
```
