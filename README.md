# Deployment

```shell
cd test && mkdir build && cd build && cmake .. && make && ./testRTDETRv2
```

# Training

## *dataset*

```
http://www.hustlangya.fun/%E6%95%B0%E6%8D%AE%E9%9B%86/AutoAim-ObjDet
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
