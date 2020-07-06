# MINet

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Folders & Files

* `backbone` : Store some code for backbone networks.
* `loss` : The code of the loss function.
* `module` : The code of important modules.
* `network` : The code of the network.
* `output` : It saves all results.
* `utils` : Some instrumental code.
    - `utils/config.py` : Configuration file for model training and testing.
    - `utils/dataset.py` : Some files about creating the dataloader.
* `main.py` : I think you can understand.

## My Environment

Recommended way to install these packages:

``` 
# create env
conda create -n pt python=3.8
conda activate pt

# install pytorch cuda cudnn
conda install pytorch torchvision cudatoolkit=10.2 cudnn -c pytorch

# some tools
pip install tqdm
# (optional) https://github.com/Lyken17/pytorch-OpCounter
# pip install thop
pip install prefetch-generator

# install tensorboard, we can use `from torch.utils.tensorboard import SummaryWriter`
pip install tensorboard

# 我使用了apex来提供混合精度和良好分布式训练支持
# https://github.com/NVIDIA/apex#linux
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## 自动混合精度训练尝试

这里的周期的时间不能固定, 因为在我的对比试验中, 使用的是多尺度训练策略, 导致每个batch的输入的尺寸是不同的, 这可能会影响每个周期的训练时间.
另外, 数据的传输速度在一定程度上也会制约训练时间, 所以这里的时间仅供参考.

* 不使用amp
    - 简单模型：~9009MiB
    - Checkpoint处理模型：~4501MiB
* 添加amp
    - 简单模型：~5527MiB
    - Checkpoint：~3385MiB
* 单独的分布式
    - 简单模型：~5733MiB+5733MiB
    - Checkpoint: ~3335MiB+2863MiB
* amp和分布式一起用
    - 简单模型：~3409MiB+3325MiB
    - Checkpoitn: ~2107MiB+2215MiB

## 注意

虽然这份代码参考了很多的资料(详见最后的列表), 但是还是存在一些不足, 暂时没有得到解决:

* 分布式测试(虽然mmdetection中实现了, 但是它嵌套得太深了, 我暂时没有动力去模仿, 自己尝试使用了如下的简单的策略, 和直接单卡测试略有不同, 暂时搞不清楚原因.
  + 使用分布式sampler
  + 对不同GPU上由不同数据子集得到的指标结果, 使用torch的分布式的收集函数 `all_reduce` 收集数据后, 除以GPU数量得到最终指标
* 使用分布式训练, 对单卡上的batchsize进行翻倍. 我将学习率也跟着线性增长. 但是这样似乎并没有太大的效果提升.
* tqdm如何用在这样的分布式训练过程中呢, 该如何设置?

## 参考

* 这里提供的关于pytorch不同的Collective Communication(集体通信)函数的示意图很不错: https://zhuanlan.zhihu.com/p/76638962
* 这里提供了pytorch多卡训练的各种手段, 非常详细, 这也是我最一开始的主要参考资料之一:
  + https://github.com/tczhangzhi/pytorch-distributed
  + https://zhuanlan.zhihu.com/p/98535650
* 一系列的官方文档: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
* 其他的:
  + https://zhuanlan.zhihu.com/p/113694038
  + https://oldpan.me/archives/pytorch-to-use-multiple-gpus
  + https://www.cnblogs.com/yh-blog/p/12877922.html
