# @Time    : 2020/7/4
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : config.py
# @Project : DistributedSOD
# @GitHub  : https://github.com/lartpang
import os
from collections import OrderedDict
from datetime import datetime

__all__ = ["user_config"]

proj_root = os.path.dirname(__file__)
datasets_root = "/home/lart/Datasets/"

ecssd_path = os.path.join(datasets_root, "Saliency/RGBSOD", "ECSSD")
dutomron_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUT-OMRON")
hkuis_path = os.path.join(datasets_root, "Saliency/RGBSOD", "HKU-IS")
pascals_path = os.path.join(datasets_root, "Saliency/RGBSOD", "PASCAL-S")
soc_path = os.path.join(datasets_root, "Saliency/RGBSOD", "SOC/Test")
dutstr_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUTS/Train")
dutste_path = os.path.join(datasets_root, "Saliency/RGBSOD", "DUTS/Test")

# 配置区域
user_config = {
    # 常用配置
    "model": "cp_res50",
    "resume_mode": "test",  # the mode for resume parameters: ['train', 'test', '']
    "version": "0.2",
    "use_aux_loss": True,  # 是否使用辅助损失
    "save_pre": True,  # 是否保留最终的预测结果
    "epoch_num": 30,  # 训练周期, 0: directly test model
    "lr": 0.05,  # 微调时缩小100倍
    "xlsx_name": "result_full.xlsx",  # the name of the record file
    "output_name": "output",
    "is_distributed": True,
    "use_amp": False,
    "rgb_data": {
        "tr_data_path": dutstr_path,
        "val_data_path": {"pascal-s": pascals_path},
        "te_data_list": OrderedDict(
            {
                "pascal-s": pascals_path,
                "ecssd": ecssd_path,
                "dut-omron": dutomron_path,
                "hku-is": hkuis_path,
                "duts": dutste_path,
                "soc": soc_path,
            }
        ),
    },
    "record_freq": 100,  # >0 使用tensorboard记录的迭代间隔
    "print_freq": 10,  # >0, 打印迭代过程中的信息的迭代间隔
    "val_freq": 5,  # >0, 验证的周期间隔
    "save_freq": 5,  # > 0, 保存模型的周期间隔
    "prefix": (".jpg", ".png"),
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名，
    # "size_list": [224, 256, 288, 320, 352],  # 使用的是fast_create_loader，所以并不是多尺度训练
    "size_list": None,
    # if you dont use the multi-scale training, you can set 'size_list': None
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    "optim": "f3_trick",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "nesterov": False,
    "sche_usebatch": False,
    "lr_type": "poly",
    "warmup_epoch": 1,
    # depond on the special lr_type, only lr_type has 'warmup', when set it to 1, it means no
    # warmup.
    "lr_decay": 0.9,  # poly
    "batch_size": 48,  # 要是继续训练, 最好使用相同的batchsize
    "num_workers": 4,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    "input_size": 320,
    "proj_root": proj_root,
}
