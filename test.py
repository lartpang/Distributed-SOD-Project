# @Time    : 2020/7/4
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : test.py
# @Project : DistributedSODProj
# @GitHub  : https://github.com/lartpang
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

import network as network_lib
from config import user_config
from utils.dataset import create_loader, ImageFolder
from utils.misc import (
    construct_exp_name,
    construct_path_dict,
    construct_print,
    write_data_to_file,
)
from utils.pipeline_ops import resume_checkpoint
from utils.recoder import XLSXRecoder
from utils.saliency_metric import CalTotalMetric

# only on GPUs
assert torch.cuda.is_available()

TEST_SETTING = dict(save_results=False, batch_size=24)

exp_name = construct_exp_name(user_config)
path_config = construct_path_dict(
    proj_root=user_config["proj_root"], exp_name=exp_name, xlsx_name=user_config["xlsx_name"]
)
xlsx_recorder = XLSXRecoder(xlsx_path=path_config["xlsx"])


def main():
    construct_print("We will test the model on one GPU.")
    model = getattr(network_lib, user_config["model"])().cuda()

    # resume model only to test model.
    resume_checkpoint(
        model=model, load_path=path_config["final_full_net"], mode="onlynet",
    )
    test(model)


def test(model):
    model.eval()

    test_dataset_dict = user_config["rgb_data"]["te_data_list"]

    total_results = {}
    for idx, (data_name, data_path) in enumerate(test_dataset_dict.items()):
        construct_print(f"Testing on the dataset: {data_name}, {data_path}")
        test_set = ImageFolder(root=data_path, in_size=user_config["input_size"], training=False)
        te_dataset_length = len(test_set)
        te_loader = create_loader(
            data_set=test_set,
            size_list=None,
            batch_size=user_config["batch_size"]
            if not TEST_SETTING["batch_size"]
            else TEST_SETTING["batch_size"],
            shuffle=False,
            num_workers=user_config["num_workers"],
            sampler=None,
            drop_last=False,
            pin_memory=True,
        )
        save_path = os.path.join(path_config["save"], data_name)
        if not os.path.exists(save_path):
            construct_print(f"{save_path} do not exist. Let's create it.")
            os.makedirs(save_path)
        results = _test_process(
            model=model,
            te_loader=te_loader,
            length=te_dataset_length,
            save_pre=user_config["save_pre"],
            save_path=save_path,
        )
        msg = f"Results on the testset({data_name}:'{data_path}'):\n{results}"
        if TEST_SETTING["save_results"]:
            write_data_to_file(msg, path_config["te_log"])
        construct_print(msg)

        total_results[data_name.upper()] = results
    return total_results


def _test_process(model, te_loader, length, save_pre: bool = False, save_path: str = "") -> dict:
    cal_total_metrics = CalTotalMetric(num=length, beta_for_wfm=1)
    to_pil = transforms.ToPILImage()

    tqdm_iter = tqdm(enumerate(te_loader), total=len(te_loader), leave=False)
    for test_batch_id, test_data in tqdm_iter:
        tqdm_iter.set_description(f"{exp_name}: te=>{test_batch_id + 1}")
        with torch.no_grad():
            in_imgs, in_names, in_mask_paths = test_data
            in_imgs = in_imgs.cuda(non_blocking=True)
            outputs = model(in_imgs)
        outputs_np = outputs.sigmoid().cpu().detach()

        for item_id, out_item in enumerate(outputs_np):
            gimg_path = os.path.join(in_mask_paths[item_id])
            gt_img = Image.open(gimg_path).convert("L")
            out_img = to_pil(out_item).resize(gt_img.size)

            if save_pre:
                oimg_path = os.path.join(save_path, in_names[item_id] + ".png")
                out_img.save(oimg_path)

            gt_img = np.asarray(gt_img)
            out_img = np.asarray(out_img)

            # 归一化
            gt_img = gt_img / (gt_img.max() + 1e-8)
            gt_img = np.where(gt_img > 0.5, 1, 0)
            out_img_max = out_img.max()
            out_img_min = out_img.min()
            if out_img_max == out_img_min:
                out_img = out_img / 255
            else:
                out_img = (out_img - out_img_min) / (out_img_max - out_img_min)

            # 更新指标记录
            cal_total_metrics.update(out_img, gt_img)
    results = cal_total_metrics.show()
    return results


if __name__ == "__main__":
    main()
