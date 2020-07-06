# @Time    : 2020/7/4
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : train.py
# @Project : DistributedSODProj
# @GitHub  : https://github.com/lartpang
import argparse
import os
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data.distributed as data_dist
from apex import amp
from apex.parallel import convert_syncbn_model, DistributedDataParallel as DDP
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.backends import cudnn
from torch.nn import BCEWithLogitsLoss
from torchvision.transforms import transforms
from tqdm import tqdm

import network as network_lib
from config import user_config
from utils.dataset import create_loader, ImageFolder
from utils.misc import (
    AvgMeter,
    check_mkdir,
    construct_exp_name,
    construct_path_dict,
    construct_print,
    init_cudnn,
    init_seed,
    write_data_to_file,
)
from utils.pipeline_ops import (
    CustomScheduler,
    get_total_loss,
    make_optimizer,
    resume_checkpoint,
    save_checkpoint,
)
from utils.recoder import TBRecorder, Timer, XLSXRecoder
from utils.saliency_metric import CalTotalMetric
from utils.tensor_ops import allreduce_tensor

my_parser = argparse.ArgumentParser(
    prog="main script",
    description="The code is based on our MINet.",
    epilog="Enjoy the program! :)",
    allow_abbrev=False,
)
my_parser.version = "1.0.0"
my_parser.add_argument("-v", "--version", action="version")
my_parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N", help="分布式训练节点总数")
my_parser.add_argument("-ng", "--ngpus_per_node", default=2, type=int, help="分布式训练每个节点的GPU数量")
my_parser.add_argument("--ip", default="127.0.0.1", type=str, help="分布式训练的主IP地址")
my_parser.add_argument("-p", "--port", default="8888", type=str, help="分布式训练的主端口")
args = my_parser.parse_args()

# only on GPUs
assert torch.cuda.is_available()

exp_name = construct_exp_name(user_config)
path_config = construct_path_dict(
    proj_root=user_config["proj_root"], exp_name=exp_name, xlsx_name=user_config["xlsx_name"]
)
tb_recorder = TBRecorder(tb_path=path_config["tb"])
xlsx_recorder = XLSXRecoder(xlsx_path=path_config["xlsx"])


def init_process(ip, port, rank, world_size):
    os.environ["MASTER_ADDR"] = ip
    os.environ["MASTER_PORT"] = port
    # initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)


def destroy_process():
    dist.destroy_process_group()


def main():
    check_mkdir(path_config["save"])
    check_mkdir(path_config["pth"])
    # shutil.copy(f"{user_config['proj_root']}/config.py", path_config["cfg_log"])
    # shutil.copy(f"{user_config['proj_root']}/train.py", path_config["trainer_log"])

    if user_config["is_distributed"]:
        construct_print("We will use the distributed training.")
        args.world_size = args.ngpus_per_node * args.nodes
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args.world_size),
        )
    else:
        construct_print("We will not use the distributed training.")
        main_worker(
            local_rank=0, ngpus_per_node=1, world_size=1,
        )
    tb_recorder.close_tb()


def main_worker(local_rank, ngpus_per_node, world_size):
    global total_iter_num, batch_size_single_gpu

    if local_rank == 0:
        construct_print(user_config)
        construct_print(f"Project Root: {user_config['proj_root']}")
        construct_print(f"Training on: {user_config['rgb_data']['tr_data_path']}")

    # https://github.com/tczhangzhi/pytorch-distributed/issues/4
    init_seed(seed=0)
    init_cudnn(benchmark=(user_config["size_list"] == None))

    if user_config["is_distributed"]:
        init_process(ip=args.ip, port=args.port, rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    batch_size_single_gpu = user_config["batch_size"] // ngpus_per_node

    train_set = ImageFolder(
        root=user_config["rgb_data"]["tr_data_path"],
        in_size=user_config["input_size"],
        training=True,
    )
    train_sampler = (
        data_dist.DistributedSampler(train_set) if user_config["is_distributed"] else None
    )
    tr_loader = create_loader(
        data_set=train_set,
        size_list=user_config["size_list"],
        batch_size=batch_size_single_gpu,
        shuffle=(train_sampler == None),
        num_workers=user_config["num_workers"],
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
    )
    total_iter_num = user_config["epoch_num"] * len(tr_loader)

    model = getattr(network_lib, user_config["model"])().cuda(local_rank)

    # 单独的测试部分
    if user_config["resume_mode"] == "test":
        if local_rank == 0:
            # resume model only to test model.
            # start_epoch is useless
            resume_checkpoint(
                model=model, load_path=path_config["final_full_net"], mode="onlynet",
            )
            test(model)
            construct_print("GPU:0 end testing...")
        else:
            construct_print("GPU:1 no testing...")
        return

    optimizer = make_optimizer(
        model=model,
        optimizer_type=user_config["optim"],
        optimizer_info=dict(
            lr=user_config["lr"],
            momentum=user_config["momentum"],
            weight_decay=user_config["weight_decay"],
            nesterov=user_config["nesterov"],
        ),
    )
    scheduler = CustomScheduler(
        optimizer=optimizer,
        total_num=total_iter_num if user_config["sche_usebatch"] else user_config["epoch_num"],
        scheduler_type=user_config["lr_type"],
        scheduler_info=dict(
            lr_decay=user_config["lr_decay"], warmup_epoch=user_config["warmup_epoch"]
        ),
    )
    if local_rank == 0:
        construct_print(f"optimizer = {optimizer}")
        construct_print(f"scheduler = {scheduler}")

    if user_config["is_distributed"]:
        model = convert_syncbn_model(model)
    if user_config["use_amp"]:
        assert cudnn.enabled, "Amp requires cudnn backend to be enabled."
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if user_config["is_distributed"]:
        model = DDP(model, delay_allreduce=True)

    # 训练部分
    if user_config["resume_mode"] == "train" and local_rank == 0:
        # resume model to train the model
        start_epoch = resume_checkpoint(
            model=model,
            optimizer=optimizer,
            amp=amp if user_config["use_amp"] else None,
            exp_name=exp_name,
            load_path=path_config["final_full_net"],
            mode="all",
            local_rank=local_rank,
        )
    else:
        # only train a new model.
        start_epoch = 0

    loss_funcs = [BCEWithLogitsLoss(reduction=user_config["reduction"]).cuda(local_rank)]
    if user_config["use_aux_loss"]:
        from loss.CEL import CEL

        loss_funcs.append(CEL().cuda(local_rank))

    train(
        model=model,
        start_epoch=start_epoch,
        end_epoch=user_config["epoch_num"],
        tr_loader=tr_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_funcs=loss_funcs,
        train_sampler=train_sampler,
        local_rank=local_rank,
    )
    construct_print("End Training...")

    if user_config["is_distributed"]:
        destroy_process()


def train(
    model,
    start_epoch,
    end_epoch,
    tr_loader,
    optimizer,
    scheduler,
    loss_funcs,
    train_sampler,
    local_rank,
):
    for curr_epoch in range(start_epoch, end_epoch):
        if user_config["is_distributed"]:
            train_sampler.set_epoch(curr_epoch)
        if not user_config["sche_usebatch"]:
            scheduler.step(optimizer=optimizer, curr_epoch=curr_epoch)

            train_epoch_prefetch_generator(
                curr_epoch,
                end_epoch,
                loss_funcs,
                model,
                optimizer,
                scheduler,
                tr_loader,
                local_rank,
            )

        if local_rank == 0:
            # note: to varify the correctness of the modl and training process
            if (user_config["val_freq"] > 0) and (curr_epoch + 1) % user_config["val_freq"] == 0:
                _ = test(model, mode="val", save_pre=False)

            if (
                (user_config["save_freq"] > 0) and (curr_epoch + 1) % user_config["save_freq"] == 0
            ) or (curr_epoch == end_epoch - 1):
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    amp=amp if user_config["use_amp"] else None,
                    exp_name=exp_name,
                    current_epoch=curr_epoch + 1,
                    full_net_path=path_config["final_full_net"],
                    state_net_path=path_config["final_state_net"],
                )

    if local_rank == 0:
        total_results = test(model, mode="test", save_pre=user_config["save_pre"])
        xlsx_recorder.write_xlsx(exp_name, total_results)


@Timer
def train_epoch_prefetch_generator(
    curr_epoch, end_epoch, loss_funcs, model, optimizer, scheduler, tr_loader, local_rank,
):
    model.train()
    train_loss_record = AvgMeter()

    for train_batch_id, (train_inputs, train_masks, train_names) in enumerate(
        BackgroundGenerator(tr_loader, max_prefetch=2)
    ):
        curr_iter = curr_epoch * len(tr_loader) + train_batch_id
        if user_config["sche_usebatch"]:
            scheduler.step(optimizer, curr_epoch=curr_iter)

        train_inputs = train_inputs.cuda(non_blocking=True)
        train_masks = train_masks.cuda(non_blocking=True)
        train_preds = model(train_inputs)

        train_loss, loss_item_list = get_total_loss(train_preds, train_masks, loss_funcs)

        optimizer.zero_grad()
        if user_config["use_amp"]:
            with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            train_loss.backward()
        optimizer.step()

        if user_config["is_distributed"]:
            reduced_loss = allreduce_tensor(train_loss)
        else:
            reduced_loss = train_loss
        train_iter_loss = reduced_loss.item()
        train_loss_record.update(train_iter_loss, train_inputs.size(0))

        if local_rank == 0:
            lr_str = ",".join(
                [f"{param_groups['lr']:.7f}" for param_groups in optimizer.param_groups]
            )
            log = (
                f"[I:{train_batch_id}/{len(tr_loader)}/{curr_iter}/{total_iter_num}][E:{curr_epoch}:{end_epoch}]>["
                f"{exp_name}]"
                f"[Lr:{lr_str}][Avg:{train_loss_record.avg:.5f}|Cur:{train_iter_loss:.5f}|"
                f"{loss_item_list}]\n"
                f"{train_names}"
            )
            if user_config["print_freq"] > 0 and (curr_iter + 1) % user_config["print_freq"] == 0:
                print(log)
            if (
                user_config["record_freq"] > 0
                and (curr_iter + 1) % user_config["record_freq"] == 0
            ):
                tb_recorder.record_curve("trloss_avg", train_loss_record.avg, curr_iter)
                tb_recorder.record_curve("trloss_iter", train_loss_record.avg, curr_iter)
                tb_recorder.record_curve("lr", optimizer.param_groups, curr_iter)
                tb_recorder.record_image("trmasks", train_masks, curr_iter)
                tb_recorder.record_image("trsodout", train_preds.sigmoid(), curr_iter)
                tb_recorder.record_image("trsodin", train_inputs, curr_iter)
                write_data_to_file(log, path_config["tr_log"])


def test(model, mode="test", save_pre=True):
    model.eval()

    test_dataset_dict = user_config["rgb_data"]["te_data_list"]
    if mode == "val":
        test_dataset_dict = user_config["rgb_data"]["val_data_path"]

    total_results = {}
    for idx, (data_name, data_path) in enumerate(test_dataset_dict.items()):
        construct_print(f"Testing on the dataset: {data_name}, {data_path}")
        test_set = ImageFolder(root=data_path, in_size=user_config["input_size"], training=False)
        length = len(test_set)
        te_loader = create_loader(
            data_set=test_set,
            size_list=None,
            batch_size=batch_size_single_gpu,
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
            length=length,
            te_loader=te_loader,
            save_pre=save_pre,
            save_path=save_path,
        )
        msg = f"Results on the {mode}set({data_name}:'{data_path}'):\n{results}"
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
