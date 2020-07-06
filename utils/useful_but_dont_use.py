# -*- coding: utf-8 -*-
# @Time    : 2020/7/5
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : useful_but_dont_use.py
# @Project : DistributedSODProj
# @GitHub  : https://github.com/lartpang

"""
一些有用的尝试，但是为了简单没有使用
"""


class SimpleImageFolder(Dataset):
    def __init__(self, root, in_size, prefix=(".jpg", ".png"), training=True):
        self.training = training

        if os.path.isdir(root):
            construct_print(f"{root} is an image folder")
            self.imgs = _make_dataset(root)
        elif os.path.isfile(root):
            construct_print(f"{root} is a list of images, we will read the corresponding image")
            self.imgs = _make_dataset_from_list(root, prefix=prefix)
        else:
            print(f"{root} is invalid")
            raise NotImplementedError

        if self.training:
            self.train_joint_transform = Compose([JointResize(in_size), RandomHorizontallyFlip(), RandomRotate(10)])
            self.train_img_transform = transforms.Compose([transforms.ColorJitter(0.1, 0.1, 0.1),])
        else:
            self.test_img_trainsform = transforms.Compose(
                [
                    # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
                    transforms.Resize((in_size, in_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        img = Image.open(img_path).convert("RGB")
        if self.training:
            mask = Image.open(mask_path).convert("L")
            img, mask = self.train_joint_transform(img, mask)
            img = self.train_img_transform(img)
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0)
            return img, mask
        else:
            img_name = (img_path.split(os.sep)[-1]).split(".")[0]
            img = self.test_img_trainsform(img)
            return img, img_name, mask_path

    def __len__(self):
        return len(self.imgs)


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            self.next_target = self.next_target.float().div_(255)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def train_epoch_data_prefetcher(curr_epoch, end_epoch, local_rank, loss_funcs, model, optimizer, scheduler, tr_loader):
    model.train()
    train_loss_record = AvgMeter()
    prefetcher = DataPrefetcher(tr_loader)
    train_inputs, train_masks = prefetcher.next()
    train_batch_id = 0
    while train_inputs is not None:
        curr_iter = curr_epoch * len(tr_loader) + train_batch_id
        if user_config["sche_usebatch"]:
            scheduler.step(optimizer, curr_epoch=curr_iter)
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
            reduced_loss = reduce_tensor(train_loss)
        else:
            reduced_loss = train_loss
        train_iter_loss = reduced_loss.item()
        train_loss_record.update(train_iter_loss, train_inputs.size(0))

        if local_rank == 0:
            lr_str = ",".join([f"{param_groups['lr']:.7f}" for param_groups in optimizer.param_groups])
            log = (
                f"[I:{train_batch_id}/{len(tr_loader)}/{curr_iter}/{total_iter_num}][E:{curr_epoch}:{end_epoch}]>["
                f"{exp_name}]"
                f"[Lr:{lr_str}][Avg:{train_loss_record.avg:.5f}|Cur:{train_iter_loss:.5f}|"
                f"{loss_item_list}]"
            )
            if user_config["print_freq"] > 0 and (curr_iter + 1) % user_config["print_freq"] == 0:
                print(log)
            if user_config["record_freq"] > 0 and (curr_iter + 1) % user_config["record_freq"] == 0:
                tb_recorder.record_curve("trloss_avg", train_loss_record.avg, curr_iter)
                tb_recorder.record_curve("trloss_iter", train_loss_record.avg, curr_iter)
                tb_recorder.record_curve("lr", optimizer.param_groups, curr_iter)
                tb_recorder.record_image("trmasks", train_masks, curr_iter)
                tb_recorder.record_image("trsodout", train_preds.sigmoid(), curr_iter)
                tb_recorder.record_image("trsodin", train_inputs, curr_iter)
                tr_recorder.record_str(log)

        train_inputs, train_masks = prefetcher.next()
        train_batch_id += 1


def train_epoch_original(curr_epoch, end_epoch, local_rank, loss_funcs, model, optimizer, scheduler, tr_loader):
    model.train()

    train_loss_record = AvgMeter()

    for train_batch_id, (train_inputs, train_masks) in enumerate(tr_loader):
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
            reduced_loss = reduce_tensor(train_loss)
        else:
            reduced_loss = train_loss
        train_iter_loss = reduced_loss.item()
        train_loss_record.update(train_iter_loss, train_inputs.size(0))

        if local_rank == 0:
            lr_str = ",".join([f"{param_groups['lr']:.7f}" for param_groups in optimizer.param_groups])
            log = (
                f"[I:{train_batch_id}/{len(tr_loader)}/{curr_iter}/{total_iter_num}][E:{curr_epoch}:{end_epoch}]>["
                f"{exp_name}]"
                f"[Lr:{lr_str}][Avg:{train_loss_record.avg:.5f}|Cur:{train_iter_loss:.5f}|"
                f"{loss_item_list}]"
            )
            if user_config["print_freq"] > 0 and (curr_iter + 1) % user_config["print_freq"] == 0:
                print(log)
            if user_config["record_freq"] > 0 and (curr_iter + 1) % user_config["record_freq"] == 0:
                tb_recorder.record_curve("trloss_avg", train_loss_record.avg, curr_iter)
                tb_recorder.record_curve("trloss_iter", train_loss_record.avg, curr_iter)
                tb_recorder.record_curve("lr", optimizer.param_groups, curr_iter)
                tb_recorder.record_image("trmasks", train_masks, curr_iter)
                tb_recorder.record_image("trsodout", train_preds.sigmoid(), curr_iter)
                tb_recorder.record_image("trsodin", train_inputs, curr_iter)
                tr_recorder.record_str(log)
