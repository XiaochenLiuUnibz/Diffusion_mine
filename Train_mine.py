
import os
from typing import Dict
import cv2
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torch.utils.data import Dataset
from Diffusion_mine import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
from Scheduler_mine import GradualWarmupScheduler


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self._root_dir = root_dir
        self._transform = transform
        self._images_list = os.listdir(self._root_dir)

    def __len__(self):
        return len(self._images_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self._root_dir,
                                self._images_list[idx])
        image = cv2.imread(img_name)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._transform:
            image = self._transform(image)

        return image


def train(modelConfig: Dict):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),  #依概率p水平翻转transforms
            transforms.ToTensor(),              #变为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #归一化（参数为 均值 方差）
        ]))

    # def read_directory(directory_name):
    #     list_all_img = []
    #     for filename in os.listdir(directory_name):
    #         # print(filename)
    #         img = cv2.imread(directory_name + "/" + filename)
    #         # img = cv2.resize(img, None, fx=0.5, fy=0.5)  # 固定比例
    #         img = cv2.resize(img, (32, 32))  # 固定比例
    #         list_all_img.append(img)
    #     list_all_img_arr = np.array(list_all_img)
    #
    #     print()
    #     return list_all_img_arr


    my_dataset = MyDataset(
        root_dir="/Users/liuxiaochen/Documents/Super Resolution/code/Diffusion_mine/WHU-RS19",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([32, 32]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    # rs_data=read_directory("D:\AAA-PycharmWorkSpace\PHD\DenoisingDiffusionProbabilityModel-ddpm--main\Diffusion_mine\WHU-RS19")
    dataloader = DataLoader(
        my_dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)


    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))


    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)


    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)


    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)


    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                # loss = trainer(x_0.type(torch.float32)).sum() / 1000.
                loss = trainer(x_0).sum() / 1000.

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        # device = torch.device(modelConfig["device"])
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        # ckpt = torch.load(os.path.join(
        #     modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location="cpu")
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])