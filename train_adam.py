from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from terminaltables import AsciiTable
import os
import sys
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd


class Trainer:
    def __init__(self, model, class_names, optimizer=None, device='cuda'):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        if optimizer is None:
            # betas = (float(model.hyperparams["beta1"]), float(model.hyperparams["beta2"]))
            # float(model.hyperparams["learning_rate"])
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                              # eps=float(model.hyperparams["eps"]),
                                              weight_decay=0)
        else:
            self.optimizer = optimizer
        self.device = device
        model.to(self.device)
        self.class_names = class_names

    def fit(self, dl_train: DataLoader, dl_valid: DataLoader, num_epochs=100, img_size=416,
            evaluation_interval=1, gradient_accumulations=2, verbose=True, start_eval=6, file_suffix_name="",
            initional_epoch=0, initional_max_mAP=0):
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_valid: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param img_size: size of each image dimension"
        :param evaluation_interval: interval evaluations on validation set
        :param gradient_accumulations: number of gradient accums before step
        :param verbose: If it is true the data will printed.


        """

        losses_train = []
        losses_valid = []
        APs = []
        precisions = []
        recalls = []
        mAPs = []
        f1s = []
        maximal_map = initional_max_mAP

        for epoch in range(initional_epoch, num_epochs):

            loss_train = self.train_epoch(dl_train, gradient_accumulations, epoch)

            if epoch % evaluation_interval == 0 and epoch > start_eval:
                losses_train.append(loss_train)

                # Evaluate the model on the validation set
                epochValidRes = self.test_epoch(
                    model=self.model,
                    dataloader=dl_valid,
                    iou_thres=0.5,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    img_size=img_size,
                    epoch_num=epoch,
                    mode="valid",
                )

                losses_valid.append(epochValidRes["loss"])
                APs.append(epochValidRes["AP"])
                precisions.append(epochValidRes["precision"])
                recalls.append(epochValidRes["recall"])
                mAPs.append(epochValidRes["mAP"])
                f1s.append(epochValidRes["f1"])

                if epochValidRes["mAP"] > maximal_map:
                    # Print class APs and mAP
                    if verbose:
                        ap_table = [["Index", "Class name", "AP"]]
                        for i, c in enumerate(epochValidRes["ap_class"]):
                            ap_table += [[c, self.class_names[c], "%.5f" % epochValidRes["AP"][i]]]

                    self._print("\n", verbose)
                    self._print(AsciiTable(ap_table).table, verbose)
                    maximal_map = epochValidRes["mAP"]
                    torch.save(self.model.state_dict(), f"checkpoints/model.pth")
                    self.save(losses_train, losses_valid, APs, mAPs, precisions, recalls, f1s, file_suffix_name)
        self.save(losses_train, losses_valid, APs, mAPs, precisions, recalls, f1s, file_suffix_name)

    def train_epoch(self, dl_train: DataLoader, gradient_accumulations, epoch_num):
        self.model.train()
        num_batches = len(dl_train)

        with tqdm.tqdm(desc=f'{"Epoch"} ({epoch_num}) {"progress"}', total=num_batches) as pbar:
            losses_list = []
            for batch_i, (_, imgs, targets) in enumerate(dl_train):
                batches_done = len(dl_train) * epoch_num + batch_i

                imgs = Variable(imgs.to(self.device))
                targets = Variable(targets.to(self.device), requires_grad=False)

                loss, output1 = self.model(imgs, targets)
                losses_list.append(loss.item())
                loss.backward()

                if batches_done % gradient_accumulations:
                    # Accumulates gradient before each step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.model.seen += imgs.size(0)
                pbar.update()
            loss_mean = np.array(losses_list).mean()
            pbar.set_postfix(train_loss=loss_mean)
            pbar.update()

            return loss_mean

    def test_epoch(self, model, dataloader: DataLoader, iou_thres, conf_thres, nms_thres,
                   img_size: int, epoch_num: int, mode):
        model.eval()

        if mode == "train":
            __print = "training set"
        elif mode == "valid":
            __print = "validation set"
        else:
            __print = ""
        losses_list = []
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        # for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=__print)):
        num_batches = len(dataloader)

        with tqdm.tqdm(desc=f'{"Epoch"} ({epoch_num }) {__print}{" evaluation"}', total=num_batches) as pbar:

            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                # print(targets)
                # print("###")

                with torch.no_grad():
                    imgs_cuda = Variable(imgs.to(self.device))
                    targets_cuda = Variable(targets.to(self.device), requires_grad=False)

                    loss, outputs = model(imgs_cuda, targets_cuda)
                    losses_list.append(loss.item())
                    # Extract labels
                    labels += targets[:, 1].tolist()
                    # Rescale target
                    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                    targets[:, 2:] *= img_size
                    outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
                pbar.update()
                sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

            # Concatenate sample statistics
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
            loss_mean = np.array(losses_list).mean()
            pbar.set_description(f'{"Epoch"} ({epoch_num }) {__print}{" evaluation"}')
            pbar.set_postfix(mAP=AP.mean(), valid_loss=loss_mean)
            pbar.update()
        return {"epoch_num": epoch_num, "AP": AP, "precision": precision.mean(), "recall": recall.mean(),
                "mAP": AP.mean(), "f1": f1.mean(), "ap_class": ap_class, "loss": loss_mean}

    @staticmethod
    def save(train_losses, valid_losses, APs, mAPs, precisions, recalls, f1s, file_suffix_name):
        df = pd.DataFrame(APs)
        df.to_csv("APs_"+file_suffix_name+".csv")
        rest = [train_losses, valid_losses, mAPs, precisions, recalls, f1s]
        rest = np.array(rest)
        rest = np.transpose(rest)
        df1 = pd.DataFrame(rest, columns=['train_losses', 'valid_losses', 'mAPs', 'precisions', 'recalls', 'f1s'])
        df1.to_csv(os.path.join("results", "rest_"+file_suffix_name+".csv"))

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)


if __name__ == "__main__":
    np.random.seed(seed=42)
    torch.manual_seed(seed=42)
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg",
                        help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str,default=None, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--num_epochs", type=int, default=500, help="number of epochs")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    print(class_names)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    print(model.hyperparams)
    model.apply(weights_init_normal)

    # If specified w start from checkpoint
    if opt.pretrained_weights is not None:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
            print("pth was loaded")
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    # Get dataloader
    dataset_train = ListDataset(
        train_path,
        augment=True,
        multiscale=opt.multiscale_training,
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset_train.collate_fn,
    )

    dataset_valid = ListDataset(
        valid_path,
        img_size=opt.img_size,
        augment=True,
        multiscale=False,
    )

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.n_cpu,
        collate_fn=dataset_valid.collate_fn
    )

    train =Trainer(model, class_names=class_names, device=device)

    # epochValidResHeads1 = train.test_epoch(
    #     model=model,
    #     dataloader=dataloader_valid,
    #     iou_thres=0.5,
    #     conf_thres=0.5,
    #     nms_thres=0.5,
    #     img_size=416,
    #     epoch_num=1,
    #     mode="valid",
    #     main_head= True
    # )

    train.fit(
        dl_train=dataloader_train,
        dl_valid=dataloader_valid,
        num_epochs=opt.num_epochs,
        file_suffix_name="monitor",
        initional_epoch=0,
        initional_max_mAP=0
    )
