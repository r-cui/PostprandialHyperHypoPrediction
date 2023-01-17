import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    def __init__(self, config, class_weight_hyper, class_weight_hypo):
        super(Model, self).__init__()
        self.scheme = config["scheme"]
        self.lr = config["lr"]
        self.hidden_size = config["hidden_size"]
        assert self.scheme in ["cla", "reg"]
        self.class_weight_hyper, self.class_weight_hypo = class_weight_hyper, class_weight_hypo  # tensor([float, float])
        self._init()

    def _init(self):
        # networks structure
        hidden_size = self.hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.relu = nn.ReLU()

        self.cla_linear_hyper = nn.Linear(hidden_size, 2)
        self.cla_linear_hypo = nn.Linear(hidden_size, 2)
        self.reg_linear_hyper = nn.Linear(hidden_size, 1)
        self.reg_linear_hypo = nn.Linear(hidden_size, 1)

        self.cla_loss_hyper = nn.CrossEntropyLoss(weight=self.class_weight_hyper)
        self.cla_loss_hypo = nn.CrossEntropyLoss(weight=self.class_weight_hypo)
        self.reg_loss = nn.MSELoss()

        # create optimizer, scheduler
        self._init_miscs()

        # single GPU assumed
        self.use_gpu = False
        self.device = None
        self.gpu_device = torch.device("cuda:0")
        self.cpu_device = torch.device("cpu")
        self.cpu_mode()

    def network_forward(self, batch):
        batch = self._prepare_batch(batch)
        glucose = batch["glucose_input"]
        lstm_out, (lstm_hn, lstm_cn) = self.lstm(glucose.unsqueeze(-1))
        return self.relu(lstm_hn[0])  # (BS, hidden_size)

    def forward_train(self, batch):
        """
        Returns:
            loss: (1)
        """
        task = batch["task"][0]
        network_out = self.network_forward(batch)
        if self.scheme == "cla":
            gt = batch["cla_label"]
            if task == "hyper":
                pred = self.cla_linear_hyper(network_out)  # (BS, 2)
                loss = self.cla_loss_hyper(pred, gt)
            elif task == "hypo":
                pred = self.cla_linear_hypo(network_out)  # (BS, 2)
                loss = self.cla_loss_hypo(pred, gt)
            else:
                raise

        elif self.scheme == "reg":
            if task == "hyper":
                pred = self.reg_linear_hyper(network_out)  # (BS, 1)
            elif task == "hypo":
                pred = self.reg_linear_hypo(network_out)  # (BS, 1)
            else:
                raise
            gt = batch["reg_label"]
            loss = self.reg_loss(pred[:, 0], gt)
        else:
            raise
        return loss

    def forward_eval(self, batch):
        task = batch["task"][0]
        network_out = self.network_forward(batch)
        if self.scheme == "cla":
            if task == "hyper":
                pred = self.cla_linear_hyper(network_out)  # (BS, 2)
            elif task == "hypo":
                pred = self.cla_linear_hypo(network_out)  # (BS, 2)
            else:
                raise
            pred = torch.argmax(pred, dim=1)
        else:
            if task == "hyper":
                pred = self.reg_linear_hyper(network_out)  # (BS, 1)
            elif task == "hypo":
                pred = self.reg_linear_hypo(network_out)  # (BS, 1)
            else:
                raise
            pred = pred[:, 0]
            # pred = batch["glucose_input"][:, -1]  # baseline: dummy (change epoch to 1 to make running fast)
        return pred  # (BS)

    ##### below are helpers #####
    def _init_miscs(self):
        """
        Key attributes created here:
            - self.optimizer
            - self.scheduler
        """
        lr = self.lr
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def _prepare_batch(self, batch):
        for key in batch.keys():
            if not isinstance(batch[key], list):
                batch[key] = batch[key].to(self.device)
        for key in ["glucose_input", "reg_label"]:
            batch[key] = batch[key].to(torch.float32)
        for key in ["cla_label"]:
            batch[key] = batch[key].to(torch.long)
        return batch

    def optimizer_step(self, loss):
        """ Update the network.
        """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

    def load_checkpoint(self, exp_folder_path, suffix):
        self.load_state_dict(torch.load(os.path.join(exp_folder_path, "{}.pt".format(suffix))))
        print("== Checkpoint ({}) is loaded from {}".format(suffix, exp_folder_path))

    def save_checkpoint(self, exp_folder_path, suffix):
        torch.save(self.state_dict(), os.path.join(exp_folder_path, "{}.pt".format(suffix)))
        # print("== Checkpoint ({}) is saved to {}".format(suffix, exp_folder_path))

    def cpu_mode(self):
        self.use_gpu = False
        self.to(self.cpu_device)
        self.device = self.cpu_device

    def gpu_mode(self):
        self.use_gpu = True
        self.to(self.gpu_device)
        self.device = self.gpu_device

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()
