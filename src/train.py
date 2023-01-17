import os
import torch
import random
import numpy as np
import pandas as pd
import yaml

from tqdm import tqdm
from pprint import pprint
from pathlib import Path

from src.model.model import Model
from src.utils.utils import n_params, ravel, load_config
from src.dataset.dataset_utils import prepare
from src.eval import Evaluator


def train(config, log_dir, patient_id):
    print("\n========== Training {} ==========".format(patient_id))
    scheme = config["scheme"]
    epochs = config["epoch"]
    assert scheme in ["cla", "reg"]

    train_dl_hyper, test_dl_hyper, train_dl_hypo, test_dl_hypo = prepare(config, patient_id)

    # get class weight
    class_count_hyper = np.array([0, 0])
    for i in range(len(train_dl_hyper.dataset)):
        if train_dl_hyper.dataset[i]["cla_label"].item() is True:
            class_count_hyper[0] += 1
        else:
            class_count_hyper[1] += 1
    class_weight_hyper = torch.from_numpy(class_count_hyper / np.sum(class_count_hyper)).to(torch.float32)
    class_count_hypo = np.array([0, 0])
    for i in range(len(train_dl_hypo.dataset)):
        if train_dl_hypo.dataset[i]["cla_label"].item() is True:
            class_count_hypo[0] += 1
        else:
            class_count_hypo[1] += 1
    class_weight_hypo = torch.from_numpy(class_count_hypo / np.sum(class_count_hypo)).to(torch.float32)

    model = Model(config, class_weight_hyper, class_weight_hypo)
    print("----Model has {} parameters, class weight hyper {}, hypo {}.".format(n_params(model), class_weight_hyper, class_weight_hypo))
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.gpu_mode()
    else:
        model.cpu_mode()

    for epoch in tqdm(
        range(1, epochs+1),
        desc="Training patient {} with lr {}".format(patient_id, model.optimizer.param_groups[0]["lr"])
    ):
        hyper_iter = iter(train_dl_hyper)
        hypo_iter = iter(train_dl_hypo)

        model.train_mode()
        while True:
            batch_hyper = None
            batch_hypo = None
            try:
                batch_hyper = next(hyper_iter)
            except StopIteration:
                pass
            try:
                batch_hypo = next(hypo_iter)
            except StopIteration:
                pass

            if batch_hyper is None and batch_hypo is None:
                break

            if batch_hyper is not None:
                loss_hyper = model.forward_train(batch_hyper)
                model.optimizer_step(loss_hyper)

            if batch_hypo is not None:
                loss_hypo = model.forward_train(batch_hypo)
                model.optimizer_step(loss_hypo)

        model.save_checkpoint(log_dir, patient_id)

    # ablation: individual training
    # dl = train_dl_hypo
    # for epoch in tqdm(
    #     range(1, epochs+1),
    #     desc="Training patient {} with lr {}".format(patient_id, model.optimizer.param_groups[0]["lr"])
    # ):
    #     dl_iter = iter(dl)
    #
    #     model.train_mode()
    #     while True:
    #         batch = None
    #         try:
    #             batch = next(dl_iter)
    #         except StopIteration:
    #             pass
    #
    #         if batch is None:
    #             break
    #
    #         if batch is not None:
    #             loss = model.forward_train(batch)
    #             model.optimizer_step(loss)
    #
    #     model.save_checkpoint(log_dir, patient_id)


if __name__ == '__main__':
    config = load_config("./src/config.yaml")

    for i in range(1):
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)

        log_dir = "./log/{}/{}".format(config["dataset"], i)
        Path(log_dir).mkdir(exist_ok=True, parents=True)

        for patient_id in config["patient_ids"]:
            train(config, log_dir, patient_id)
        with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
