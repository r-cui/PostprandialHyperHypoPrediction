import os.path

import torch
import numpy as np

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.utils.utils import prepare_confmat, ravel, de_standardize, load_config
from src.dataset.dataset_utils import prepare
from src.model.model import Model


class PatientResult:
    def __init__(self, conf_mat, rmse=None, decision_time=None):
        self.conf_mat = conf_mat
        self.rmse = rmse
        self.decision_time = decision_time
        self.tn, self.fp, self.fn, self.tp, self.se, self.sp, self.fa, self.mcc = ravel(conf_mat)


class PatientsResult:
    def __init__(self):
        self.l = []
        self.score = None

    def append(self, result):
        self.l.append(result)

    def stats(self):
        sum_conf_mat = np.zeros((2, 2), dtype=np.int64)
        for x in self.l:
            sum_conf_mat = sum_conf_mat + prepare_confmat(x.conf_mat)
        tn, fp, fn, tp, se, sp, fa, mcc = ravel(sum_conf_mat)
        def nan_mean(l):
            l = np.array(l).astype(float)
            l = l[~np.isnan(l)]
            return np.mean(l)
        rmse_mean = nan_mean([x.rmse for x in self.l])
        decision_time_mean = nan_mean([x.decision_time for x in self.l])

        print("SE={}, SP={}, FA={}, MCC={}, RMSE={}, Time={}".format(
            se, sp, fa, mcc, rmse_mean, round(decision_time_mean * 5.0, 2)
        ))
        self.score = {
            "se": se,
            "sp": sp,
            "fa": fa,
            "mcc": mcc,
            "rmse": rmse_mean,
            "dt": round(decision_time_mean * 5.0, 2)
        }


class Evaluator(object):
    def __init__(self, config):
        self.scheme = config["scheme"]
        self.hyper_thres = config["hyper_thres"]
        self.hypo_thres = config["hypo_thres"]

    def eval_single(self, model, dl, task):
        if task == "hyper":
            condition = lambda x: x >= self.hyper_thres
        elif task == "hypo":
            condition = lambda x: x <= self.hypo_thres
        else:
            raise

        model.eval_mode()
        preds_cla, gts_cla, preds_reg, gts_reg = [], [], [], []
        pos_times = []
        meals = []
        with torch.no_grad():
            for batch in dl:
                if self.scheme == "cla":
                    pred_cla = model.forward_eval(batch)
                else:
                    pred_reg = model.forward_eval(batch)
                    pred_reg = de_standardize(pred_reg, dl.dataset.mean_std)
                    gt_reg = de_standardize(batch["reg_label"], dl.dataset.mean_std)
                    preds_reg.append(pred_reg)
                    gts_reg.append(gt_reg)
                    pred_cla = condition(pred_reg)
                gt_cla = batch["cla_label"]
                preds_cla.append(pred_cla)
                gts_cla.append(gt_cla)

                pos_times.append(batch["pos_time"])
                meals.append(batch["meal"])

        preds_cla = torch.cat(preds_cla, dim=0)
        gts_cla = torch.cat(gts_cla, dim=0)
        pos_times = torch.cat(pos_times, dim=0)
        meals = torch.cat(meals, dim=0)

        conf_mat = prepare_confmat(confusion_matrix(gts_cla.cpu(), preds_cla.cpu()))

        # the averaged event time in all correctly predicted examples
        avg_pos_time = torch.mean(torch.masked_select(pos_times, torch.logical_and(preds_cla == True, gts_cla == True))).item()
        # the detection time averaged by meals
        unique_meals = torch.unique(meals)
        detection_times = []
        for meal in unique_meals:
            pool = (meals == meal) & (preds_cla == True) & (gts_cla == True)
            if pool.any():
                detection_times.append(torch.max(torch.masked_select(pos_times, pool)))
        avg_detection_time = torch.mean(torch.tensor(detection_times)).item()

        rmse = None
        if self.scheme == "reg":
            preds_reg = torch.cat(preds_reg, dim=0)
            gts_reg = torch.cat(gts_reg, dim=0)
            rmse = round(torch.sqrt(torch.mean(torch.square(preds_reg - gts_reg))).item(), 2)
        else:
            pass
        return PatientResult(conf_mat, rmse, avg_detection_time)

    def eval_joint(self, model, hyper_dl, hypo_dl):
        return self.eval_single(model, hyper_dl, "hyper"), self.eval_single(model, hypo_dl, "hypo")


def eval(config, pretrain_dir, patient_id):
    print("\n========== Evaluating {} ==========".format(patient_id))
    train_dl_hyper, test_dl_hyper, train_dl_hypo, test_dl_hypo = prepare(config, patient_id)
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
    model.load_checkpoint(pretrain_dir, patient_id)

    hyper_result, hypo_result = Evaluator(config).eval_joint(model, test_dl_hyper, test_dl_hypo)
    return hyper_result, hypo_result


def kfold_eval(logs_dir):
    hypers, hypos = [], []
    for folder in sorted(os.listdir(logs_dir)):
        log_dir = os.path.join(logs_dir, folder)
        config = load_config(os.path.join(log_dir, "config.yaml"))
        hyper, hypo = PatientsResult(), PatientsResult()
        for patient_id in config["patient_ids"]:
            a, b = eval(config, log_dir, patient_id)
            hyper.append(a)
            hypo.append(b)
        hyper.stats()
        hypo.stats()
        hypers.append(hyper)
        hypos.append(hypo)

    print("Hyper")
    for key in hypers[0].score.keys():
        l = np.array([x.score[key] for x in hypers])
        print(key, round(np.mean(l), 2), round(np.std(l), 2))
    print("Hypo")
    for key in hypers[0].score.keys():
        l = np.array([x.score[key] for x in hypos])
        print(key, round(np.mean(l), 2), round(np.std(l), 2))


if __name__ == '__main__':
    kfold_eval("./pretrained/ohiot1dm")
