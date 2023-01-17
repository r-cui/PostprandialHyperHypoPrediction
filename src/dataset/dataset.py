import copy
import os.path

import numpy as np
import pandas as pd
import datetime as dt
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter

from src.utils.utils import load_config
from src.utils.plot_utils import plot_meal, naming_plot

pd.set_option('display.max_rows', None)


class MealDataset(Dataset):
    """ Basic dataset, each example is built with a meal.
    Not for training but for dataset stats.
    """
    def __init__(self, config, patient_id, split, external_mean_std=None):
        """
        Args:
            external_mean_std: {
                "mean": float,
                "std": float
            }
                If none, self fit.
        """
        self.config = config
        self.dataset = config["dataset"]
        assert self.dataset in ["ohiot1dm", "umt1dm"]
        self.hyper_thres = config["hyper_thres"]
        self.hypo_thres = config["hypo_thres"]
        self.input_len = config["input_len"]
        self.future_len = config["future_len"]
        # time considered around a meal: (input_len + 1 + future_len)

        self.patient_id = patient_id
        self.split = split
        self.df = pd.read_csv("./data/{}/preprocessed/{}_{}.csv".format(self.dataset, patient_id, split))
        self.df.replace(to_replace=-1, value=np.nan, inplace=True)
        self.df["index"] = pd.to_datetime(self.df["index"])

        # meal_indices are those without missing entry
        self.meal_indices, self.all_meal_indices = self._get_aftermeal_examples()

        # standardize
        self.mean_std = external_mean_std if not external_mean_std is None else self._get_meanstd()

        print(
            "== {} {}: {}/{} meals.".format(
                self.patient_id, self.split, len(self.meal_indices), len(self.all_meal_indices)
            )
        )

    def _get_aftermeal_examples(self):
        """ Screen the full dataset and finds meals.
        """
        if self.dataset == "ohiot1dm":
            all_meal_indices = self.df.loc[~pd.isna(self.df["carbs"])].index.to_list()
        else:
            all_meal_indices = []
            for i in range(len(self.df)):
                timestamp = self.df["index"].iloc[i]
                if timestamp.time() in [dt.time(hour=8), dt.time(hour=12), dt.time(hour=18)]:  # [8:00, 12:00, 18:00]
                    all_meal_indices.append(i)

        def all_consecutive(example):
            """ Check if the example is all consecutive in time."""
            for i in range(1, len(example)):
                if (example["index"].iloc[i] - example["index"].iloc[i-1]) != dt.timedelta(minutes=5):
                    return False
            return True

        meal_indices = []
        for x in all_meal_indices:
            example = self.df.iloc[x-self.input_len: x+self.future_len+1]
            # 1. exclude meals at the start/end of dataset
            if (x - self.input_len < 0) or (x + self.future_len) >= len(self.df):
                continue
            # 2. no missing entry in each example
            elif pd.isna(example["glucose"]).any():
                continue
            elif not all_consecutive(example):
                continue
            else:
                meal_indices.append(x)
        return meal_indices, all_meal_indices

    def _get_meanstd(self):
        """ Self-fit the mean and std of this dataset, only run for training sets.
        """
        mean_std = {
            "mean": dict(),
            "std": dict()
        }
        for col in ["glucose"]:
            mean_std["mean"][col] = np.nanmean(self.df[col].to_numpy())
            mean_std["std"][col] = np.nanstd(self.df[col].to_numpy())
        return mean_std

    def get_meal_scenario(self, meal_idx):
        """ To extract a meal from dataset.

        Returns:
            history_start, future_end: such that [history_start, future_end] inclusive
        """
        history_start = meal_idx - self.input_len
        future_end = meal_idx + self.future_len
        return history_start, future_end

    ## below for meal stats
    def __len__(self):
        return len(self.meal_indices)

    def __getitem__(self, i):
        """
        All numpy array.
        {
            "glucose": (input_len + 1 + future_len)
            "peak_2h": float
            "bottom_2h_4h": float
        }
        """
        meal_idx = self.meal_indices[i]
        history_start, future_end = self.get_meal_scenario(meal_idx)
        glucose = self.df["glucose"].iloc[history_start: future_end+1].to_numpy()

        # label
        peak_2h = self.df["glucose"].iloc[meal_idx+1: meal_idx+1+24].max()
        bottom_2h_4h = self.df["glucose"].iloc[meal_idx+24+1: meal_idx+48+1].min()
        hyper_label = True if peak_2h >= self.hyper_thres else False
        hypo_label = True if bottom_2h_4h <= self.hypo_thres else False

        res = {
            "hyper_label": hyper_label,
            "hypo_label": hypo_label
        }
        return res

    def stats(self):
        cnt_hyper = Counter([self[i]["hyper_label"] for i in range(len(self))])
        cnt_hypo = Counter([self[i]["hypo_label"] for i in range(len(self))])
        return {
            "hyper": cnt_hyper[True],
            "not_hyper": cnt_hyper[False],
            "hypo": cnt_hypo[True],
            "not_hypo": cnt_hypo[False]
        }

    ## below for plotting a meal
    def get_df_from_meal_idx(self, idx):
        history_start = idx - self.input_len
        future_end = idx + self.future_len
        return self.df.iloc[history_start: future_end + 1]

    def get_df_of_ith_meal(self, i):
        meal_idx = self.meal_indices[i]
        return self.get_df_from_meal_idx(meal_idx)


class MovingDataset(MealDataset):
    """ Multiple examples are extracted per meal.
    """
    def __init__(self, config, task, patient_id, split, external_mean_std=None):
        super(MovingDataset, self).__init__(
            config=config, patient_id=patient_id, split=split, external_mean_std=external_mean_std
        )
        self.ph = self.config["ph"]  # ph only exists for moving pred
        self.task = task  # "hyper"/"hypo"
        self.window = self.config["{}_window".format(task)]
        self.examples = self._get_moving_examples()
        print(
            "== {} {} (moving): {}/{} meals, {} examples.".format(
                self.patient_id, self.split, len(self.meal_indices), len(self.all_meal_indices), len(self)
            )
        )

    def _get_moving_examples(self):
        """ Run the "moving" screening to extract all examples.
        Window selected is different to hyper/hypo task according to config.
        """
        res = []  # [(meal_id, shift)]
        for meal_idx in self.meal_indices:
            for shift in range(self.window[0], self.window[1] - self.ph + 1):
                # skipping prediction when already abnormal
                if self.task == "hyper":
                    if self.df["glucose"].iloc[meal_idx+shift] > self.hyper_thres:
                        break
                elif self.task == "hypo":
                    if self.df["glucose"].iloc[meal_idx+shift] < self.hypo_thres:
                        break
                else:
                    raise Exception("Wrong task given.")
                res.append((meal_idx, shift))
        return res

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        meal_idx, shift = self.examples[i]
        history_start, future_end = self.get_meal_scenario(meal_idx)
        glucose = self.df["glucose"].iloc[history_start: future_end+1].to_numpy()
        glucose_input = self.df["glucose"].iloc[meal_idx+shift-self.input_len+1: meal_idx+shift+1].to_numpy()

        # regression
        glucose_ph = self.df["glucose"].iloc[meal_idx+shift+1: meal_idx+shift+self.ph+1].to_numpy()

        def two_consecutive_true(array):
            """
            array: (n) boolean
            """
            for i in range(len(array)-1):
                if array[i] and array[i+1]:
                    return np.True_
            return np.False_

        if self.task == "hyper":
            reg_label = np.max(glucose_ph)
            cla_label = two_consecutive_true(glucose_ph >= self.hyper_thres)
        elif self.task == "hypo":
            reg_label = np.min(glucose_ph)
            cla_label = two_consecutive_true(glucose_ph <= self.hypo_thres)
        else:
            raise

        glucose_input = (glucose_input - self.mean_std["mean"]["glucose"]) / self.mean_std["std"]["glucose"]
        reg_label = (reg_label - self.mean_std["mean"]["glucose"]) / self.mean_std["std"]["glucose"]

        pos_time = -1
        if cla_label == True:
            if self.task == "hyper":
                pos_time = np.argmax(glucose_ph >= self.config["hyper_thres"]) + 1
            elif self.task == "hypo":
                pos_time = np.argmax(glucose_ph <= self.config["hypo_thres"]) + 1
            else:
                raise Exception("Wrong task given.")

        res = {
            "task": self.task,
            "glucose": glucose,
            "shift": shift,
            "meal": meal_idx,

            "glucose_input": glucose_input,  # (input_len)
            "reg_label": reg_label,  # (1)
            "cla_label": cla_label,

            "pos_time": float(pos_time)
        }
        return res

    def stats(self):
        cnt = Counter([self[i]["cla_label"] for i in range(len(self))])
        return {
            "pos": cnt[True],
            "neg": cnt[False],
        }


# for plotting all meals
def save_plots(config):
    patient_ids = [540, 544, 552, 584, 596, 559, 563, 570, 575, 588, 591]
    assert config["dataset"] == "ohiot1dm"
    for patient_id in patient_ids:
        train_dataset = MealDataset(config, patient_id, "train")
        test_dataset = MealDataset(config, patient_id, "test", train_dataset.mean_std)

        plot_dir = "./plot"
        for dataset in [train_dataset, test_dataset]:
            plot_subdir = os.path.join(plot_dir, "{}_{}".format(dataset.patient_id, dataset.split))
            Path(plot_subdir).mkdir(parents=True, exist_ok=True)
            for i in range(len(dataset)):
                plot_meal(
                    df=dataset.get_df_of_ith_meal(i),
                    show=False,
                    save_dir=os.path.join(plot_subdir, naming_plot(i, dataset[i]))
                )


# if __name__ == '__main__':
#     from src.utils.stats_utils import save_meal_stats, save_moving_stats
#
#     config = load_config("./src/config.yaml")
#     save_meal_stats(config)
#     save_moving_stats(config)
