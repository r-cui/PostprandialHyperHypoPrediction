import pandas as pd
import datetime as dt

from src.dataset.dataset_utils import prepare
from src.dataset.dataset import MealDataset
from src.utils.utils import load_config

from scipy.stats import ks_2samp


def get_postprandial(df, meal_idx):
    """ Check if the example is all consecutive in time."""
    res = 0
    for i in range(1, 49):
        if (meal_idx + i) >= len(df):
            return i
        if (df["index"].iloc[meal_idx+i] - df["index"].iloc[meal_idx]) < dt.timedelta(hours=4):
            res = i
    return res


if __name__ == '__main__':
    config = load_config("./src/config.yaml")
    for patient_id in config["patient_ids"]:
        ds = MealDataset(config, patient_id, "train")
        df = ds.df
        global_mean, global_std = df["glucose"].mean(), df["glucose"].std()

        meal_cgm_all = []
        for meal_idx in ds.all_meal_indices:
            meal_cgm = df.iloc[meal_idx: meal_idx + get_postprandial(df, meal_idx) + 1]
            meal_cgm_all.append(meal_cgm)
        meal_cgm_all = pd.concat(meal_cgm_all)
        postprandial_mean, postprandial_std = meal_cgm_all["glucose"].mean(), meal_cgm_all["glucose"].std()

        not_meal_cgm_all = df[~df.index.isin(meal_cgm_all.index)]
        non_postprandial_mean, non_postprandial_std = not_meal_cgm_all["glucose"].mean(), not_meal_cgm_all["glucose"].std()

        print("Patient {}".format(patient_id))
        print(ks_2samp(meal_cgm_all["glucose"].dropna(), not_meal_cgm_all["glucose"].dropna()))
        print("Global {:.1f} ({:.1f})".format(global_mean, global_std))
        print("Postprandial {:.1f} ({:.1f})".format(postprandial_mean, postprandial_std))
        print("Non postprandial {:.1f} ({:.1f})\n\n".format(non_postprandial_mean, non_postprandial_std))
