import pandas as pd

from src.dataset.dataset import MealDataset, MovingDataset


def save_meal_stats(config):
    train_lens, test_lens = [], []
    train_stats, test_stats = [], []
    for patient_id in config["patient_ids"]:
        train_dataset = MealDataset(config, patient_id, "train")
        test_dataset = MealDataset(config, patient_id, "test", train_dataset.mean_std)
        train_lens.append(len(train_dataset))
        test_lens.append(len(test_dataset))
        train_stats.append(train_dataset.stats())
        test_stats.append(test_dataset.stats())
    df = pd.DataFrame(columns=config["patient_ids"])
    df.loc["train"] = train_lens
    df.loc["train hyper/not_hyper"] = ["{}/{}".format(x["hyper"], x["not_hyper"]) for x in train_stats]
    df.loc["train hypo/not_hypo"] = ["{}/{}".format(x["hypo"], x["not_hypo"]) for x in train_stats]
    df.loc["test"] = test_lens
    df.loc["test hyper/not_hyper"] = ["{}/{}".format(x["hyper"], x["not_hyper"]) for x in test_stats]
    df.loc["test hypo/not_hypo"] = ["{}/{}".format(x["hypo"], x["not_hypo"]) for x in test_stats]

    df["sum"] = [
        sum(train_lens),
        "{}/{}".format(
            sum([x["hyper"] for x in train_stats]),
            sum([x["not_hyper"] for x in train_stats]),
        ),
        "{}/{}".format(
            sum([x["hypo"] for x in train_stats]),
            sum([x["not_hypo"] for x in train_stats]),
        ),
        sum(test_lens),
        "{}/{}".format(
            sum([x["hyper"] for x in test_stats]),
            sum([x["not_hyper"] for x in test_stats]),
        ),
        "{}/{}".format(
            sum([x["hypo"] for x in test_stats]),
            sum([x["not_hypo"] for x in test_stats]),
        ),
    ]
    df.to_csv("./{}_stats_meal.csv".format(config["dataset"]))


def save_moving_stats(config):
    for task in ["hyper", "hypo"]:
        train_lens, test_lens = [], []
        train_stats, test_stats = [], []
        for patient_id in config["patient_ids"]:
            train_dataset = MovingDataset(config, task, patient_id, "train")
            test_dataset = MovingDataset(config, task, patient_id, "test", train_dataset.mean_std)
            train_lens.append(len(train_dataset))
            test_lens.append(len(test_dataset))
            train_stats.append(train_dataset.stats())
            test_stats.append(test_dataset.stats())
        df = pd.DataFrame(columns=config["patient_ids"])
        df.loc["train"] = train_lens
        df.loc["train {}/not_{}".format(task, task)] = ["{}/{}".format(x["pos"], x["neg"]) for x in train_stats]
        df.loc["test"] = test_lens
        df.loc["test {}/not_{}".format(task, task)] = ["{}/{}".format(x["pos"], x["neg"]) for x in test_stats]

        df["sum"] = [
            sum(train_lens),
            "{}/{}".format(
                sum([x["pos"] for x in train_stats]),
                sum([x["neg"] for x in train_stats]),
            ),
            sum(test_lens),
            "{}/{}".format(
                sum([x["pos"] for x in test_stats]),
                sum([x["neg"] for x in test_stats]),
            ),
        ]
        df.to_csv("./{}_stats_{}.csv".format(config["dataset"], task))
