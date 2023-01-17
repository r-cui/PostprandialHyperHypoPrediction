from torch.utils.data import DataLoader

from src.dataset.dataset import MovingDataset
# from src.dataset.dataset_full import MovingDataset


def prepare(config, patient_id):
    train_dataset_hyper = MovingDataset(config, "hyper", patient_id, "train")
    test_dataset_hyper = MovingDataset(config, "hyper", patient_id, "test", external_mean_std=dict(train_dataset_hyper.mean_std))
    train_dataset_hypo = MovingDataset(config, "hypo", patient_id, "train", external_mean_std=dict(train_dataset_hyper.mean_std))
    test_dataset_hypo = MovingDataset(config, "hypo", patient_id, "test", external_mean_std=dict(train_dataset_hyper.mean_std))

    batch_size = config["batch_size"]
    train_dataloader_hyper = DataLoader(
        train_dataset_hyper,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader_hyper = DataLoader(
        test_dataset_hyper,
        batch_size=batch_size,
        shuffle=True
    )
    train_dataloader_hypo = DataLoader(
        train_dataset_hypo,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader_hypo = DataLoader(
        test_dataset_hypo,
        batch_size=batch_size,
        shuffle=True
    )
    return train_dataloader_hyper, test_dataloader_hyper, \
           train_dataloader_hypo, test_dataloader_hypo
