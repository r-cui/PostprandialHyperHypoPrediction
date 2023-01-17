from matplotlib import pyplot as plt


def plot_meal(df, show=False, save_dir=None):
    """
    Args:
        df: sub-dataframe containing ["glucose", "basal", "bolus", "carbs"]
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    line_hyper = ax.plot(
        df["index"], [180] * len(df["index"]),
        color="r", linestyle="dashed",
        # label="Hyperglycemia thres"
    )
    line_hyper = ax.plot(
        df["index"], [70] * len(df["index"]),
        color="r", linestyle="dashed",
        # label="Hypoglycemia thres"
    )

    line_glucose = ax.plot(
        df["index"], df["glucose"],
        marker=".", markersize=10, color="blue",
        label="Glucose"
    )
    line_bolus = ax.plot(
        df["index"], df["bolus"],
        marker="v", markersize=10, color="black", linestyle="None",
        label="Bolus"
    )
    for i, j in zip(df["index"], df["bolus"]):
        plt.annotate(str(j), xy=(i, j+4))
    line_carbs = ax.plot(
        df["index"], df["carbs"],
        marker="^", markersize=10, color="red", linestyle="None",
        label="Carbs"
    )
    for i, j in zip(df["index"], df["carbs"]):
        plt.annotate(str(j), xy=(i, j+4))

    ax2 = ax.twinx()
    line_basal = ax2.plot(
        df["index"], df["basal"],
        color="c", linestyle="dashed",
        label="Basal"
    )

    ax.set_ylim(0, 400)
    ax2.set_ylim(0, 3.0)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax.grid()
    ax.xaxis.set_ticks(df["index"][::4])
    ax.xaxis.set_ticklabels(df["index"][::4], rotation=60, ha="right")

    plt.subplots_adjust(bottom=0.3)
    if show:
        plt.show()
    if not save_dir is None:
        plt.savefig(save_dir)
        plt.close()


def naming_plot(i, example):
    """ For saving a plot to local.
    """
    hyper_label = example["hyper_label"]
    hypo_label = example["hypo_label"]
    return "{}_{}_{}.png".format(i, "hyper" if hyper_label else "nohyper", "hypo" if hypo_label else "nohypo")
