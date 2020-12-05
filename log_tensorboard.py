from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("log/not_categorized")
labels = ["Train Acc", "Test Acc", "Train Loss", "Test Loss"]


def init(exp_id: str) -> None:
    global writer
    writer = SummaryWriter("log/" + exp_id)


def save() -> None:
    global writer
    writer.flush()


def add(values: tuple, epoch_idx: int) -> None:
    global writer
    for i in range(len(values)):
        writer.add_scalar(labels[i], values[i], epoch_idx)
    print("Epoch", epoch_idx, "added to board log")
