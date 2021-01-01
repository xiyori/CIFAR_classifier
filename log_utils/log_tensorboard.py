from torch.utils.tensorboard import SummaryWriter

writer = None  # SummaryWriter("log/not_categorized")
scalar_labels = ["Train Acc", "Test Acc", "Train Loss", "Test Loss"]
image_labels = ["Confusion matrix"]


def init(exp_id: str) -> None:
    global writer
    writer = SummaryWriter("log/" + exp_id)


def save() -> None:
    global writer
    writer.flush()


def add(epoch_idx: int, scalars: tuple=None, images: tuple=None) -> None:
    global writer
    if scalars is not None:
        for i in range(len(scalars)):
            writer.add_scalar(scalar_labels[i], scalars[i], epoch_idx)
    if images is not None:
        for i in range(len(images)):
            writer.add_image(image_labels[i], images[i], epoch_idx)
    print("Epoch", epoch_idx, "added to board log")
