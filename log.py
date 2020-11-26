from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import pickle

log = [[], [], []]
labels = ["Train", "Test", "Loss"]


def load() -> None:
    global log
    log = pickle.load(open('log/log.pkl', 'rb'))


def save() -> None:
    pickle.dump(log, open('log/log.pkl', 'wb'))


def clear() -> None:
    global log
    log.clear()


def plot() -> None:
    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()

    host.set_ylim(0, 100)

    host.set_xlabel("Epoch")
    host.set_ylabel("Accuracy")
    par1.set_ylabel("Loss")

    for i in range(2):
        p1, = host.plot([j for j in range(len(log[i]))], log[i], label=labels[i])
    p2, = par1.plot([j for j in range(len(log[2]))], log[2], label=labels[2])

    par1.set_ylim(0, max(log[2]) * 1.5)

    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.draw()
    plt.show()


def add(values: tuple) -> None:
    global log
    for i in range(3):
        log[i].append(values[i])
