params_list = [[0.001, 0.9] for _ in range(2)] +\
              [[0.0004, 0.9] for _ in range(2)] +\
              [[0.0001, 0.9] for _ in range(2)] +\
              [[0.00001, 0.9] for _ in range(2)]


def count_epoch() -> int:
    return len(params_list)
