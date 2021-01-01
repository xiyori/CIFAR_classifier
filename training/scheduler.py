params_list = [[0.001, 0.9] for _ in range(2)] +\
              [[0.0003, 0.9] for _ in range(2)] +\
              [[0.00008, 0.9] for _ in range(2)] +\
              [[0.000006, 0.9] for _ in range(2)]


def count_epoch() -> int:
    return len(params_list)
