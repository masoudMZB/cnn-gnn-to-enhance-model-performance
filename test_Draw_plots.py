import numpy as np
from Draw_plots import select_best_run_final_epoch, select_best_run_overall, mean_performance

def test_select_best_run_final_epoch():
    logs = [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [1, 2, 1]]
    ]
    # is_loss=True -> min value in last epoch
    res = select_best_run_final_epoch(logs, is_loss=True)
    assert res == [[1, 2, 3], [1, 2, 1]]

def test_mean_performance():
    logs = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]
    res = mean_performance(logs)
    assert np.allclose(res[0], [2, 3])
    assert np.allclose(res[1], [6, 7])

if __name__ == "__main__":
    test_select_best_run_final_epoch()
    test_mean_performance()
    print("test_Draw_plots passed")
