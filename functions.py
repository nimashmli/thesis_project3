import torch 

def k_fold_data_segmentation(x, y, k):
    n = x.shape[0]
    idx = list(range(n))

    for i in range(k):
        start = int(i * n / k)
        end = int((i + 1) * n / k)

        idx_test = idx[start:end]
        idx_test_set = set(idx_test)
        idx_train = [j for j in idx if j not in idx_test_set]

        x_train = x[idx_train, :, :]
        y_train = y[idx_train]
        x_test = x[idx_test, :, :]
        y_test = y[idx_test]

        yield (x_train, x_test, y_train, y_test)
