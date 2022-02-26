from random import gauss
from matplotlib.pyplot import axis
import numpy as np
import numpy.random as nr

import preprocess

ERROR_CACHE = None

def poly_kernel(x, z, c):
    return (np.dot(x, z) + c)**2

def gaussian_kernel(x, z, sigma):
    return np.exp(-(np.linalg.norm(x - z)**2) / (2*sigma**2))

def apply_k_all(x, inp, kernel=np.inner, kernel_arg=None):
    if kernel_arg:
        res = np.apply_along_axis(lambda xi: kernel(xi, inp, kernel_arg), 1, x)
    else:
        res = np.apply_along_axis(lambda xi: kernel(xi, inp), 1, x)
    return res

def svm_eval(inp,x,y,a,b,kernel=np.inner, kernel_arg=None):
    fx = apply_k_all(x,inp,kernel=kernel, kernel_arg=kernel_arg)
    fx = np.sum(a * y * fx) + b
    return np.sign(fx)


def smo_train(x, y, c, tol, max_iter, kernel=np.inner, kernel_arg=None):
    b = 0
    m = len(y)
    passes = 0
    examineAll = True
    n_changed_alpha = 0
    while (n_changed_alpha > 0 or examineAll) and passes < max_iter:
        n_changed_alpha = 0
        if examineAll:
            for i in range(m):
                    n_changed_alpha += examine_one(x,y,c,tol,i,kernel=kernel,kernel_arg=kernel_arg)
        else:
            for i in range(m):
                if alpha[i] != 0 and alpha[i] != c:
                    n_changed_alpha += examine_one(x,y,c,tol,i,kernel=kernel,kernel_arg=kernel_arg)
        if examineAll:
            examineAll = False
        elif n_changed_alpha == 0:
            examineAll = True
        passes = passes + 1
        print(f"Passes: {passes}")
    return alpha, b


def examine_one(x, y, c, tol, i, kernel=np.inner, kernel_arg=None):
    global b
    m = len(y)
    fx = apply_k_all(x, x[i], kernel, kernel_arg)

    fx = np.sum(alpha * y * fx)
    fx += b
    err_i = fx - y[i]
    if (y[i] * err_i < -tol and alpha[i] < c) or (
            y[i] * err_i > tol and alpha[i] > 0
    ):
        j = i
        while j == i:
            j = nr.randint(0, m)
        fx_j = apply_k_all(x, x[j], kernel, kernel_arg)
        fx_j = np.sum(alpha * y * fx_j) + b
        err_j = fx_j - y[j]
        ai_old = alpha[i]
        aj_old = alpha[j]
        if y[i] != y[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(c, c + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[i] + alpha[j] - c)
            H = min(c, alpha[i] + alpha[j])
        if L == H:
            return 0
        if kernel_arg:
            eta = 2 * kernel(x[i], x[j], kernel_arg) - kernel(x[i], x[i], kernel_arg) - kernel(x[j], x[j], kernel_arg)
        else:
            eta = 2 * kernel(x[i], x[j]) - kernel(x[i], x[i]) - kernel(x[j], x[j])
        if eta >= 0:
            return 0
        alpha[j] = alpha[j] - (y[j] * (err_i - err_j)) / eta
        if alpha[j] < L:
            alpha[j] = L
        elif alpha[j] > H:
            alpha[j] = H
        if np.abs(alpha[j] - aj_old) < 0.00001:  # In pseudocode it says 10^-5
            return 0
        s = y[i] * y[j]
        alpha[i] += s * (aj_old - alpha[j])
        if kernel_arg:
            b_part = y[i] * (alpha[i] - ai_old) * kernel(x[i], x[i], kernel_arg) - y[j] * (
                    alpha[j] - aj_old
            ) * kernel(x[j], x[j], kernel_arg)
        else:
            b_part = y[i] * (alpha[i] - ai_old) * kernel(x[i], x[i]) - y[j] * (
                    alpha[j] - aj_old
            ) * kernel(x[j], x[j])
        b1 = b - err_i - b_part
        b2 = b - err_j - b_part
        b_old = b
        if alpha[i] > 0 and alpha[i] < c:
            b = b1
        elif alpha[j] > 0 and alpha[j] < c:
            b = b2
        else:
            b = (b1 + b2) / 2
        return 1
    return 0

def n_folds_np(N, labels, examples):
    indexes = np.arange(labels.shape[0])
    nr.shuffle(indexes)
    fold_indexes = np.asarray(np.split(indexes, N))
    fold_list = []
    for i in range(N):
        test_idxs = fold_indexes[i]
        train_idxs = np.concatenate(
            fold_indexes[
                np.arange(N) != i,
            ],
            axis=0,
        )
        test_labels = labels[test_idxs]
        test_examples = examples[test_idxs]
        train_labels = labels[train_idxs]
        train_examples = examples[train_idxs]
        fold = {
            "train": (train_examples, train_labels),
            "test": (test_examples, test_labels),
        }
        fold_list.append(fold)
    return fold_list


if __name__ == "__main__":
    nr.seed(1234)
    labels, examples = preprocess.spirals(plot=True, svm=True)
    labels = labels[:200]
    examples = examples[:200]
    folds = n_folds_np(5, labels, examples)
    print("boop")
    curr_fold_x, curr_fold_y = folds[0]["train"]
    alpha = np.zeros_like(curr_fold_y)
    b = 0
    a, b = smo_train(curr_fold_x, curr_fold_y, c=100, tol=0.05, max_iter=10, kernel=gaussian_kernel, kernel_arg=1)
    correct = 0
    total = len(folds[0]["test"][1])
    for idx in range(len(folds[0]["test"][1])):
        test_ex = folds[0]["test"][0][idx]
        test_lbl = folds[0]["test"][1][idx]
        predicted = svm_eval(test_ex, folds[0]["train"][0], folds[0]["train"][1], a, b, kernel=gaussian_kernel, kernel_arg=1)
        if predicted == test_lbl:
            correct += 1
    print(f"Accuracy: {correct/total}")
