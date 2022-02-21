import numpy as np
import numpy.random as nr


def apply_k_all(x, idx, kernel=np.inner):
    res = np.apply_along_axis(lambda xi: kernel(xi, x[idx]), 1, x)
    return res


def smo_train(examples, labels, c, tol, max_iter, kernel=np.inner):
    b = 0
    m = len(labels)
    alpha = np.zeros(m)
    passes = 0
    while passes < max_iter:
        n_changed_alpha = 0
        for i in range(m):
            fx = apply_k_all(examples, i, kernel)

            fx = np.sum(alpha * labels * fx)
            fx += b
            err_i = fx - labels[i]
            if (labels[i] * err_i < -tol and alpha[i] < c) or (
                labels[i] * err_i > tol and alpha[i] > 0
            ):
                j = i
                while j == i:
                    j = nr.randint(0, m)
                fx_j = apply_k_all(examples, j, kernel)
                fx_j = np.sum(alpha * labels * fx_j) + b
                err_j = fx_j - labels[j]
                ai_old = alpha[i]
                aj_old = alpha[j]
                if labels[i] == labels[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(c, c + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - c)
                    H = min(c, alpha[i] + alpha[j])
                if L == H:
                    print("l=h")
                    continue
                nu = 2 * kernel(examples[i], examples[j]) - kernel(
                    examples[i], examples[i]
                ) - kernel(examples[j], examples[j])
                if nu >= 0:
                    print("nu positive")
                    continue
                alpha[j] = alpha[j] - (labels[j]*(err_i - err_j)) / nu
                if np.abs(alpha[j] - aj_old) < 0.00001:
                    print("alpha unchanged")
                    continue

                alpha[i] += labels[i] * labels[j] * (aj_old - alpha[j])
                b_part = labels[i] * (alpha[i] - ai_old) * kernel(
                    examples[i], examples[i]
                ) - labels[j] * (alpha[j] - aj_old) * kernel(examples[j], examples[j])
                b1 = b - err_i - b_part
                b2 = b - err_j - b_part
                if alpha[i] > 0 and alpha[i] < c:
                    b = b1
                elif alpha[j] > 0 and alpha[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                n_changed_alpha += 1
        if n_changed_alpha == 0:
            passes = passes + 1
        else:
            passes = 0
    return alpha, b

if __name__ == "__main__":
    nr.seed(1234)
    examples = nr.randint(0,12,(6,2))
    print(examples)
    labels = np.asarray([1,-1,-1,1,1,-1])
    a, b = smo_train(examples,labels,c=2,tol=0.01,max_iter=10)
    print(a)
    print(b)