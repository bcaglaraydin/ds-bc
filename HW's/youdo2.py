#Berdan Çağlar Aydın
import numpy as np
import pandas as pd
from tqdm import trange


# select data to work on
def read_data(select):
    if select == "small":
        return np.array([[7, 6, 7, 4, 5, 4],
                         [6, 7, np.nan, 4, 3, 4],
                         [np.nan, 3, 3, 1, 1, np.nan],
                         [1, 2, 3, 3, 3, 4],
                         [1, np.nan, 1, 2, 3, 3]])
    elif select == "large":
        df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                         engine='python', names=['user_id', 'item_id', 'rating', 'timestamp'])
        return df.pivot(index='user_id', columns='item_id', values='rating').values
    else:
        print("selection not recognized!")


# train test split
def create_test_data(r):
    irow, jcol = np.where(~np.isnan(r))
    idx = np.random.choice(np.arange(3), 3, replace=False)
    test_irow = irow[idx]
    test_jcol = jcol[idx]

    r_copy = r.copy()
    for i in test_irow:
        for j in test_jcol:
            r_copy[i][j] = np.nan
    return r_copy, test_irow, test_jcol


# calculate loss
def loss(r, r_pred, r_not_nan):
    loss = []
    for i, j in r_not_nan:
        loss.append(np.power((r[i, j] - r_pred[i, j]), 2))
        s = sum(loss)
    return s / 2


def model(r, alpha, max_iter):
    # get all not-nan index on r
    r_not_nan = np.argwhere(~np.isnan(r))

    m, n = r.shape
    r_pred = np.empty(r.shape)
    r_pred[:] = np.nan

    # user and item bias terms
    bu = np.random.rand(m)
    bi = np.random.rand(n)
    for iteration in trange(max_iter):

        # un-comment to track loss (high complexity)
        # for u, j in r_not_nan:
        #     r_pred[u, j] = bu[u] + bi[j]
        # if iteration%(max_iter/10) == 0:
        #     print(loss(r, r_pred, r_not_nan))

        # gradient for each user and item
        g_bu = np.zeros(m)
        g_bi = np.zeros(n)

        for u, j in r_not_nan:
            gu = bi[j] + bu[u] - r[u, j]  # derivative of loss function for bu
            gi = bi[j] + bu[u] - r[u, j]  # derivative of loss function for bi

            # update for all users and items
            g_bu[u] += gu
            g_bi[j] += gi

        bu_prev = np.copy(bu)
        bi_prev = np.copy(bi)

        # update bias terms
        bu = bu - g_bu * alpha
        bi = bi - g_bi * alpha

        # stop condition
        if np.linalg.norm(bu - bu_prev) < 0.0001 and np.linalg.norm(bi - bi_prev) < 0.0001:
            print(iteration, "iterations")
            return bu, bi

    return bu, bi


# testing on previously nan injected ratings
def test(bu, bi, r_true, test_irow, test_jcol):
    err = []
    for i in test_irow:
        for j in test_jcol:
            err.append(((bu[i] + bi[j]) - r_true[i, j]) ** 2)

    return np.sqrt(np.nanmean(np.array(err)))


def model_regularized(r, alpha, max_iter, lam):
    # get all not-nan index on r
    r_not_nan = np.argwhere(~np.isnan(r))

    m, n = r.shape
    r_pred = np.empty(r.shape)
    r_pred[:] = np.nan

    # user and item bias terms
    bu = np.random.rand(m)
    bi = np.random.rand(n)

    for iteration in trange(max_iter):

        # gradient for each user and item
        g_bu = np.zeros(m)
        g_bi = np.zeros(n)

        for u, j in r_not_nan:
            gu = (bu[u] + bi[j] - r[u, j]) + lam * bu[u]  # regularization term included
            gi = (bu[u] + bi[j] - r[u, j]) + lam * bi[j]  # regularization term included
            # update for all users and items
            g_bu[u] += gu
            g_bi[j] += gi

        bu_prev = np.copy(bu)
        bi_prev = np.copy(bi)

        # update bias terms
        bu = bu - g_bu * alpha
        bi = bi - g_bi * alpha

        #stop condition
        if np.linalg.norm(bu - bu_prev) < 0.0001 and np.linalg.norm(bi - bi_prev) < np.linalg.norm(
                bu - bu_prev) < 0.0001:
            print(iteration, "iterations")
            return bu, bi
    return bu, bi


r = read_data("large")
r_copy, test_irow, test_jcol = create_test_data(r)

bu, bi = model2(r_copy, alpha=0.001, max_iter=1000)
test2(bu, bi, r, test_irow, test_jcol)

# hyperparameter (lambda) optimization
lambdas = [0, 0.1, 0.5, 1, 1.5, 2]
for lam in lambdas:
    bu, bi = model_regularized(r_copy, alpha=0.001, max_iter=1000, lam=lam)
    print("lambda: ", lam)
    print("bu, bi: ", bu, bi)
    print(test(bu, bi, r, test_irow, test_jcol))
