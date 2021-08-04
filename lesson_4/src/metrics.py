"""
Metrics

"""
import numpy as np

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list_k = recommended_list[:k]

    prices_recommended = np.array(prices_recommended)
    prices_recommended_k = prices_recommended[:k]

    flags = np.isin(recommended_list_k, bought_list)

    prices_bought_k = flags * prices_recommended_k

    return prices_bought_k.sum() / prices_recommended_k.sum()


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    return flags.sum() / len(bought_list)


def recall_at_k(recommended_list, bought_list, k=5):
    return recall(recommended_list[:k], bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list_k = recommended_list[:k]

    prices_recommended = np.array(prices_recommended)
    prices_recommended_k = prices_recommended[:k]

    flags = np.isin(recommended_list_k, bought_list)

    prices_bought_k = flags * prices_recommended_k

    prices_bought = np.array(prices_bought)

    return prices_bought_k.sum() / prices_bought.sum()


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)

    if sum(flags) == 0:
        return 0

    sum_ = 0
    for i in range(1, k + 1):

        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k

    return sum_ / k


def map_k(recommended_list, bought_list, k=3):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    apk = []
    for i in range(len(recommended_list)):
        flags = np.isin(recommended_list[i], bought_list[i])

        if sum(flags) == 0:
            return 0

        sum_ = 0
        for m in range(1, k + 1):

            if flags[m] == True:
                p_k = precision_at_k(recommended_list[i], bought_list[i], k=m)
                sum_ += p_k

        result = sum_ / k
        apk.append(result)
    return sum(apk) / len(recommended_list)


def reciprocal_rank(recommended_list, bought_list):
    recommended_list = np.array(recommended_list)
    bought_list = np.array(bought_list)
    flags = np.isin(bought_list, recommended_list)

    if sum(flags) == 0:
        return 0
    k_index = np.where(flags != 0)[0][0]
    k_index += 1
    return 1 / k_index
