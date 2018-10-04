__author__ = "litao, ltipchrome@gmail.com"
import numpy as np


class ColFilter:
    """

    """
    def __init__(self, similarity='Jaccard'):
        self.similarity = similarity

    def fit(self, user_item, rating_threshold=3):
        """

        :param user_item:
        :return:
        """

        self.user_item_ = user_item
        n_user, n_item = user_item.shape
        user_user = np.zeros((n_user, n_user))
        for iu in range(n_user):
            for iv in range(n_user):
                if user_user[iu, iv] > 0:
                    continue
                else:
                    Nu = np.sum(user_item[iu, :] >= rating_threshold)
                    Nv = np.sum(user_item[iv, :] >= rating_threshold)
                    if self.similarity == 'Jaccard':
                        user_user[iu, iv] = user_user[iv, iu] = Jaccard(Nu, Nv)
                    elif self.similarity == 'cos':
                        user_user[iu, iv] = user_user[iv, iu] = cos(Nu, Nv)
                    else:
                        raise ValueError('similarity function not supported')
        self.user_user_ = user_user
        return self

    def predict(self, user_id, n_pred=5):

        to_predict = np.isnan(self.user_item_[user_id, :])
        if not np.any(to_predict):
            return "nothing to recommend"

        n_user, n_item = self.user_item_.shape
        mask = np.ones(n_user, dtype=bool)
        mask[user_id] = False
        u_item = self.user_item_[mask, :]
        maks_nan = np.isnan(u_item)
        ws = self.user_user_[mask, user_id]
        _rating = np.nansum(u_item * ws[:, np.newaxis], axis=0)
        _ws = np.sum(~maks_nan * ws[:, np.newaxis], axis=0)
        _rating /= _ws
        self.rating_ = _rating.copy()
        _rating[~to_predict] = -1
        pred = np.argsort(_rating)[-1:-1-n_pred:-1]
        return pred


def Jaccard(Nu, Nv):
    w = np.sum(Nu & Nv)/np.sum(Nu | Nv)
    return w


def cos(Nu, Nv):
    w = np.sum(Nu & Nv)/np.sqrt(Nu * Nv)
    return w


def test_ColFilter():

    # data: user-item interaction matrix
    seed = 1
    N_item = 100
    N_user = 50
    N_miss = 500
    user_id = 0
    n_pred = 5
    np.random.seed(seed)
    user_item = np.ones(N_user)[:, np.newaxis] * np.random.choice(5, N_item).astype(np.float)[np.newaxis, :]
    r_ture = user_item[user_id, :]
    miss_idx = np.random.choice(N_user * N_item, N_miss)
    user_item[np.unravel_index(miss_idx, user_item.shape)] = np.nan
    user_item -= np.random.choice(1, user_item.shape)
    user_item = np.abs(user_item)

    cf = ColFilter()
    cf.fit(user_item)

    pred_items = cf.predict(user_id, n_pred=n_pred)
    r_ture[~np.isnan(user_item[0, :])] = -1
    to_pred = np.argsort(r_ture)[-1:-1-n_pred:-1]
    print(f'pred_items: {pred_items}')
    print(f'to_pred: {to_pred}')
    assert np.any(np.isin(pred_items, to_pred))


if __name__ == "__main__":
    test_ColFilter()



