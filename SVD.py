__author__ = "Li Tao, ltipchrome@gmail.com"
import numpy as np


class SVDRec:
    """
    recommendation system using SVD
    reference: http://nicolas-hug.com/blog/matrix_facto_4
    """
    def __init__(self, n_components=2):
        self.n_components=n_components

    def fit(self, user_item, learning_rate=.001, max_epochs=1000, rmse=1.):
        self.user_item_ = user_item
        N_user, N_item = user_item.shape
        self.user_embedding = np.random.randn(N_user, self.n_components)
        self.item_embedding = np.random.randn(self.n_components, N_item)
        user_idx, item_idx = np.where(~np.isnan(user_item))
        n_epoch = 0
        while True:
            for uid, iid in zip(user_idx, item_idx):
                user_embedding = self.user_embedding[uid, :]
                item_embedding = self.item_embedding[:, iid]
                err = user_item[uid, iid] - user_embedding.dot(item_embedding)
                grad_u, grad_v = -err*item_embedding, -err*user_embedding
                self.user_embedding[uid, :] -= learning_rate*grad_u
                self.item_embedding[:, iid] -= learning_rate*grad_v
            RMSE = np.sqrt(np.nansum((self.user_embedding.dot(self.item_embedding) - user_item)**2)/len(user_idx))
            print(f'RMSE:{RMSE}')
            n_epoch += 1
            if (RMSE <= rmse) or (n_epoch > max_epochs):
                break
        return self

    def predict(self, user_id, item2pred=None, n_pred=5):
        if item2pred is None:
            item2pred = mask2index(np.isnan(self.user_item_[user_id, :]))[0]
        uvec = self.user_embedding[user_id, :]
        scores = uvec.dot(self.item_embedding)
        pred = item2pred[np.argsort(scores[item2pred])[-1:-1-n_pred:-1]]
        return pred


def mask2index(mask):
    return np.where(mask)


def test_SVDRec():
    np.random.seed(1)
    u = np.abs(np.random.randn(100, 10))
    v = np.abs(np.random.randn(10, 100))
    n_miss = 5000
    lr = .05
    n_pred = 5
    user_item_true = u.dot(v)
    user_item = user_item_true.copy()
    miss_idx = np.random.choice(np.prod(user_item.shape), n_miss, replace=False)
    miss_idx = np.unravel_index(miss_idx, user_item.shape)
    user_item[miss_idx] = np.nan
    svd = SVDRec(n_components=10)
    pred = svd.fit(user_item, learning_rate=lr, rmse=.1).predict(0, n_pred=5)
    to_pred = np.where(np.isnan(user_item[0, :]))[0]
    true2pred = to_pred[np.argsort(user_item_true[0, to_pred])[-1:-1-n_pred:-1]]
    print(f'truth is：{true2pred}')
    print(f' svd pred is: {pred}')
    assert np.any(np.isin(pred, true2pred))


if __name__ == "__main__":
    test_SVDRec()
