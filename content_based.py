import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split


class ContentRS:
    """
    content based recommendation system:
    推荐【类似于用户喜欢过的物品的物品】
    """
    def __init__(self):
        pass

    def fit(self, raw):
        """

        :param X: features of items,
        :return:
        """
        self.X_ = self.feature_extract(raw)
        # linear kernel: linear_kernel(X, Y) = X.dot(Y.T)
        # 计算X和Y中那个样本的内积
        self.fitted_ = True
        return self

    def feature_extract(self, raw):
        """
        :param raw: raw content of items
        :return:
        """
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=0, stop_words='english')
        X_ = self.vectorizer.fit_transform(raw)
        return X_

    def predict(self, X, n_items=5):
        if not self.fitted_:
            raise ValueError('not fitted')
        _X = self.vectorizer.transform(X)
        similarities = linear_kernel(_X, self.X_)
        item_idx = np.argsort(similarities, axis=1)[:, -2:-2-n_items:-1]
        return item_idx

    # alias
    recommend = predict


def test_content():
    # test data: eCommerce Item Data
    # https://www.kaggle.com/cclark/simple-content-based-recommendation-engine/data

    df = pd.read_csv('sample-data.csv', index_col=0)
    X_train, X_test = train_test_split(df, test_size=10, random_state=1)
    ct = ContentRS().fit(X_train['description'])
    preds = ct.predict(X_test['description'], n_items=3)
    recommended = ct.recommend(X_test['description'], n_items=3)
    assert recommended.shape == (10, 3)
    assert np.all(preds == recommended)


if __name__ == "__main__":
    test_content()
    # 喜欢t-shirt, 推荐的也是t-shirt
    # Inter-continental品牌的pants：
    #    推荐的也是Inter-continental capris，Inter-continental shorts，Solimar pants

