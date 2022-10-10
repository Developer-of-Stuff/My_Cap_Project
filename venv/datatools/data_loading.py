import numpy as np
import torch.nn as nn
from torch import tensor, LongTensor
from torch.utils.data.dataset import Dataset


class Loader(Dataset):
    def __init__(self, ratings):
        self.ratings = ratings.copy()

        users = self.ratings.user_id.unique()
        items = self.ratings.appid.unique()

        self.user_id_to_index = {o: i for i, o in enumerate(users)}
        self.appid_to_index = {o: i for i, o in enumerate(items)}

        self.index_to_user_id = {i: o for o, i in self.user_id_to_index.items()}
        self.index_to_appid = {i: o for o, i in self.appid_to_index.items()}

        self.ratings.appid = self.ratings.appid.apply(lambda x: self.appid_to_index[x])
        self.ratings.user_id = self.ratings.user_id.apply(lambda x: self.user_id_to_index[x])

        self.x = self.ratings.drop(['game_title', 'hours_played'], axis=1).values
        self.y = self.ratings['hours_played'].values

        self.x = tensor(self.x)
        self.y = tensor(self.y)

    def get_user_data(self, game_index):
        values = []
        users = []
        output = []
        for item in self.x:
            if item.numpy()[1] == game_index:
                values.append(item.numpy())
        for user in values:
            users.append(user[0])
        for item in self.x:
            game_user_list = []
            for user in users:
                if item.numpy()[1] != game_index:
                    game_user_list.append([user, item.numpy()[1]])
            if game_user_list:
                output.append(LongTensor(np.array(game_user_list)))
        return output

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.ratings)


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=True)
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        users, items = data[:, 0], data[:, 1]
        return (self.user_factors(users) * self.item_factors(items)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)
