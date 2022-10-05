import pandas as pd
import numpy as np
from pathlib import Path
from operator import itemgetter

from tqdm import tqdm


def load_data(name=None, test_negative_samples=None, train_negative_samples=None):
    raw_data = None
    user2user_encoded = None
    movie2movie_encoded = None
    if name == "movielens":
        data_path = Path("../data/movielens")
        raw_data = pd.read_csv(data_path / "u.data", sep="\t",
                               usecols=[0, 1, 2], names=["user", "item", "rating"],
                               dtype={"user": int, "item": int, "rating": float})
        user_ids = raw_data["user"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        movie_ids = raw_data["item"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        raw_data["user"] = raw_data["user"].map(user2user_encoded)
        raw_data["item"] = raw_data["item"].map(movie2movie_encoded)
    elif name == "amazon":
        data_path = Path("../data/amazon")
        raw_data = pd.read_csv(data_path / "ratings_Grocery_and_Gourmet_Food.csv", header=0,
                               usecols=[0, 1, 2], names=["user", "item", "rating"],
                               dtype={"user": str, "item": str, "rating": float})
        user_ids = raw_data["user"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        movie_ids = raw_data["item"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        raw_data["user"] = raw_data["user"].map(user2user_encoded)
        raw_data["item"] = raw_data["item"].map(movie2movie_encoded)
    elif name == "twitter":
        data_path = Path("../data/twitter/")
        raw_data = pd.read_csv(data_path / "rating.csv", header=0,
                               usecols=[0, 1, 2], names=["user", "item", "rating"],
                               dtype={"user": int, "item": int, "rating": float})
        user_id = set(raw_data["user"].unique().tolist() + raw_data["item"].unique().tolist())
        user2user_encoded = {x: i for i, x in enumerate(user_id)}
        movie2movie_encoded = user2user_encoded
        raw_data["user"] = raw_data["user"].map(user2user_encoded)
        raw_data["item"] = raw_data["item"].map(movie2movie_encoded)
    else:
        data_path = Path("../data/" + str(name))
        # TODO: Load your customized data into a DataFrame.
    n_users = len(user2user_encoded)
    n_items = len(movie2movie_encoded)
    user_ids = range(n_users)
    item_ids = range(n_items)
    df = raw_data.sample(frac=1)  # Shuffle data
    avg_rating_df = pd.DataFrame(df.groupby('item')['rating'].mean())
    avg_rating = avg_rating_df['rating'].values
    x = np.reshape(df[["user", "item"]].values, (-1, 2))
    y = np.reshape(df["rating"].values, (-1, 1)) / 5.0
    train_indices = int(0.8 * df.shape[0])
    val_indices = int(0.9 * df.shape[0])
    x_train, x_val, x_test, y_train, y_val, y_test = (
        x[: train_indices],
        x[train_indices: val_indices],
        x[val_indices:],
        y[: train_indices],
        y[train_indices: val_indices],
        y[val_indices:]
    )
    if name == "twitter":
        for u in tqdm(user_ids):
            interacted_user = x[np.where(x[:, 0] == u)[0], 1]
            non_interacted_user = np.setdiff1d(user_ids, interacted_user, True)
            non_interacted_user = non_interacted_user[non_interacted_user != u]
            if train_negative_samples > 0:
                train_negative_items = np.reshape(
                    np.random.choice(non_interacted_user, train_negative_samples, False),
                    (-1, 1))
                train_keys = np.hstack(
                    (np.reshape(np.repeat(np.array([u]), train_negative_samples), (-1, 1)), train_negative_items))
                train_y = np.zeros((test_negative_samples, 1))
                x_train = np.concatenate((x_train, train_keys), 0)
                y_train = np.concatenate((y_train, train_y), 0)
            val_negative_items = np.reshape(np.random.choice(non_interacted_user, test_negative_samples, False),
                                            (-1, 1))
            test_negative_items = np.reshape(np.random.choice(non_interacted_user, test_negative_samples, False),
                                             (-1, 1))
            val_keys = np.hstack(
                (np.reshape(np.repeat(np.array([u]), test_negative_samples), (-1, 1)), val_negative_items))
            test_keys = np.hstack(
                (np.reshape(np.repeat(np.array([u]), test_negative_samples), (-1, 1)), test_negative_items))
            val_y = np.zeros((test_negative_samples, 1))
            test_y = np.zeros((test_negative_samples, 1))
            x_val = np.concatenate((x_val, val_keys), 0)
            x_test = np.concatenate((x_test, test_keys), 0)
            y_val = np.concatenate((y_val, val_y), 0)
            y_test = np.concatenate((y_test, test_y), 0)
    else:
        for u in tqdm(user_ids):
            interacted_item = x[np.where(x[:, 0] == u)[0], 1]
            non_interacted_user = np.setdiff1d(item_ids, interacted_item, True)
            if train_negative_samples > 0:
                train_negative_items = np.reshape(
                    np.random.choice(non_interacted_user, train_negative_samples, False),
                    (-1, 1))
                train_keys = np.hstack(
                    (np.reshape(np.repeat(np.array([u]), train_negative_samples), (-1, 1)), train_negative_items))
                train_y = np.zeros((test_negative_samples, 1))
                x_train = np.concatenate((x_train, train_keys), 0)
                y_train = np.concatenate((y_train, train_y), 0)
            val_negative_items = np.reshape(np.random.choice(non_interacted_user, test_negative_samples, False),
                                            (-1, 1))
            test_negative_items = np.reshape(np.random.choice(non_interacted_user, test_negative_samples, False),
                                             (-1, 1))
            val_keys = np.hstack(
                (np.reshape(np.repeat(np.array([u]), test_negative_samples), (-1, 1)), val_negative_items))
            test_keys = np.hstack(
                (np.reshape(np.repeat(np.array([u]), test_negative_samples), (-1, 1)), test_negative_items))
            val_y = np.zeros((test_negative_samples, 1))
            test_y = np.zeros((test_negative_samples, 1))
            x_val = np.concatenate((x_val, val_keys), 0)
            x_test = np.concatenate((x_test, test_keys), 0)
            y_val = np.concatenate((y_val, val_y), 0)
            y_test = np.concatenate((y_test, test_y), 0)
    # Start negative sampling

    # Got user and item groups
    user_group = None
    item_group = None
    if name == "movielens":
        user_df = pd.read_csv(data_path / "u.user", sep='|',
                              usecols=[0, 2], names=["user", "gender"],
                              dtype={"user": int, "gender": str})
        user_df['gender'] = user_df['gender'].map({'M': 0, 'F': 1})
        user_df['user'] = user_df['user'].map(user2user_encoded)
        user2gender = dict(zip(user_df['user'], user_df['gender']))
        user_group = np.reshape(itemgetter(*list(user_ids))(user2gender), [-1, 1])
        item_df = pd.DataFrame(data=x[:, 1], columns=['id'])
        fre_item = item_df['id'].value_counts().index.values.tolist()
        active_idx = int(n_items * 0.05)
        active_items = fre_item[: active_idx]
        inactive_items = fre_item[active_idx:]
        num_active_items = len(active_items)
        num_inactive_items = len(inactive_items)
        item_labels = np.concatenate((np.ones(num_active_items), np.zeros(num_inactive_items)), axis=None)
        item_groups = dict(zip(fre_item, item_labels))
        item_groups = itemgetter(*list(item_ids))(item_groups)
        item_group = np.reshape(item_groups, [-1, 1])
    elif name == "twitter":
        user_df = pd.DataFrame(data=x.flatten(), columns=['id'])
        fre_user = user_df['id'].value_counts().index.values.tolist()
        active_idx = int(n_users * 0.05)
        active_users = fre_user[: active_idx]
        inactive_users = fre_user[active_idx:]
        num_active_users = len(active_users)
        num_inactive_users = len(inactive_users)
        user_labels = np.concatenate((np.ones(num_active_users), np.zeros(num_inactive_users)), axis=None)
        user_groups = dict(zip(fre_user, user_labels))
        user_groups = itemgetter(*list(user_ids))(user_groups)
        user_group = np.reshape(user_groups, [-1, 1])
        item_group = user_group
    else:
        user_df = pd.DataFrame(data=x[:, 0], columns=['id'])
        fre_user = user_df['id'].value_counts().index.values.tolist()
        active_idx = int(n_users * 0.05)
        active_users = fre_user[: active_idx]
        inactive_users = fre_user[active_idx:]
        num_active_users = len(active_users)
        num_inactive_users = len(inactive_users)
        user_labels = np.concatenate((np.ones(num_active_users), np.zeros(num_inactive_users)), axis=None)
        user_groups = dict(zip(fre_user, user_labels))
        user_groups = itemgetter(*list(user_ids))(user_groups)
        user_group = np.reshape(user_groups, [-1, 1])

        item_df = pd.DataFrame(data=x[:, 1], columns=['id'])
        fre_item = item_df['id'].value_counts().index.values.tolist()
        active_idx = int(n_items * 0.05)
        active_items = fre_item[: active_idx]
        inactive_items = fre_item[active_idx:]
        num_active_items = len(active_items)
        num_inactive_items = len(inactive_items)
        item_labels = np.concatenate((np.ones(num_active_items), np.zeros(num_inactive_items)), axis=None)
        item_groups = dict(zip(fre_item, item_labels))
        item_groups = itemgetter(*list(item_ids))(item_groups)
        item_group = np.reshape(item_groups, [-1, 1])
    return n_users, n_items, x_train, x_val, x_test, y_train, y_val, y_test, user_group, item_group, avg_rating

if __name__ == "__main__":
    pass