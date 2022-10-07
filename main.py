import argparse
import pickle
import numpy as np
from tqdm import tqdm
from dataloader import load_data
from Model import *
from Evaluator import Evaluator
import pandas as pd
import os


def train_test(flags, n_users, n_items, x_train, x_val, x_test, y_train, y_val, y_test, user_group, item_group,
               avg_rating):
    loss_recorder = 9999.0
    model = FAiR(flags, n_users, n_items, user_group, item_group, avg_rating)
    train_user = np.squeeze(x_train[:, 0])
    train_item = np.squeeze(x_train[:, 1])
    train_y = np.squeeze(y_train)
    train_data = tf.data.Dataset.from_tensor_slices((train_user, train_item, train_y)).shuffle(100). \
        batch(flags.batch_size)
    for epoch in range(flags.epoch):
        with tqdm(total=len(train_data)) as t:
            t.set_description('Training Epoch %i' % epoch)
            for user_t, item_t, y_t in train_data:
                if epoch < flag.n_pretrain:
                    model.pretrain((user_t, item_t, y_t))
                    model.pretrain_loss_tracker.reset_states()
                else:
                    model.adv_train((user_t, item_t, y_t))
                t.update()
        if epoch > flag.n_pretrain and epoch % 1 == 0:
            vali_data = tf.data.Dataset.from_tensor_slices(
                (np.squeeze(x_val[:, 0]), np.squeeze(x_val[:, 1]), np.squeeze(y_val)))
            y_pred = None
            for user_val, item_val, _ in vali_data.batch(5000):
                if y_pred is None:
                    y_pred = model((user_val, item_val), False)
                else:
                    y_pred = tf.concat((y_pred, model((user_val, item_val), False)), 0)
            evaluator = Evaluator(x_val, y_val, y_pred.numpy(), user_group, item_group)
            prec, recall, ugf_prec, ugf_recall, rsp, reo = evaluator.evaluate(flags.n_rec)
            if loss_recorder > rsp + reo + ugf_prec + ugf_recall:
                loss_recorder = rsp + reo + ugf_prec + ugf_recall
                model.save("saved_model")
    optim_model = tf.keras.models.load_model("saved_model", compile=False)
    test_data = tf.data.Dataset.from_tensor_slices(
        (np.squeeze(x_test[:, 0]), np.squeeze(x_test[:, 1]), np.squeeze(y_test)))
    y_test_pred = None
    for user_test, item_test, _ in test_data.batch(5000):
        user_test = tf.cast(user_test, tf.int32)
        item_test = tf.cast(item_test, tf.int32)
        if y_test_pred is None:
            y_test_pred = optim_model((user_test, item_test), False)
        else:
            y_test_pred = tf.concat((y_test_pred, optim_model((user_test, item_test), False)), 0)
    evaluator = Evaluator(x_test, y_test, y_test_pred.numpy(), user_group, item_group)
    prec, recall, ugf_prec, ugf_recall, rsp, reo = evaluator.evaluate(flags.n_rec)
    return prec, recall, ugf_prec, ugf_recall, rsp, reo


def main(flags):
    n_users, n_items, x_train, x_val, x_test, y_train, y_val, y_test, user_group, item_group, avg_rating = load_data(
        flags.dataset, flags.n_test_negative_samples, flags.n_train_negative_samples)
    prec, recall, ugf_prec, ugf_recall, rsp, reo = train_test(flags, n_users, n_items, x_train, x_val, x_test,
               y_train, y_val, y_test, user_group, item_group, avg_rating)
    print(' / '.join(str(x) for x in [prec, recall, ugf_prec, ugf_recall, rsp, reo]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='movielens', type=str, help="The dataset used.")
    parser.add_argument("--dimension", default=128, type=int, help="number of features per user/item.")
    parser.add_argument("--d_step", default=1, type=int, help="number of discriminator training steps")
    parser.add_argument("--g_step", default=1, type=int, help="number of generator training steps")
    parser.add_argument("--batch_size", default=1600, type=int, help="batch size")
    parser.add_argument("--n_user_samples", default=128, type=int,
                        help="batch of users for training implicit discriminator")
    parser.add_argument("--n_rec", default=10, type=int,
                        help="number of recommendation")
    parser.add_argument("--n_item_samples", default=200, type=int,
                        help="batch of items for training implicit discriminator")
    parser.add_argument("--lambda_1", default=1.0, type=float, help="weight parameter for generator loss")
    parser.add_argument("--lambda_2", default=1.0, type=float, help="weight parameter for generator loss")
    parser.add_argument("--lambda_3", default=1.0, type=float, help="weight parameter for generator loss")
    parser.add_argument("--reg_term", default=0.00001, type=float, help="A term for parameter regularization.")
    parser.add_argument("--rec_layer_units",
                        default=[128, 64, 32, 16, 8],
                        type=list,
                        help="number of nodes each layer for MLP layers in Base Recommendation model")
    parser.add_argument("--filter_units",
                        default=[128, 128, 128, 128, 128],
                        type=list,
                        help="number of nodes each layer for MLP layers in Filters")
    parser.add_argument("--explicit_layer_units",
                        default=[128, 64, 32, 16, 8],
                        type=list,
                        help="number of nodes each layer for MLP layers in Explicit Discriminators")
    parser.add_argument("--implicit_layer_units",
                        default=[128, 64, 32, 16, 8],
                        type=list,
                        help="number of nodes each layer for MLP layers in Implicit Discriminators")
    parser.add_argument("--epoch", default=200, type=int,
                        help="Number of epochs in the training")
    parser.add_argument("--n_test_negative_samples", default=10, type=int,
                        help="ratio of negative samples each user")
    parser.add_argument("--n_train_negative_samples", default=0, type=int,
                        help="ratio of negative samples each user")
    parser.add_argument("--n_pretrain", default=10, type=int,
                        help="number of pretrain epochs.")
    flag = parser.parse_args()
    main(flag)
