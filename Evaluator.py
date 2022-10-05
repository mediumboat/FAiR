import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, x_test, y_test, prediction, user_group, item_group):
        self.user = np.squeeze(x_test[:, 0])
        self.item = np.squeeze(x_test[:, 1])
        self.y_true = np.squeeze(y_test)
        self.y_pred = np.squeeze(prediction)
        self.user_group = user_group
        self.item_group = item_group
        d = {"user": self.user, "item": self.item, "y_true": self.y_true, "y_pred": self.y_pred}
        self.df = pd.DataFrame(data=d)
        self.df['y_true'].values[self.df['y_true'].values > 0.0] = 1.0

    def rank(self, k):
        df = self.df.copy(deep=True)
        df = df.sort_values(by=["user", "y_pred"], ascending=False)
        df_ranking = df.groupby("user").head(k)
        return df_ranking

    def prec(self, df_ranking):
        return np.nanmean(df_ranking.loc[:, "y_true"].values)

    def recall(self, df_ranking):
        return np.nansum(df_ranking.loc[:, "y_true"].values) / np.nansum(self.df.loc[:, "y_true"].values)

    def ugf(self, df_ranking):
        df = self.df.copy(deep=True)
        prec_list = []
        rec_list = []
        for g in np.unique(self.user_group):
            user_group = np.where(self.user_group == g)[0]
            df_ranking_group = df_ranking[df_ranking["user"].isin(user_group)]
            prec_group = np.nanmean(df_ranking_group.loc[:, "y_true"].values)
            recall_group = np.nansum(df_ranking_group.loc[:, "y_true"].values) / np.nansum(
            df.loc[:, "y_true"].values)
            prec_list.append(prec_group)
            rec_list.append(recall_group)
        if len(prec_list) <= 2:
            ugf_prec = np.abs(prec_list[0] - prec_list[1])
            ugf_recall = np.abs(rec_list[0] - rec_list[1])
        else:
            ugf_prec_list = []
            ugf_rec_list = []
            for i in range(0, len(prec_list) - 1):
                for j in range(i + 1, len(prec_list)):
                    ugf_prec_list.append(np.abs(prec_list[i] - prec_list[j]))
                    ugf_rec_list.append(np.abs(rec_list[i] - rec_list[j]))
            ugf_prec = np.mean(ugf_prec_list)
            ugf_recall = np.mean(ugf_rec_list)
        return ugf_prec, ugf_recall

    def reo_rsp(self, df_ranking):
        df = self.df.copy(deep=True)
        p_rsp = []
        p_reo = []
        for g in np.unique(self.item_group):
            item_group = np.where(self.item_group == g)[0]
            df_ranking_group = df_ranking[df_ranking["item"].isin(item_group)]
            df_group = df[df["item"].isin(item_group)]
            p_rsp.append(float(len(df_ranking_group) / len(df_group)))
            df_positive_ranking_group = df_ranking[df_ranking["item"].isin(item_group) & df_ranking["y_true"] == 1.0]
            df_positive_group = df[df["item"].isin(item_group) & df["y_true"] == 1.0]
            p_reo.append(float(len(df_positive_ranking_group) / len(df_positive_group)))
        rsp = float(np.std(p_rsp) / np.mean(p_rsp))
        reo = float(np.std(p_reo) / np.mean(p_reo))
        return rsp, reo

    def evaluate(self, k=10):
        df_ranking = self.rank(k=k)
        prec = self.prec(df_ranking)
        recall = self.recall(df_ranking)
        ugf_prec, ugf_recall = self.ugf(df_ranking)
        rsp, reo = self.reo_rsp(df_ranking)
        return prec, recall, ugf_prec, ugf_recall, rsp, reo





