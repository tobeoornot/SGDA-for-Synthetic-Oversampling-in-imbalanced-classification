import collections
import math
import warnings

import numpy as np
from sklearn import metrics


# 输出模型的性能参数， recall,f1-score,prc...
def model_val(X_, y_, model):
    pred_prob = model.predict_proba(X_)
    pred = model.predict(X_)
    con_matrix = metrics.confusion_matrix(y_, pred)
    TN, FP, FN, TP = con_matrix.reshape((4,))
    f1_score = metrics.f1_score(y_, pred)
    print(f"con_matrix is:\r\n {con_matrix}\r\n"
          f"f1_score is:{f1_score}")

    FPR, TPR, threshold = metrics.roc_curve(y_, pred_prob[:, 1])
    recall = metrics.recall_score(y_, pred)
    print(f"recall is:{recall}")
    AUC = metrics.auc(FPR, TPR)
    print(f"AUC is:{AUC}")
    specificity = TN / (TN + FP)
    print(f"tpr is:{TP/(TP+FN)}, fpr is: {1- specificity}")
    G_mean = (recall * specificity) ** 0.5
    mcc = metrics.matthews_corrcoef(y_, pred)
    ks = max(abs(FPR - TPR))
    print(f"G-mean is:{G_mean}\r\n"
          f"mcc is:{mcc}\r\n"
          f"ks is:{ks}")

    return pred_prob, f1_score, model, pred


class MyArraysTransformer:
    """A class to convert sampler output arrays to their original types."""

    def __init__(self, X, y):
        self.x_props = self._gets_props(X)
        self.y_props = self._gets_props(y)

    def transform(self, X, y):
        X = self._trans_from_one(X, self.x_props)
        y = self._trans_from_one(y, self.y_props)
        if self.x_props["type"].lower() == "dataframe" and self.y_props[
            "type"
        ].lower() in {"series", "dataframe"}:
            # We lost the y.index during resampling. We can safely use X.index to align
            # them.
            y.index = X.index
        return X, y

    @staticmethod
    def _gets_props(array):
        props = {"type": array.__class__.__name__, "columns": getattr(array, "columns", None),
                 "name": getattr(array, "name", None), "dtypes": getattr(array, "dtypes", None)}
        return props

    @staticmethod
    def _trans_from_one(array, props):
        type_ = props["type"].lower()
        if type_ == "list":
            ret = array.tolist()
        elif type_ == "dataframe":
            import pandas as pd

            ret = pd.DataFrame(array, columns=props["columns"])
            ret = ret.astype(props["dtypes"])
        elif type_ == "series":
            import pandas as pd

            ret = pd.Series(array, dtype=props["dtypes"], name=props["name"])
        else:
            ret = array
        return ret


class GradientDistributionLikelihood:
    def __init__(self, bins, X, y, pred_model):
        self.bins = bins
        self.X = X
        self.y = y
        self.pred_model = pred_model
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-5
        self.to_generate_nums, self.maj_samples, self.min_samples = self.get_train_set_info()
        self.maj_bins_info = self.get_difficulty_bins(self.maj_samples, True)
        self.min_bins_info, self.min_samples_with_difficulty = self.get_difficulty_bins(self.min_samples, False)
        self.generate_info = self.get_difficulty_bins_info()
        # self.generate_info = self.get_to_generate_num_for_every_bins()

    # 以训练集正负样本的数量计算需要生成的样本数量、提取出多数类样本集合和少数类样本集合
    def get_train_set_info(self) -> tuple:
        min_samples_num = self.y.sum()
        maj_samples_num = self.X.shape[0] - min_samples_num
        to_generate_num = maj_samples_num - min_samples_num
        print(f"min_samples_num is:{min_samples_num}, maj_samples_num is:{maj_samples_num}")
        maj_samples_index = []
        min_samples_index = []
        for i in range(len(self.X)):
            if self.y.values[i] == 1:
                to_append_index = min_samples_index
            else:
                to_append_index = maj_samples_index
            to_append_index.append(self.X.iloc[i].name)
        maj_samples = self.X.loc[maj_samples_index]
        min_samples = self.X.loc[min_samples_index]
        return to_generate_num, maj_samples, min_samples

    # 计算少数类或者多数类样本的难度分布字典，并且为少数类样本拼接difficulty这一列
    def get_difficulty_bins(self, samples, is_maj_samples):
        # 如果是多数类
        if is_maj_samples:
            difficulty = self.pred_model.predict_proba(samples)[:, 1]
        else:
            difficulty = 1 - self.pred_model.predict_proba(samples)[:, 1]
            samples['difficulty'] = difficulty
        this_region_num = []
        for i in range(len(difficulty)):
            if difficulty[i] == 0:
                this_region_num.append(1)
            else:
                this_region_num.append(math.ceil(difficulty[i] / (1 / self.bins)))
        if is_maj_samples:
            return collections.Counter(this_region_num)
        else:
            return collections.Counter(this_region_num), samples

    # 获取多数类和少数类都有的且少数类数量比多数类数量少的区间的索引，并计算每个区间应该生成的样本数量
    def get_difficulty_bins_info(self) -> dict:
        return_dict = {}
        for i in self.min_bins_info.keys():
            if i in self.maj_bins_info.keys():
                if self.min_bins_info[i] < self.maj_bins_info[i]:
                    this_key_value = {i: self.maj_bins_info[i] - self.min_bins_info[i]}
                    return_dict.update(this_key_value)
        return_dict = dict(sorted(return_dict.items(), key=lambda x: x[1], reverse=True))
        return return_dict

    # 按照少数类样本的原始难度区间分布，计算每个区间应该生成的样本数量
    def get_to_generate_num_for_every_bins(self) -> dict:
        # 判断少数类区间字典和需要生成的数量值是否为空
        if self.min_bins_info is None or self.to_generate_nums <= 0:
            warnings.warn("Gradient Information for Minority Class or the Number of Samples to Generate is Ambiguous")
            return {}
        to_generate_bins_info = {}
        next_bins_info = self.min_bins_info.copy()
        # for key in next_bins_info.keys():
        #     if key > self.bins / 2:     #  在这里弹性
        #         self.min_bins_info.pop(key)
        bins_sample_num = sum(value for value in self.min_bins_info.values())
        for key in self.min_bins_info.keys():
            to_generate_bins_info.update({key: math.ceil(self.min_bins_info[key] / bins_sample_num * self.to_generate_nums)})
        return to_generate_bins_info

    # 按照生成字典，为每个区间生成样本,  返回生成后多数类和少数类数量一致的样本集
    def fit_resample(self, flag):
        X_resampled = [self.X.copy()]
        y_resampled = [self.y.copy()]
        new_generate_info = {}
        if flag:
            generate_info_sum = sum(value for value in self.generate_info.values())
            if generate_info_sum <= self.to_generate_nums:
                # 如果是字典里的总数小于等于需要生成的样本数量，直接遍历字典生成
                new_generate_info = self.generate_info
            else:
                # 如果字典里的总数大于需要生成的样本数量，按照从大到小的顺序加，直到等于需要生成样本的数量，改变字典
                temp_sum = 0
                for x in self.generate_info.keys():
                    temp_sum += self.generate_info[x]
                    already_add_sum = sum(value for value in new_generate_info.values())
                    print(f"this is already_add_sum is : {already_add_sum}")
                    if temp_sum <= self.to_generate_nums:
                        new_generate_info.update({x: self.generate_info[x]})
                    else:
                        if already_add_sum > 0:
                            new_generate_info.update({x: self.to_generate_nums - already_add_sum})
                        else:
                            new_generate_info.update({x: self.to_generate_nums})
                        break
        else:
            new_generate_info = self.get_to_generate_num_for_every_bins()
        print(f"new_generate_info:{new_generate_info}")
        max_info_key = max(new_generate_info.keys(), key=lambda k: new_generate_info[k])
        # 遍历生成信息每个字典元素，生成
        for i in new_generate_info.keys():
            # 判断是不是最后一个区间，也就是区间索引是不是最大值

            print(f"ith is:{i}, edge is:{self.edges[i - 1], self.edges[i]}")
            if i != max_info_key:
                this_bin_sample = self.min_samples_with_difficulty[(self.min_samples_with_difficulty['difficulty'] >=
                                                                    self.edges[i - 1]) &
                                                                   (self.min_samples_with_difficulty['difficulty'] <
                                                                    self.edges[i])]

                # 按照 difficulty升序排序
                this_bins_samples_sort_dataframe = this_bin_sample.sort_values(by="difficulty", ascending=True).drop(
                    columns=['difficulty'])
                # 计算区间的样本数量，便于生成随机数
                this_bins_samples_length = this_bin_sample.shape[0]
                # 将dataframe转换成array 方便计算
                this_bins_samples_sort_array = this_bins_samples_sort_dataframe.values
                # 产生这个区间需要生成的数量个 随机数，从0 到区间de样本数量范围内， 也就是有放回的抽取。
                this_root_samples_indices = np.random.RandomState(42).randint(low=0, high=this_bins_samples_length,
                                                                              size=new_generate_info[i])
                print(f"this_root_samples_indices.shape is :{this_root_samples_indices.shape}")
                # 生成(0,1)之间的随机数
                steps = np.random.uniform(size=new_generate_info[i])[:, np.newaxis]

                # 如果这个少数类区间内只有一个样本， 根样本索引和辅助样本索引都一样
                if this_bins_samples_length == 1:
                    this_assistant_indices = this_root_samples_indices
                else:
                    this_assistant_indices_list = []
                    for j in this_root_samples_indices:
                        if j < (this_bins_samples_length - 1):
                            this_assistant_indices_list.append(j + 1)
                        else:
                            this_assistant_indices_list.append(j - 1)
                    this_assistant_indices = np.array(this_assistant_indices_list)
                # print(f"this_root_indexs is{this_root_samples_indices}")
                # print(f"this_assistant indexs is:{this_assistant_indices}")
                #  这一行有好几次报错，索引0越界或者其他
                # print("xxxxxxxxxxxxxxxxxx")
                # print(this_assistant_indices, this_root_samples_indices)
                this_diffs = this_bins_samples_sort_array[this_assistant_indices] - this_bins_samples_sort_array[
                    this_root_samples_indices]

                X_resampled.append(this_bins_samples_sort_array[this_root_samples_indices] + steps * this_diffs)
                y_resampled.append(np.full(new_generate_info[i], fill_value=1))
            # 最大索引区间，需要取到下一个难度区间的样本
            else:
                this_bin_sample = self.min_samples_with_difficulty[(self.min_samples_with_difficulty['difficulty'] >=
                                                                    self.edges[i - 1]) &
                                                                   (self.min_samples_with_difficulty['difficulty'] <
                                                                    self.edges[i])].drop(columns=['difficulty'])
                next_bin_sample = self.min_samples_with_difficulty[(self.min_samples_with_difficulty['difficulty'] >=
                                                                    self.edges[i]) &
                                                                   (self.min_samples_with_difficulty['difficulty'] <
                                                                    self.edges[i + 1])]
                # 这是取下一个区间难度最低的样本，测试在下一个区间里随机取
                next_bin_sort_sample = next_bin_sample. sort_values(by="difficulty", ascending=True).drop(
                    columns=['difficulty'])
                # 计算区间的样本数量，便于生成随机数
                this_bins_samples_length = this_bin_sample.shape[0]
                # 将dataframe转换成array 方便计算
                this_bins_samples_array = this_bin_sample.values
                # 产生这个区间需要生成的数量个 随机数，从0 到区间de样本数量范围内， 也就是有放回的抽取。
                this_root_samples_indices = np.random.RandomState(42).randint(low=0, high=this_bins_samples_length,
                                                                              size=new_generate_info[i])
                # 生成(0,1)之间的随机数
                steps = np.random.uniform(size=new_generate_info[i])[:, np.newaxis]
                # 下一个区间的难度最低的样本作为辅助样本
                this_diffs = next_bin_sort_sample.values[0] - this_bins_samples_array[
                    this_root_samples_indices]

                X_resampled.append(this_bins_samples_array[this_root_samples_indices] + steps * this_diffs)
                y_resampled.append(np.full(new_generate_info[i], fill_value=1))
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        arrays_transformer = MyArraysTransformer(self.X, self.y)
        X_, y_ = arrays_transformer.transform(X_resampled, y_resampled)

        return X_, y_
