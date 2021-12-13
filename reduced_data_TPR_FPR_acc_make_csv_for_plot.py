# encoding:UTF-8
# データ量を減らして作った.pklを読み込んで性能評価 TPR,FPR,acc全部をcsvに書き込む

import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import sys
from torchvision import models
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import datetime
from sklearn.model_selection import train_test_split
import random
import os
import glob
import csv
import pickle

def sn_to_upperlimit_mag_diff(sn):
    upperlimit_mag_diff = [-2.5*np.log10(3/x) for x in sn]
    return upperlimit_mag_diff

# 1個のインスタンスに(tvt,train_proportion)1ペアが対応する
class CalcTprFprAcc():
    #ul_mg_threshold = [0.75257499,2.25772497] # SN 5~10,10~40,40~80
    #ul_mg_threshold = [1.30719686, 2.30204688] # SN 3~10,10~25,25~50
    ul_mg_threshold = sn_to_upperlimit_mag_diff([5,25]) # SN 3~5,5~25,25~50
    def __init__(self, tvt, train_proportion):
        self.tvt = tvt
        self.train_proportion = train_proportion
           
    def net_param_date_list_maker(self):
        netfile_path = f"{dire}/Net/reduce_train_10times/{network_name}/{self.train_proportion}/a_*"
        netfile_path_list = glob.glob(netfile_path)
        param_date_list = []
        for nf_pth in netfile_path_list:
            basename = os.path.basename(nf_pth)
            file_name = os.path.splitext(basename)[0]
            param_date = file_name.split("_")[-1]
            param_date_list.append(param_date)
        return param_date_list


    def param_date_to_y_preds0(self, param_date):
        y_and_y_preds0 = pickle.load(open(f"{dire}/csv/reduce_train_10times/{network_name}/y_y_preds0_{self.tvt}_5_{network_name}_{self.train_proportion}_{inputs_date}_{param_date}.csv.pkl", 'rb'))
        y = y_and_y_preds0[f"y_{self.tvt}"].values
        y_preds0 = y_and_y_preds0[f"y_preds0_{self.tvt}"].values
        return y, y_preds0


    def divide_y_preds0_to_some_classes(self, y, y_preds0):
        #ul_mg
        #ul_mg_threshold = [1,2]
        idx_ul_mg_01 = np.array((self.csv['pn'] == 'positive') & (self.csv['ul_mg'] <= self.ul_mg_threshold[0]))
        idx_ul_mg_12 = np.array((self.csv['pn'] == 'positive') & (self.csv['ul_mg'] > self.ul_mg_threshold[0]) & (self.csv['ul_mg'] <= self.ul_mg_threshold[1]))
        idx_ul_mg_23 = np.array((self.csv['pn'] == 'positive') & (self.csv['ul_mg'] > self.ul_mg_threshold[1]))

        #print(len(idx_ul_mg_01))
        #print((idx_ul_mg_01+idx_ul_mg_12+idx_ul_mg_23).sum())
        
        y_ul_mg_01, y_preds0_ul_mg_01 = y[idx_ul_mg_01], y_preds0[idx_ul_mg_01]
        y_ul_mg_12, y_preds0_ul_mg_12 = y[idx_ul_mg_12], y_preds0[idx_ul_mg_12]
        y_ul_mg_23, y_preds0_ul_mg_23 = y[idx_ul_mg_23], y_preds0[idx_ul_mg_23]

        range_ul_mg = np.array(["01", "12", "23"])
        y_ul_mg = np.array([y_ul_mg_01, y_ul_mg_12, y_ul_mg_23], dtype=object)
        y_preds0_ul_mg = np.array([y_preds0_ul_mg_01, y_preds0_ul_mg_12, y_preds0_ul_mg_23], dtype=object)
        
        return range_ul_mg, y_ul_mg, y_preds0_ul_mg

    def calc_tpr_fpr_acc_net(self, y, y_preds0):
        acc_net = ((y == 0)&(y_preds0 <= 0.5) | (y == 1)&(y_preds0 > 0.5)).sum() / len(y)
    
        fpr, tpr, thresholds = metrics.roc_curve(y, y_preds0)
        auc = metrics.auc(fpr, tpr)
        tpr_fpr = [t - f for f,t in zip(fpr, tpr)]
        tpr_tnr = [abs(t - (1-f)) for f,t in zip(fpr, tpr)]
        max_tpr_fpr = max(tpr_fpr)
        min_tpr_tnr = min(tpr_tnr)
        best_th1 = thresholds[tpr_fpr.index(max_tpr_fpr)]
        best_th2 = thresholds[tpr_tnr.index(min_tpr_tnr)]
        # 閾値を共有
        self.y_threshold = best_th2

        #print(f"range {i} by min_tpr_tnr criterion:")
        tpr_net = tpr[tpr_tnr.index(min_tpr_tnr)]
        fpr_net = fpr[tpr_tnr.index(min_tpr_tnr)]
        #print(f"TPR={tpr_res:.4f}")
        #print(f"FPR={fpr[tpr_tnr.index(min_tpr_tnr)]:.4f}")
        return tpr_net, fpr_net, acc_net
        
    def calc_metrics(self, y, y_preds0):
        
        tp = ((y == 1)&(y_preds0 >= self.y_threshold)).sum() / len(y)
        fn = ((y == 1)&(y_preds0 <= self.y_threshold)).sum() / len(y)
        fp = ((y == 0)&(y_preds0 >= self.y_threshold)).sum() / len(y)
        tn = ((y == 0)&(y_preds0 <= self.y_threshold)).sum() / len(y)
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        acc = (tp + tn) / (tp + fn + fp + tn)
        return tpr, fpr, acc

    def calc_tpr_each_param(self, y, y_preds0):
        # each: 明るさなどの階級別のTPRを計算する
        tpr_each = []
        
        range_class, y_class, y_preds0_class = self.divide_y_preds0_to_some_classes(y, y_preds0)
        
        for i, j, k in zip(range_class, y_class, y_preds0_class):
            tpr_each.append(self.calc_metrics(j,k)[0])
        return tpr_each
        
    def calc_tpr_fpr_acc_all(self):
        tpr_list, tpr_ave, tpr_std = [], [], []
        tpr_net_list, fpr_net_list, acc_net_list = [], [], [] # netはS/Nなどで分類しないトータルの値という意味
        
        param_date_list = self.net_param_date_list_maker()
        for param_date in param_date_list:
            if self.tvt == "test":
                self.csv = pd.read_csv(f"{dire}/csv/dataset_{inputs_date}.csv")
                self.csv = self.csv.query('tvt == "test"')
                self.csv = pd.concat([self.csv[::2], self.csv[1::2]])
            elif self.tvt == "train" or self.tvt == "valid":
                self.csv = pickle.load(open(f"{dire}/csv/reduce_train_10times/{network_name}/{self.tvt}_{fold}_{network_name}_{self.train_proportion}_{inputs_date}_{param_date}.csv.pkl", 'rb'))
            
            y, y_preds0 = self.param_date_to_y_preds0(param_date)
            for i, metrics in zip([0,1,2], [tpr_net_list,fpr_net_list,acc_net_list]):
                metrics.append(self.calc_tpr_fpr_acc_net(y, y_preds0)[i])
            tpr_list.append(self.calc_tpr_each_param(y, y_preds0))
            
        tpr_list = np.array(tpr_list)
        tpr_net_list = np.array(tpr_net_list)
        fpr_net_list = np.array(fpr_net_list)
        acc_net_list = np.array(acc_net_list)
        
        tpr_net_ave = np.mean(tpr_net_list)
        tpr_net_std = np.std(tpr_net_list)
        fpr_net_ave = np.mean(fpr_net_list)
        fpr_net_std = np.std(fpr_net_list)
        acc_net_ave = np.mean(acc_net_list)
        acc_net_std = np.std(acc_net_list)
        
        class_num = tpr_list.shape[1] # 分類の数
        for i in range(class_num):
            tpr_ave.append(np.mean(tpr_list.T[i]))
            tpr_std.append(np.std(tpr_list.T[i]))

        return tpr_ave, tpr_std, tpr_net_ave, tpr_net_std, fpr_net_ave, fpr_net_std, acc_net_ave, acc_net_std


    
def csv_writer(tvt, train_proportion, data):
    tpr_ave, tpr_std, tpr_net_ave, tpr_net_std, fpr_net_ave, fpr_net_std, acc_net_ave, acc_net_std = data
    class_sn = ["3_5","5_25","25_50"]
    header = ["train_proportion", "SN", "TPR_ave", "TPR_std", "tpr_net_ave", "tpr_net_std", "fpr_net_ave", "fpr_net_std", "acc_net_ave", "acc_net_std"]
    #body = np.empty((0,(len(header))))    
    body = []

    for i in range(len(class_sn)):
        body.append([train_proportion, class_sn[i], tpr_ave[i], tpr_std[i], "", "", "", "", "", ""])
    body.append([train_proportion, "all", "", "", tpr_net_ave, tpr_net_std, fpr_net_ave, fpr_net_std, acc_net_ave, acc_net_std])

    with open(csv_path[tvt], 'a') as f:
        writer = csv.writer(f)
        if train_proportion == train_proportion_list[0]:
       	    writer.writerow(header)
        writer.writerows(body)


if __name__ == "__main__":
    dire = "/gs/hs0/tga-transient-mitsume/nito/transient_detection"
    inputs_date = "200503235749"
    #network_name = "resnet50_pretrained_True"
    network_name = "vgg11_bn_pretrained_True"
    fold = 5


    #csv_name = "reduce_train_10times_test_SN10-40_2.csv"
    csv_path = {"train":f"{dire}/csv/reduce_train_10times/{network_name}/plot_train_{network_name}_SN5-25_{inputs_date}_samethreshold.csv", "test":f"{dire}/csv/reduce_train_10times/{network_name}/plot_test_{network_name}_SN5-25_{inputs_date}_samethreshold.csv"}
    for path in csv_path.values():
        if(os.path.isfile(path)):
            os.remove(path)


    #tvt_list = ["train", "test"]
    tvt_list = ["test"]
    #train_proportion_list = [0.125,0.25,0.5,0.6,0.7,0.8,0.9,0.99]
    train_proportion_list = [1.0]

    print(f"ul_mg threshold is{CalcTprFprAcc.ul_mg_threshold}")

    for tvt in tvt_list:
        for tp in train_proportion_list:
            Calc = CalcTprFprAcc(tvt, tp)
            tpr_ave, tpr_std, tpr_net_ave, tpr_net_std, fpr_net_ave, fpr_net_std, acc_net_ave, acc_net_std = Calc.calc_tpr_fpr_acc_all()
            csv_writer(tvt, tp, (tpr_ave, tpr_std, tpr_net_ave, tpr_net_std, fpr_net_ave, fpr_net_std, acc_net_ave, acc_net_std))


#r
"""
# csv_test["pn"] == "positive" を追加
idx_r_0_5 = np.array(csv_test['r'] <= 5)
idx_r_5_10 = np.array((csv_test['r'] > 5) & (csv_test['r'] <= 10))
idx_r_10_15 = np.array((csv_test['r'] > 10) & (csv_test['r'] <= 15))
idx_r_15_20 = np.array((csv_test['r'] > 15) & (csv_test['r'] <= 20))

y_r_0_5, y_preds0_r_0_5 = y_[idx_r_0_5], y_preds0_[idx_r_0_5]
y_r_5_10, y_preds0_r_5_10 = y_[idx_r_5_10], y_preds0_[idx_r_5_10]
y_r_10_15, y_preds0_r_10_15 = y_[idx_r_10_15], y_preds0_[idx_r_10_15]
y_r_15_20, y_preds0_r_15_20 = y_[idx_r_15_20], y_preds0_[idx_r_15_20]

range_r= ["0_5", "5_10", "10_15", "15_20"]
y_r = [y_r_0_5, y_r_5_10, y_r_10_15, y_r_15_20]
y_preds0_r = [y_preds0_r_0_5, y_preds0_r_5_10, y_preds0_r_10_15, y_preds0_r_15_20]
"""

"""
# ul_mg and r 明るさと距離の組み合わせ
r_thr, ul_mg_thr = 5, 1.5 # threshold
idx_mgr_close_dark = np.array((csv_test['r'] <= r_thr) & (csv_test['ul_mg'] <= ul_mg_thr))
idx_mgr_close_bright = np.array((csv_test['r'] <= r_thr) & (csv_test['ul_mg'] >= ul_mg_thr))
idx_mgr_far_dark = np.array((csv_test['r'] >= r_thr) & (csv_test['ul_mg'] <= ul_mg_thr))
idx_mgr_far_bright = np.array((csv_test['r'] >= r_thr) & (csv_test['ul_mg'] >= ul_mg_thr))

y_mgr_close_dark, y_preds0_mgr_close_dark = y_[idx_mgr_close_dark], y_preds0_[idx_mgr_close_dark]
y_mgr_close_bright, y_preds0_mgr_close_bright = y_[idx_mgr_close_bright], y_preds0_[idx_mgr_close_bright]
y_mgr_far_dark, y_preds0_mgr_far_dark = y_[idx_mgr_far_dark], y_preds0_[idx_mgr_far_dark]
y_mgr_far_bright, y_preds0_mgr_far_bright = y_[idx_mgr_far_bright], y_preds0_[idx_mgr_far_bright]

range_mgr= ["close_dark", "close_bright", "far_dark", "far_bright"]
y_mgr = [y_mgr_close_dark, y_mgr_close_bright, y_mgr_far_dark, y_mgr_far_bright]
y_preds0_mgr = [y_preds0_mgr_close_dark, y_preds0_mgr_close_bright, y_preds0_mgr_far_dark, y_preds0_mgr_far_bright]
"""



