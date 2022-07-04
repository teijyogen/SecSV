import json
import numpy as np
import pandas as pd
from sveval import SVEval
from client import Clients
from tqdm import tqdm
# pd.set_option('precision', 12)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_dict(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def cal_mae_msv(benchmark, distribution, filename, run_nb=10):
    df_ls = []
    for run in range(run_nb):
        dirs = "data/%s/%s/%s/" % (benchmark, distribution, run)
        exact_dict = load_dict(dirs+"sv.json")
        target_dict = load_dict(dirs+filename)
        df1 = pd.DataFrame(exact_dict["msv"], index=[run])
        df2 = pd.DataFrame(target_dict["msv"], index=[run])
        df_ls.append(df1 - df2)

    df = pd.concat(df_ls)
    return df.abs().mean().mean()

def cal_mae_ssv(benchmark, distribution, filename, run_nb=10):
    df_ls = []
    for run in range(run_nb):
        dirs = "data/%s/%s/%s/" % (benchmark, distribution, run)
        exact_dict = load_dict(dirs+"sv.json")
        target_dict = load_dict(dirs+filename)
        df1 = pd.DataFrame(exact_dict["ssv"])
        df2 = pd.DataFrame(target_dict["ssv"])
        df_ls.append(df1 - df2)

    df = pd.concat(df_ls)
    return df.abs().mean().mean()

def cal_time_dict(benchmark, distribution, filename, run_nb=10):
    dict_ls = []
    for run in range(run_nb):
        dirs = "data/%s/%s/%s/" % (benchmark, distribution, run)
        skip_dict = load_dict(dirs+filename)["time"]
        df = pd.DataFrame(skip_dict, index=[run])
        # print(df)
        dict_ls.append(df)
    df = pd.concat(dict_ls).mean()
    return df

def cal_skip(benchmark, distribution, filename, run_nb=10):
    dict_ls = []
    for run in range(run_nb):
        dirs = "data/%s/%s/%s/" % (benchmark, distribution, run)
        skip_dict = load_dict(dirs+filename)["skip"]
        df = pd.DataFrame(skip_dict)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index).sum()
        # print(df)
        dict_ls.append(df)
    df = pd.DataFrame(dict_ls).sum()
    return df

def cal_wrong_rate(benchmark, distribution, filename, run_nb=10):
    df = cal_skip(benchmark, distribution, filename, run_nb)
    # failure_rate = (df["wrong naive"] + df["wrong hard"]) / (df["naive size"] + df["hard size"])
    failure_rate_naive = df["wrong naive"] / df["naive size"]
    # failure_rate_hard = df["wrong hard"] / df["hard size"]
    return failure_rate_naive,

if __name__ == '__main__':

    print(cal_wrong_rate("cifar_cnn2", "dir0.1", "sv_skip+.json"))
    print(cal_wrong_rate("cifar_cnn2", "dir0.5", "sv_skip+.json"))
    print(cal_wrong_rate("cifar_cnn2", "dir1.0", "sv_skip+.json"))
    print(cal_wrong_rate("cifar_cnn2", "dir2.0", "sv_skip+.json"))
    print(cal_wrong_rate("cifar_cnn2", "dir5.0", "sv_skip+.json"))
    print(cal_wrong_rate("cifar_cnn2", "iid", "sv_skip+.json"))

