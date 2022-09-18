import json
import numpy as np
import pandas as pd
from sveval import SVEval
from client import Clients
from tqdm import tqdm
import types
from utils import plot
import os
pd.set_option("display.precision", 6)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_dict(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def save_dict(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    f.close()

def preprocess_skip_dict():
    for model in ["mrna_rnn", "bank_logi", "agnews_logi", "mnist_cnn"]:
        for dirt in [0.5, 1.0, 2.0, 5.0, 10.0]:
            for r in range(10):
                filename = f"data/{model}/dirt{dirt}sr1.0/{r}/sv_skip.json"
                whole_dict = load_dict(filename)
                skip_dict = whole_dict["skip"]
                total_corrects, corrects, naive, wrong_naive = 0, 0, 0, 0

                for sub_dict in skip_dict.values():
                    print(type(sub_dict))
                    if type(sub_dict) == dict:
                        for subsub_dict in sub_dict.values():
                            total_corrects += subsub_dict["total corrects"]
                            corrects += subsub_dict["corrects"]
                            naive += subsub_dict["naive"]
                            wrong_naive += len(subsub_dict["wrong naive"])

                skip_dict["total corrects"] = total_corrects
                skip_dict["corrects"] = corrects
                skip_dict["naive"] = naive
                skip_dict["wrong naive"] = wrong_naive

                whole_dict["skip"] = skip_dict
                save_dict(whole_dict, filename)

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
    print("MSE:", ((df ** 2).sum(axis=1) ** 0.5).mean())
    print("MAE:", df.abs().sum(axis=1).mean())

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
        skip_dict = {"total corrects": skip_dict["total corrects"], "corrects": skip_dict["corrects"],
                     "naive": skip_dict["naive"], "wrong naive": skip_dict["wrong naive"]}
        # df = pd.DataFrame(skip_dict)
        # df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index).sum()
        # print(df)
        dict_ls.append(skip_dict)
    df = pd.DataFrame(dict_ls)
    return df.mean()

def cal_wrong_rate(benchmark, distribution, filename, run_nb=10):
    df = cal_skip(benchmark, distribution, filename, run_nb)
    sr = df["naive"] / df["total corrects"]
    fr = df["wrong naive"] / df["naive"]
    return sr, fr

def cal_speedup(benchmark, distribution, filename, run_nb=10):
    df_ls = []
    for run in range(run_nb):
        dirs = "data/%s/%s/%s/" % (benchmark, distribution, run)
        he_dict = load_dict(dirs+"hesv.json")
        target_dict = load_dict(dirs+filename)
        df1 = pd.DataFrame(he_dict["time"], index=[run])
        df2 = pd.DataFrame(target_dict["time"], index=[run])
        df_ls.append(df1 / df2)

    df = pd.concat(df_ls)
    print(df.mean())

def plot_speedup_secsv():
    x_list = [[20, 40, 60, 80, 100] for i in range(4)]
    y_list = [
        [2.414064, 3.264778, 3.762762, 4.40272, 4.648164],
        [4.563482, 7.693457, 10.516691, 12.408015, 14.202511],
        [3.213318, 4.157264, 4.293033, 4.374724, 4.479895],
        [5.305614, 6.378831, 7.077235, 7.912266, 7.607498]
    ]

    labels = ["AGNEWS", "BANK", "MNIST", "miRNA-mRNA"]
    title = "SecSV"
    xlabel = "% of test samples"
    ylabel = "Speedup"
    filename = "figure/speedup_secsv.png"

    plot(x_list, y_list, labels, title, filename, xlabel, ylabel)

def plot_speedup_sampleskip():
    x_list = [[20, 40, 60, 80, 100] for i in range(4)]
    y_list = [
        [2.588446, 3.652167, 4.489445, 5.299566, 5.875017],
        [4.594245, 8.381192, 12.046988, 14.32128, 18.003027],
        [6.186935, 8.274224, 9.704975, 10.95685, 12.581319],
        [12.027789, 12.280512, 13.76509, 14.678981, 15.614444]
    ]

    labels = ["AGNEWS", "BANK", "MNIST", "miRNA-mRNA"]
    title = "SecSV+SampleSkip"
    xlabel = "% of test samples"
    ylabel = "Speedup"
    filename = "figure/speedup_sampleskip.png"

    plot(x_list, y_list, labels, title, filename, xlabel, ylabel)

if __name__ == '__main__':
    plot_speedup_secsv()
    plot_speedup_sampleskip()

    # print(cal_time_dict("mrna_rnn", "dirt0.5sr1.0", "secsv_gt_skip.json", run_nb=10))
    # print(cal_mae_msv("mrna_rnn", "dirt0.5sr1.0", "secsv_gt_skip.json", run_nb=10))

    # print(cal_speedup("mrna_rnn", "dirt0.5sr1.0", "secsv.json", run_nb=10))
    # print(cal_speedup("mrna_rnn", "dirt0.5sr1.0", "secsv_skip.json", run_nb=10))

