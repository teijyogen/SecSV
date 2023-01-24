from client import Clients
from hesv import *
from securesv import *
from secretsv import *
from sveval import *
import numpy as np
np.seterr(all="ignore")
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--track', type=str, default="main")
parser.add_argument('--device', type=str, default="")
args = parser.parse_args()

if __name__ == '__main__':

    if args.track == "main":
        print("\nEvaluate SecSV for main track")
        # for model in ["agnews_logi"]:
        for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            sec_model = sec_model_func_dict[model]
            sveval = SecSV(clients, sec_model())
            sveval.sv_eval_mul_rnds_rparallel()
            suffix = "8_23_27_new"
            sveval.save_stat("secsv", skip=False, suffix=suffix)
            sveval.save_stat("secsv_skip", skip=True, suffix=suffix)
            sveval.save("secsv", suffix=suffix)

        print("\nEvaluate Kernel SHAP for main track")
        for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            sec_model = sec_model_func_dict[model]
            sveval = SecSVKernelSHAP(clients, sec_model())
            sveval.sv_eval_mul_rnds_rparallel()
            suffix = "8_23_27_new"
            sveval.save_stat("secsv_ks", skip=False, suffix=suffix)
            sveval.save_stat("secsv_ks_skip", skip=True, suffix=suffix)
            sveval.save("secsv_ks", suffix=suffix)

        print("\nEvaluate Group Testing for main track")
        for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            sec_model = sec_model_func_dict[model]
            sveval = SecSVGroupTesting(clients, sec_model(), 0.25, 0.1)
            sveval.sv_eval_mul_rnds_rparallel()
            suffix = "8_23_27_new"
            sveval.save_stat("secsv_gt", skip=False, suffix=suffix)
            sveval.save_stat("secsv_gt_skip", skip=True, suffix=suffix)
            sveval.save("secsv_gt", suffix=suffix)

        print("\nEvaluate Permutation Sampling for main track")
        for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            sec_model = sec_model_func_dict[model]
            sveval = SecSVPerm(clients, sec_model(), 0.25, 0.1)
            sveval.sv_eval_mul_rnds_rparallel()
            suffix = "8_23_27_new"
            sveval.save_stat("secsv_ps", skip=False, suffix=suffix)
            sveval.save_stat("secsv_ps_skip", skip=True, suffix=suffix)
            sveval.save("secsv_ps", suffix=suffix)

        print("\nEvaluate HESV for main track")
        for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            he_model = he_model_func_dict[model]
            sveval = HESV(clients, he_model())
            sveval.sv_eval_mul_rnds_rparallel()
            suffix = "8_23_27_new"
            sveval.save_stat("hesv", suffix=suffix)
            sveval.save("hesv", suffix=suffix)

        print("\nEvaluate SV for main track")
        for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            sveval = SVEval(clients, no_cuda=True)
            sveval.skip = False
            sveval.skip_compare = False
            sveval.sv_eval_mul_rnds_rparallel()
            suffix = "cpu"
            sveval.save_stat("sv", suffix=suffix)
            sveval.save("sv", suffix=suffix)

        print("\nEvaluate SecretSV for main track")
        for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            he_model = ss_model_func_dict[model]
            sveval = SecretSV(clients, he_model())
            sveval.sv_eval_mul_rnds_rparallel()
            suffix = "8_23_27"
            sveval.save_stat("secretsv", suffix=suffix)
            sveval.save("secretsv", suffix=suffix)

    elif args.track == "usage":
        print("\nEvaluate HESV for main track")
        for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
            for usage in [0.1, 0.25, 0.5, 0.75]:
                exp_name = f"usage/{model}/usage{usage}/{args.run}/"
                print(exp_name)
                clients = Clients(exp_name)
                clients.load("clients.data")
                he_model = he_model_func_dict[model]
                sveval = HESV(clients, he_model(), test_data_usage=usage)
                sveval.sv_eval_mul_rnds_rparallel()
                suffix = "8_23_27_new"
                sveval.save_stat("hesv", suffix=suffix)
                sveval.save("hesv", suffix=suffix)

        print("\nEvaluate SecSV for usage track")
        for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
            for usage in [0.1, 0.25, 0.5, 0.75]:
                exp_name = f"usage/{model}/usage{usage}/{args.run}/"
                print(exp_name)
                clients = Clients(exp_name)
                clients.load("clients.data")
                sec_model = sec_model_func_dict[model]
                sveval = SecSV(clients, sec_model(), test_data_usage=usage)
                sveval.sv_eval_mul_rnds_rparallel()
                suffix = "8_23_27_new"
                sveval.save_stat("secsv", skip=False, suffix=suffix)
                sveval.save_stat("secsv_skip", skip=True, suffix=suffix)
                sveval.save("secsv", suffix=suffix)

    elif args.track == "ncl":
        print("\nEvaluate SecSV for ncl track")
        for ncl in tqdm([4, 6, 8, 10]):
            for model in ["bank_logi", "agnews_logi", "mnist_cnn", "mrna_rnn"]:
                exp_name = f"ncl/{model}/ncl{ncl}/{args.run}/"
                print(exp_name)
                clients = Clients(exp_name)
                clients.load("clients.data")
                sec_model = sec_model_func_dict[model]
                sveval = SecSV(clients, sec_model(), test_data_usage=0.1)
                sveval.sv_eval_mul_rnds_rparallel()
                suffix = "8_23_27_new"
                sveval.save_stat("secsv", skip=False, suffix=suffix)
                sveval.save_stat("secsv_skip", skip=True, suffix=suffix)
                sveval.save("secsv", suffix=suffix)

    elif args.track == "skip":
        print("\nEvaluate skip track")
        for run in range(100):
            for model in ["bank_logi", "agnews_logi"]:
                exp_name = f"skip/{model}/{run}/"
                print(exp_name)
                clients = Clients(exp_name)
                clients.load("clients.data")
                sec_model = sec_model_func_dict[model]
                sec_model = sec_model()
                sec_model.gamma1 = 27
                sec_model.gamma2 = 100
                sveval = SecSV_Skip_Effect(clients, sec_model)
                sveval.int_prec = 8
                sveval.frac_prec = 23
                sveval.sv_eval_mul_rnds_rparallel()
                suffix = "8_23_27_"
                sveval.save_stat("vary_skip_rate", suffix=suffix)
                sveval.save("vary_skip_rate", suffix=suffix)

    elif args.track == "nlayer":
        # for model in ["agnews_dnn", "bank_dnn", "mnist_dnn", "mrna_dnn"]:
        #     for nlayer in (1, 3, 5, 7):
        #         for run in tqdm(range(10)):
        for model in ["agnews_dnn"]:
            for nlayer in (5,):
                for run in tqdm(range(1)):
                    exp_name = f"nlayer/{model}/nlayer{nlayer}/{run}/"
                    print(exp_name)
                    clients = Clients(exp_name)
                    clients.load("clients.data")
                    # sec_model = sec_model_func_dict[model]
                    # sveval = SecSV(clients, sec_model(layer_nb=nlayer), test_data_usage=1.0)
                    he_model = he_model_func_dict[model]
                    sveval = HESV(clients, he_model(layer_nb=nlayer), test_data_usage=1.0)
                    # sveval.debug = True
                    # sveval.skip = False
                    # sveval.skip_init_model = False
                    sveval.sv_eval_mul_rnds_rparallel()

    elif args.track == "hesv":
        print("\nTest HESV")
        for model in ["mrna_rnn"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            he_model = he_model_func_dict[model]
            sveval = HESV(clients, he_model(), test_data_usage=0.1)
            # sveval.debug = True
            # sveval.skip_init_model = False
            sveval.sv_eval_mul_rnds_rparallel()

    elif args.track == "secsv":
        print("\nTest SecSV")
        for model in ["agnews_logi"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            sec_model = sec_model_func_dict[model]
            sveval = SecSV(clients, sec_model(), test_data_usage=1.0)
            # sveval.debug = True
            # sveval.skip = False
            # sveval.skip_init_model = False
            sveval.sv_eval_mul_rnds_rparallel()

    elif args.track == "secretsv":
        print("\nTest SecretSV")
        for model in ["agnews_logi"]:
            exp_name = f"main/{model}/{args.run}/"
            print(exp_name)
            clients = Clients(exp_name)
            clients.load("clients.data")
            ss_model = ss_model_func_dict[model]
            sveval = SecretSV(clients, ss_model())
            sveval.sv_eval_mul_rnds_rparallel()

    elif args.track == "sv":
        for model in ["mrna_rnn"]:
            dir = f"main/{model}/{args.run}/"
            clients = Clients(dir)
            clients.load("clients.data")
            sveval = SVEval(clients, no_cuda=True)
            sveval.skip = False
            sveval.sv_eval_mul_rnds_rparallel()

    elif args.track == "sv_comp":
        for model in ["mnist_cnn", "mrna_rnn"]:
            for run in tqdm(range(10)):
                dir = f"main/{model}/{run}/"
                clients = Clients(dir)
                clients.load("clients.data")
                sveval = SVEval(clients)
                sveval.skip = True
                sveval.skip_compare = True
                sveval.sv_eval_mul_rnds_rparallel()
                sveval.save_stat("sv_comp", skip=True)

    elif args.track == "dnn":
        for model in ["agnews_dnn", "bank_dnn", "mnist_dnn", "mrna_dnn"]:
            for nlayer in (1, 3, 5, 7):
                for run in tqdm(range(10)):
                    dir = f"nlayer/{model}/nlayer{nlayer}/{run}/"
                    clients = Clients(dir)
                    clients.load("clients.data")
                    sveval = SVEval(clients, nocuda=False)
                    sveval.sv_eval_mul_rnds_rparallel()
                    sveval.save_stat("sv.json")

                    clients = Clients(dir)
                    clients.load("clients.data")
                    sveval = SVEval(clients, no_cuda=False)
                    sveval.nonskip = False
                    sveval.skip = True
                    sveval.skip_compare = True
                    sveval.sv_eval_mul_rnds_rparallel()
                    sveval.save_stat("sv_skip_comp", skip=True)

    else:
        for gamma1, gamma2 in [(25, 102), (27, 100), (29, 100)]:
            suffix = f"8_23_{gamma1}_{args.device}"
            print(suffix)
            for model in ["bank_logi", "agnews_logi"]:
                for run in tqdm(range(0, 10)):
                    exp_name = f"main/{model}/{run}/"
                    print(exp_name)
                    clients = Clients(exp_name)
                    clients.load("clients.data")
                    sec_model = sec_model_func_dict[model]
                    sec_model = sec_model()
                    # sec_model.input_nb *= 2
                    sec_model.gamma1 = gamma1
                    sec_model.gamma2 = gamma2
                    sveval = SecSV(clients, sec_model)
                    sveval.poly_modulus_degree = 2 ** 12
                    sveval.int_prec = 8
                    sveval.frac_prec = 23
                    sveval.sv_eval_mul_rnds_rparallel()
                    sveval.save_stat("secsv", skip=False, suffix=suffix)
                    sveval.save_stat("secsv_skip", skip=True, suffix=suffix)
                    sveval.save("secsv", suffix=suffix)








