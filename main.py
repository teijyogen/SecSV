from client import *
from hesv import *
from securesv import *

if __name__ == '__main__':

    for run in tqdm(range(5, 10)):
        for dist in ["iid", "dir0.5"]:
            dirs = "data/mnist_cnn1/%s/%s/" % (dist, run)
            print("\n-----------------------------------")
            print("\n-----------------------------------")
            print(dirs)
            clients = Clients()
            clients.dirs = dirs
            clients.load("clients.data")

            print("\n-----------------------------------")
            print("\nSecSV")
            sveval = SecSV(clients, Sec_CNN1_MNIST())
            sveval.sv_eval_mul_rnds_rparallel()
            sveval.dirs = clients.dirs
            sveval.save_stat("secsv_.json", skip=False)
            sveval.save_stat("secsv_skip_.json", skip=True)
            del sveval

            print("\n-----------------------------------")
            print("\nSecSV + Permutation Sampling")
            sveval = SecSVPerm(clients, Sec_CNN1_MNIST(), 0.25, 0.1)
            sveval.sv_eval_mul_rnds_rparallel()
            sveval.dirs = clients.dirs
            sveval.save_stat("secsv_ps.json", skip=False)
            sveval.save_stat("secsv_ps_skip.json", skip=True)
            del sveval

            print("\n-----------------------------------")
            print("\nSecSV + Group Testing")
            sveval = SecSVGroupTesting(clients, Sec_CNN1_MNIST(), 0.25, 0.1)
            sveval.sv_eval_mul_rnds_rparallel()
            sveval.dirs = clients.dirs
            sveval.save_stat("secsv_gt.json", skip=False)
            sveval.save_stat("secsv_gt_skip.json", skip=True)
            del sveval

            dirs = "data/mnist_logi/%s/%s/" % (dist, run)
            print("\n-----------------------------------")
            print("\n-----------------------------------")
            print(dirs)
            clients = Clients()
            clients.dirs = dirs
            clients.load("clients.data")

            print("\n-----------------------------------")
            print("\nSecSV")
            sveval = SecSV(clients, Sec_Logi_MNIST())
            sveval.input_shape = (-1, 784)
            sveval.batch_size = 392
            sveval.sv_eval_mul_rnds_rparallel()
            sveval.dirs = clients.dirs
            sveval.save_stat("secsv_.json", skip=False)
            sveval.save_stat("secsv_skip_.json", skip=True)
            del sveval

            print("\n-----------------------------------")
            print("\nSecSV + Permutation Sampling")
            sveval = SecSVPerm(clients, Sec_Logi_MNIST(), 0.25, 0.1)
            sveval.input_shape = (-1, 784)
            sveval.batch_size = 392
            sveval.sv_eval_mul_rnds_rparallel()
            sveval.dirs = clients.dirs
            sveval.save_stat("secsv_ps.json", skip=False)
            sveval.save_stat("secsv_ps_skip.json", skip=True)
            del sveval

            print("\n-----------------------------------")
            print("\nSecSV + Group Testing")
            sveval = SecSVGroupTesting(clients, Sec_Logi_MNIST(), 0.25, 0.1)
            sveval.input_shape = (-1, 784)
            sveval.batch_size = 392
            sveval.sv_eval_mul_rnds_rparallel()
            sveval.dirs = clients.dirs
            sveval.save_stat("secsv_gt.json", skip=False)
            sveval.save_stat("secsv_gt_skip.json", skip=True)
            del sveval