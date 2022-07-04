from client import *
from hesv import *
from securesv import *

if __name__ == '__main__':

    for run in tqdm(range(1, 2)):
        print("\n-----------------------------------")
        print("Experiment run:", run)
        for dist in ["dir0.5"]:
            # for model in ["mnist_cnn1", "cifar_cnn2"]:
            for model in ["cifar_cnn2"]:
                dirs = "data/%s/%s/%s/" % (model, dist, run)
                print("\n-----------------------------------")
                print("\n-----------------------------------")
                print(dirs)
                clients = Clients()
                clients.dirs = dirs
                clients.load("clients.data")

                if model == "mnist_cnn1":
                    titles_svevals = [
                        ("\nHESV", HESV(clients, HE_CNN1_MNIST()), "hesv.json"),
                        ("\nSecSV", SecSV(clients, Sec_CNN1_MNIST()), ["secsv.json", "secsv_skip.json"]),
                        ("\nSecSV + Permutation Sampling", SecSVPerm(clients, Sec_CNN1_MNIST(), 0.25, 0.1),
                         ["secsv_ps.json", "secsv_ps_skip.json"]),
                        ("\nSecSV + Group Testing", SecSVGroupTesting(clients, Sec_CNN1_MNIST(), 0.25, 0.1),
                         ["secsv_gt.json", "secsv_gt_skip.json"])
                    ]
                else:
                    titles_svevals = [
                        # ("\nHESV", HESV(clients, HE_CNN2_CIFAR()), "hesv.json"),
                        ("\nSecSV", SecSV(clients, Sec_CNN2_CIFAR()), ["secsv.json", "secsv_skip.json"]),
                        ("\nSecSV + Permutation Sampling", SecSVPerm(clients, Sec_CNN2_CIFAR(), 0.25, 0.1),
                         ["secsv_ps.json", "secsv_ps_skip.json"]),
                        ("\nSecSV + Group Testing", SecSVGroupTesting(clients, Sec_CNN2_CIFAR(), 0.25, 0.1),
                         ["secsv_gt.json", "secsv_gt_skip.json"])
                    ]

                for title, sveval, filename in tqdm(titles_svevals):
                    print("\n-----------------------------------")
                    print(title)
                    sveval.dirs = clients.dirs
                    sveval.sv_eval_mul_rnds_rparallel()
                    if type(sveval) == HESV:
                        sveval.save_stat(filename)
                    else:
                        sveval.save_stat(filename[0], skip=False)
                        sveval.save_stat(filename[1], skip=True)

                    del sveval



    # for run in tqdm(range(0, 1)):
    #     for dist in ["dir0.5"]:
    #         for model in ["mnist_cnn1"]:
    #             dirs = "data/%s/%s/%s/" % (model, dist, run)
    #             print("\n-----------------------------------")
    #             print("\n-----------------------------------")
    #             print(dirs)
    #             clients = Clients()
    #             clients.dirs = dirs
    #             clients.load("clients.data")
    #
    #             if model == "mnist_cnn1":
    #                 titles_svevals = [
    #                     # ("\nHESV", HESV(clients, HE_CNN1_MNIST()), "hesv.json"),
    #                     ("\nSecSV", SecSV(clients, Sec_CNN1_MNIST()), "secsv.json"),
    #                     ("\nSecSV", SecSV(clients, Sec_CNN1_MNIST()), "secsv_skip.json"),
    #                     # ("\nSecSV + Permutation Sampling", SecSVPerm(clients, Sec_CNN1_MNIST(), 0.25, 0.1), "secsv_ps.json"),
    #                     # ("\nSecSV + Group Testing", SecSVGroupTesting(clients, Sec_CNN1_MNIST(), 0.25, 0.1), "secsv_gt.json")
    #                 ]
    #             else:
    #                 titles_svevals = [
    #                     ("\nHESV", HESV(clients, HE_CNN2_CIFAR()), "hesv.json"),
    #                     ("\nSecSV", SecSV(clients, Sec_CNN2_CIFAR()), ["secsv.json", "secsv_skip.json"]),
    #                     ("\nSecSV + Permutation Sampling", SecSVPerm(clients, Sec_CNN2_CIFAR(), 0.25, 0.1),
    #                      ["secsv_ps.json", "secsv_ps_skip.json"]),
    #                     ("\nSecSV + Group Testing", SecSVGroupTesting(clients, Sec_CNN2_CIFAR(), 0.25, 0.1),
    #                      ["secsv_gt.json", "secsv_gt_skip.json"])
    #                 ]
    #
    #             for title, sveval, filename in tqdm(titles_svevals):
    #                 print("\n-----------------------------------")
    #                 print(title)
    #                 sveval.dirs = clients.dirs
    #                 if type(sveval) == HESV:
    #                     sveval.sv_eval_mul_rnds_rparallel()
    #                     sveval.save_stat(filename)
    #                 elif type(sveval) == SecSV and filename == "secsv.json":
    #                     sveval.noskip = True
    #                     sveval.sv_eval_mul_rnds_rparallel()
    #                     sveval.save_stat(filename, skip=False)
    #                 elif type(sveval) == SecSV and filename == "secsv_skip.json":
    #                     sveval.skip = True
    #                     sveval.sv_eval_mul_rnds_rparallel()
    #                     sveval.save_stat(filename, skip=True)
    #                 else:
    #                     sveval.noskip = True
    #                     sveval.save_stat(filename, skip=False)
    #                 del sveval




