from client import *
from hesv import *
from securesv import *

model_func_dict = {
    "mnist_cnn": (HE_MNIST_CNN, Sec_MNIST_CNN),
    "bank_logi": (HE_BANK_Logi, Sec_BANK_Logi),
    "agnews_logi": (HE_AGNEWS_Logi, Sec_AGNEWS_Logi),
    "mrna_rnn": (HE_mRNA_RNN, Sec_mRNA_RNN)
}

if __name__ == '__main__':

    for run in tqdm(range(0, 5)):
        print("\n-----------------------------------")
        print("Experiment run:", run)
        for dist in ["dir0.5sr0.1"]:
            for model in ["mnist_cnn", "bank_logi", "agnews_logi", "mrna_rnn"]:
            # for model in ["mnist_cnn1"]:
                dirs = "%s/%s/%s/" % (model, dist, run)
                print("\n-----------------------------------")
                print("\n-----------------------------------")
                print(dirs)
                clients = Clients(dirs)
                clients.load("clients.data")

                he_model, sec_model = model_func_dict[model]
                titles_svevals = [
                    ("\nHESV", HESV(clients, he_model()), "hesv.json"),
                    ("\nSecSV", SecSV(clients, sec_model()), ["secsv.json", "secsv_skip.json"]),
                    ("\nSecSV + Permutation Sampling", SecSVPerm(clients, sec_model(), 0.25, 0.1),
                     ["secsv_ps.json", "secsv_ps_skip.json"]),
                    ("\nSecSV + Group Testing", SecSVGroupTesting(clients, sec_model(), 0.25, 0.1),
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





