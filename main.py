from client import *
from hesv import *
from securesv import *
from tqdm import tqdm

if __name__ == '__main__':

    for run in tqdm(range(0, 10)):
        dirs = "data/mnist_cnn1/iid/%s/" % (run)
        print("\n-----------------------------------")
        print("\n-----------------------------------")
        print(dirs)
        clients = Clients()
        clients.dirs = dirs
        clients.load("clients.data")

        print("\n-----------------------------------")
        print("\nSecSV")
        sveval = SecSV(clients, Sec_CNN1_MNIST())
        sveval.skip = False
        # sveval.input_shape = (-1, 784)
        # sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("secsv.json")
        del sveval

        print("\n-----------------------------------")
        print("\nSecSV+SampleSkip")
        sveval = SecSV(clients, Sec_CNN1_MNIST())
        sveval.skip = True
        # sveval.input_shape = (-1, 784)
        # sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("secsv_skip.json")
        del sveval

        print("\n-----------------------------------")
        print("\nHESV")
        sveval = HESV(clients, HE_CNN1_MNIST())
        # sveval.input_shape = (-1, 784)
        # sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("hesv.json")
        del sveval
        del clients


        dirs = "data/mnist_cnn1/dir0.5/%s/" % (run)
        print("\n-----------------------------------")
        print("\n-----------------------------------")
        print(dirs)
        clients = Clients()
        clients.dirs = dirs
        clients.load("clients.data")

        print("\n-----------------------------------")
        print("\nSecSV")
        sveval = SecSV(clients, Sec_CNN1_MNIST())
        sveval.skip = False
        # sveval.input_shape = (-1, 784)
        # sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("secsv.json")
        del sveval

        print("\n-----------------------------------")
        print("\nSecSV+SampleSkip")
        sveval = SecSV(clients, Sec_CNN1_MNIST())
        sveval.skip = True
        # sveval.input_shape = (-1, 784)
        # sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("secsv_skip.json")
        del sveval

        print("\n-----------------------------------")
        print("\nHESV")
        sveval = HESV(clients, HE_CNN1_MNIST())
        # sveval.input_shape = (-1, 784)
        # sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("hesv.json")
        del sveval
        del clients


        dirs = "data/mnist_logi/iid/%s/" % (run)
        print("\n-----------------------------------")
        print("\n-----------------------------------")
        print(dirs)
        clients = Clients()
        clients.dirs = dirs
        clients.load("clients.data")

        print("\n-----------------------------------")
        print("\nSecSV")
        sveval = SecSV(clients, Sec_Logi_MNIST())
        sveval.skip = False
        sveval.input_shape = (-1, 784)
        sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("secsv.json")
        del sveval

        print("\n-----------------------------------")
        print("\nSecSV+SampleSkip")
        sveval = SecSV(clients, Sec_Logi_MNIST())
        sveval.skip = True
        sveval.input_shape = (-1, 784)
        sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("secsv_skip.json")
        del sveval

        print("\n-----------------------------------")
        print("\nHESV")
        sveval = HESV(clients, HE_Logi_MNIST())
        sveval.input_shape = (-1, 784)
        sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("hesv.json")
        del sveval
        del clients


        dirs = "data/mnist_logi/dir0.5/%s/" % (run)
        print("\n-----------------------------------")
        print("\n-----------------------------------")
        print(dirs)
        clients = Clients()
        clients.dirs = dirs
        clients.load("clients.data")

        print("\n-----------------------------------")
        print("\nSecSV")
        sveval = SecSV(clients, Sec_Logi_MNIST())
        sveval.skip = False
        sveval.input_shape = (-1, 784)
        sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("secsv.json")
        del sveval

        print("\n-----------------------------------")
        print("\nSecSV+SampleSkip")
        sveval = SecSV(clients, Sec_Logi_MNIST())
        sveval.skip = True
        sveval.input_shape = (-1, 784)
        sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("secsv_skip.json")
        del sveval

        print("\n-----------------------------------")
        print("\nHESV")
        sveval = HESV(clients, HE_Logi_MNIST())
        sveval.input_shape = (-1, 784)
        sveval.batch_size = 784
        sveval.sv_eval_mul_rnds_rparallel()
        sveval.dirs = clients.dirs
        sveval.save_stat("hesv.json")
        del sveval
        del clients
