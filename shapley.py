import itertools
import math
import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from scipy.special import comb
import pandas as pd



def make_all_subsets(list_of_members):
    # make every possible subsets of given list_of_members
    # for size in (list_of_members):
    # use combinations to enumerate all combinations of size elements
    # append all combinations to self.data

    set_of_all_subsets = set([])

    for i in range(len(list_of_members), -1, -1):
        for element in itertools.combinations(list_of_members, i):
            # element = sorted(element)
            set_of_all_subsets.add(frozenset(element))
    return set_of_all_subsets

class ShapleyValue:
    '''A class to produce a fuzzy measure of based on a list of criteria'''

    def __init__(self, list_of_members, mu):
        # initialize a class to hold all fuzzyMeasure related objects
        self.set_of_all_subsets = sorted(set(mu.keys()))
        self.mu = mu
        self.svs = {}
        self.list_of_members = frozenset(list_of_members)
        if len(self.mu) < 1:
            return

    def calculate_svs(self):
        # print("*************")
        memberShapley = 0
        total = 0
        factorialTotal = math.factorial(len(self.list_of_members))
        for member in self.list_of_members:
            for subset in self.set_of_all_subsets:
                if member in subset:
                    # The muSet is the mu of Mu(B U {j})
                    muSet = self.mu.get(subset)
                    remainderSet = subset.difference(set({member}))
                    muRemainer = self.mu.get(remainderSet)
                    difference = muSet - muRemainer
                    b = len(remainderSet)
                    factValue = (len(self.list_of_members) - b - 1)
                    divisor = (math.factorial(factValue) * math.factorial(b) * 1.0) / (factorialTotal * 1.0)
                    weightValue = divisor * difference
                    memberShapley = memberShapley + weightValue
            self.svs[member] = memberShapley
            # print("Shapley Value of Client " + str(member) + ": " + str(memberShapley))
            total = total + memberShapley
            memberShapley = 0
        # print("Total: " + str(total))
        # print("*************")

    def calculate_svs_perm(self, perm_ls):

        sv_dict = {member: .0 for member in self.list_of_members}
        for perm in perm_ls:
            u_pre = self.mu.get(frozenset())
            for i in range(len(perm)):
                member = perm[i]
                sv_dict[member] = sv_dict[member] + (self.mu.get(frozenset(perm[:i+1])) - u_pre) / len(perm_ls)
                u_pre = self.mu.get(frozenset(perm[:i+1]))

        self.svs = sv_dict.copy()

    def calculate_svs_group_testing(self, beta_mat, model_ls, params):
        N = len(self.list_of_members)
        T = beta_mat.shape[0]
        id_index_df = pd.DataFrame({
            "Index": np.arange(N),
            "ID": sorted(self.list_of_members)
        })

        tiled_beta_mat = np.tile(beta_mat, (1, N)).reshape((T, N, N))
        transposed_tiled_beta_mat = np.transpose(tiled_beta_mat, (0, 2, 1))

        diff_beta_mat = transposed_tiled_beta_mat - tiled_beta_mat
        utilities = np.array([self.mu[model_ls[t]] for t in range(T)]).reshape((T, 1, 1))
        diff_u_mat = np.sum(utilities * diff_beta_mat, axis=0) * params["Z"] / T

        svs_cvxpy = cp.Variable((N))
        constraints = [cp.sum(svs_cvxpy) == params["Utot"]]
        for i in range(N):
            for j in range(i, N):
                constraint = cp.abs(svs_cvxpy[i] - svs_cvxpy[j] - diff_u_mat[i, j]) <= params["epsi"] / (2 * N ** 0.5)
                constraints.append(constraint)

        objective = cp.Minimize(params["Utot"])
        prob = cp.Problem(objective, constraints)
        prob.solve()
        svs = svs_cvxpy.value

        for i in range(N):
            cid = id_index_df[id_index_df["Index"] == i]["ID"].item()
            self.svs[cid] = svs[i]

    def calculate_svs_kernel_shap(self, samples):
        phi_init = self.mu[frozenset()]
        phi_all = self.mu[self.list_of_members]
        nclients = len(self.list_of_members)
        id_index_df = pd.DataFrame({
            "Index": np.arange(nclients),
            "ID": sorted(self.list_of_members)
        })


        z_matrix, y_vec, weights = [], [], []
        for cids, weight in samples.items():
            indices = id_index_df[id_index_df["ID"].isin(cids)]["Index"].tolist()
            z = np.zeros(nclients)
            z[indices] = 1.0
            z_matrix.append(z)
            y = self.mu[frozenset(cids)]
            y_vec.append(y)
            weight = weight ** 0.5
            weights.append(weight)

        z_matrix = np.vstack(z_matrix)
        y_vec = np.array(y_vec)
        weights = np.array(weights)

        A = (z_matrix[:, :-1] - z_matrix[:, -1].reshape(-1, 1)) * weights.reshape(-1, 1)
        y_vec_ = (y_vec - z_matrix[:, -1] * phi_all - (1 - z_matrix[:, -1]) * phi_init) * weights

        # print(A)
        # print(y_vec_)

        phi_vec = np.linalg.lstsq(A, y_vec_, rcond=None)[0]
        phi_vec = np.append(phi_vec, phi_all - phi_vec.sum() - phi_init)

        for i in range(nclients):
            cid = id_index_df[id_index_df["Index"] == i]["ID"].item()
            self.svs[cid] = phi_vec[i]





# class KernelShap(nn.Module):
#     def __init__(self, n_clients):
#         super(KernelShap, self).__init__()
#         self.M = n_clients
#         self.linear = nn.Linear(self.M, 1)
#
#     def kernel(z_):
#         count_ones = z_[z_ == 1].sum(dim=1)
#         pi = (self.M - 1) / comb(self.M, count_ones) / count_ones / (self.M - count_ones)
#         return pi
#
#     def forward(self, z_):
#         return self.linear(z_)
#
#     def loss(self, z_, y):




# if __name__ == '__main__':


# mu = {frozenset(): 0.4, frozenset({1}): 0.5, frozenset({2}):0.6, frozenset({1,2}): 0.9}
# sv = ShapleyValue([1, 2], mu)
# print(sv.svs)

