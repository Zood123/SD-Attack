#!/usr/bin/env python
# coding: utf-8
import pickle
import numpy as np
from SD_Attack import utils, GCN
from SD_Attack import SDA
from tqdm import tqdm
import os
import scipy.sparse as sp
from scipy import linalg
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter




gpu_id = 0
cur_path = os.path.abspath('.')

for path in ["citeseer","cora","pubmed"]:
    if not os.path.exists(os.path.join(cur_path, 'SD_Attack_logs', path)):
            os.makedirs(os.path.join(cur_path, 'SD_Attack_logs', path))
parser = ArgumentParser("rdlink_gcn",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument("--dataset", default='cora', help='dataset string.') # 'citeseer', 'cora', 'pubmed'
parser.add_argument("--load_eigenvalue", default=None, help='load_eigenvalue dict.')
parser.add_argument("--load_attack", default=None, help=' load attack.')


args = parser.parse_args()
dataset = args.dataset
load_eigenvalue = args.load_eigenvalue
load_attack = args.load_attack

_A_obs = sp.load_npz("data/" + dataset + "_adj.npz")
_X_obs = sp.load_npz("data/" + dataset + "_features.npz")
_z_obs = np.load("data/" + dataset + "_labels.npy")

perturb_save_logs = os.path.join(cur_path, 'SD_Attack_logs/' + dataset + '/cora_attack.txt')

_A_obs = _A_obs + _A_obs.T
_A_obs[_A_obs > 1] = 1
lcc = utils.largest_connected_components(_A_obs)
_A_obs = _A_obs[lcc][:,lcc] #Use the largest connected_component for train




assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes" #each node should have at least one edge


_X_obs = sp.csr_matrix(_X_obs[lcc].astype('float32'))
_z_obs = _z_obs[lcc]
_X_obs = _X_obs.astype('float32')
_N = _A_obs.shape[0]
_K = _z_obs.max()+1
_Z_obs = np.eye(_K)[_z_obs]
_An = utils.preprocess_graph(_A_obs)
sizes = [32, _K]
degrees = _A_obs.sum(0).A1
np.random.seed(0)


seed = 1
unlabeled_share = 0.8
val_share = 0.1
train_share = 1 - unlabeled_share - val_share
np.random.seed(seed)

split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(np.arange(_N),
                                                                       train_size=train_share,
                                                                       val_size=val_share,
                                                                       test_size = 0.2,  #cora:0.2 citeseer:0.2(425) pubmed: test_size=0.015,
                                                                       stratify= _z_obs, random_state= seed)



gcn_before = GCN.GCN(sizes, utils.preprocess_features(_X_obs), "gcn", gpu_id=gpu_id)
gcn_before.train(_An, split_train, split_val, _Z_obs)
gcn_before.test(split_unlabeled, _Z_obs, _An)


pbar = tqdm(range(len(split_unlabeled)))
attacked = 0
A_processed = _An
A_I = _A_obs
A_I[A_I > 1] = 1
rowsum = A_I.sum(1).A1
degree_mat = sp.diags(rowsum)



if load_eigenvalue is  None:
    eig_vals, eig_vec = linalg.eigh(A_I.todense(), degree_mat.todense())

    save_eigen = {"eig_vals": eig_vals, "eig_vec" : eig_vec}

    with open(os.path.join(cur_path, 'SD_Attack_logs/' + dataset + '/save_eigen'), 'wb') as handle:
        pickle.dump(save_eigen, handle)
else:
    with open(os.path.join(cur_path,'SD_Attack_logs/' + dataset+load_eigenvalue), 'rb') as handle:
        eigen = pickle.load(handle)
        eig_vals = eigen["eig_vals"]
        eig_vec  = eigen["eig_vec"]


eigen_vals2  = utils.eigen_2(eig_vals,k=len(eig_vals)) #k=64 for pubmed
eig_vec2 = np.sqrt(degree_mat) * eig_vec
coef = np.transpose(eig_vec)


read_attack =perturb_save_logs

if read_attack != None:
    load_target,edges = utils.read_attack(perturb_save_logs)
else:
    load_target = []

for pos in pbar:
    u = split_unlabeled[pos]

    if gcn_before.test ([u], _Z_obs, _An ) == 0:
        attacked += 1
        continue

    if u in load_target:
        k = load_target.index(u)
        perturbed = utils.change_A(edges[k],_A_obs)
        acc_after = gcn_before.test([u], _Z_obs, perturbed)

        if acc_after < 1.0:
            attacked += 1
        continue

    inverse_u = utils.coef_calculate_one(eig_vals, eig_vec, u, eigen_vals2,coef,len(eig_vals))

    GF_Attack = SDA.W1_attack(_A_obs, _z_obs, u, eig_vals, eig_vec, perturb_save_logs,degrees, eig_vec2,inverse_u,A_I)
    GF_Attack.attack_model()

    print ('After attack:')
    acc_after = gcn_before.test ( [u], _Z_obs, GF_Attack.adj_preprocessed)

    if acc_after < 1.0:
        attacked += 1

    pbar.set_description('current attack: {}'.format(attacked))

print('Final accuracy after attack is: {} \n'.format(1.0 - attacked/len(split_unlabeled)))
gcn_before.test ( split_unlabeled, _Z_obs , _An)
