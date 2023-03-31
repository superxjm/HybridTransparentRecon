import sys
sys.path.append('../code')
import argparse
# import GPUtil

from training.idr_train import IDRTrainRunner

import torch
# import random
# import numpy as np
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=12000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--only_opt_silhouette', default=False, action="store_true", help='')
    parser.add_argument('--train_explicit_vertices_original', default=False, action="store_true", help='')
    parser.add_argument('--train_explicit_vertices_large_step', default=False, action="store_true", help='')
    parser.add_argument('--train_idr_sdf_single_mlp', default=False, action="store_true", help='')
    parser.add_argument('--train_mesh_sdf_single_mlp', default=False, action="store_true", help='')
    parser.add_argument('--train_displacements_single_mlp', default=False, action="store_true", help='')
    parser.add_argument('--train_displacements_multi_mlp', default=False, action="store_true", help='')
    parser.add_argument('--use_our_corr', default=False, action="store_true", help='')
    parser.add_argument('--use_DRT_corr', default=False, action="store_true", help='')
    parser.add_argument('--enable_remeshing', default=False, action="store_true", help='')
    parser.add_argument('--vsa_num', type=int, default=150, help='number of vsa clusters')

    opt = parser.parse_args()

    # if opt.gpu == "auto":
    #     deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
    #     gpu = deviceIDs[0]
    # else:
    #     gpu = opt.gpu

    trainrunner = IDRTrainRunner(conf=opt.conf,
                                 batch_size=opt.batch_size,
                                 nepochs=opt.nepoch,
                                 expname=opt.expname,
                                 gpu_index='ignore',
                                 exps_folder_name='/home/2TB/xjm/exps/',
                                 is_continue=opt.is_continue,
                                 vsa_num=opt.vsa_num,
                                 timestamp=opt.timestamp,
                                 checkpoint=opt.checkpoint,
                                 train_explicit_vertices_original=opt.train_explicit_vertices_original,
                                 train_explicit_vertices_large_step=opt.train_explicit_vertices_large_step,
                                 train_idr_sdf_single_mlp=opt.train_idr_sdf_single_mlp,
                                 train_mesh_sdf_single_mlp=opt.train_mesh_sdf_single_mlp,
                                 train_displacements_single_mlp=opt.train_displacements_single_mlp,
                                 train_displacements_multi_mlp=opt.train_displacements_multi_mlp,
                                 use_our_corr=opt.use_our_corr,
                                 use_DRT_corr=opt.use_DRT_corr,
                                 enable_remeshing=opt.enable_remeshing)

    trainrunner.run()
