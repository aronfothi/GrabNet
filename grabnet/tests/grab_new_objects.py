# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import numpy.matlib
import torch

import os, glob
import argparse

import pickle


import mano
import smplx

from psbody.mesh import MeshViewers, Mesh
from grabnet.tools.meshviewer import Mesh as M
from grabnet.tools.vis_tools import points_to_spheres
from grabnet.tools.utils import euler
from grabnet.tools.cfg_parser import Config
from grabnet.tests.tester import Tester

from bps_torch.bps import bps_torch

from psbody.mesh.colors import name_to_rgb
from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu

from copy import copy

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}

def DotDict(in_dict):

    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def get_rigid_transform(A, B):
    cenA = np.mean(A, 0)  # 3
    cenB = np.mean(B, 0)  # 3
    N = A.shape[0]  # 24
    H = np.dot((B - np.matlib.repmat(cenB, N, 1)).transpose(),
               (A - np.matlib.repmat(cenA, N, 1)))

    [U, S, V] = np.linalg.svd(H)
    R = np.dot(U, V)  # matlab returns the transpose: .transpose()
    if np.linalg.det(R) < 0:
        U[:, 2] = -U[:, 2]
        R = np.dot(U, V.transpose())
    t = np.dot(-R, cenA.transpose()) + cenB.transpose()
    return R, t

def get_rigid_transform_posed_mano(posed_model, mano_model):
    rigid_transform = get_rigid_transform(mano_model.J_transformed[1:].r,
                                          posed_model.J_transformed[37:].r)

    # Concatenate rotation and translation
    rigid_transform = np.asarray(
        np.concatenate((rigid_transform[0], np.matrix(rigid_transform[1]).T),
                       axis=1))
    rigid_transform = np.concatenate((rigid_transform, np.array([[0, 0, 0,
                                                                  1]])))
    return rigid_transform


def vis_results(dorig, coarse_net, refine_net, rh_model , save=False, save_dir = None):

    with torch.no_grad():        
        imw, imh = 1920, 780
        cols = len(dorig['bps_object'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mvs = MeshViewers(window_width=imw, window_height=imh, shape=[1, cols], keepalive=True)

        drec_cnet = coarse_net.sample_poses(dorig['bps_object'])
        verts_rh_gen_cnet = rh_model(**drec_cnet).vertices

        _, h2o, _ = point2point_signed(verts_rh_gen_cnet, dorig['verts_object'].to(device))
        
        drec_cnet['trans_rhand_f'] = drec_cnet['transl']
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
        drec_cnet['verts_object'] = dorig['verts_object'].to(device)
        drec_cnet['h2o_dist']= h2o.abs()
        
        drec_rnet = refine_net(**drec_cnet)
        verts_rh_gen_rnet = rh_model(**drec_rnet).vertices

        ####################################################################
        grab_path = '/home/fothar/GRAB/grab/'
        all_seqs = glob.glob(grab_path + '/*/*eat*.npz')
        sequence = all_seqs[10]
        
        seq_data = parse_npz(sequence)
        n_comps = seq_data['n_comps']
        gender = seq_data['gender']

        T = seq_data.n_frames

        sbj_mesh = os.path.join(grab_path, '..', seq_data.body.vtemp)
        sbj_vtemp = np.array(Mesh(filename=sbj_mesh).v)

        model_folder = '/home/fothar/GRAB/models/'
        corr_fname = '/home/fothar/correspondences/MANO_SMPLX_vertex_ids.pkl'

        
        smplx_model = smplx.create(model_path=model_folder,
                             ext='npz',
                             model_type='smplx',
                             gender=gender,
                             num_pca_comps=n_comps,
                             v_template=sbj_vtemp,
                             batch_size=T)
        
        hand_color=(0.3, 0.3, 0.6)
        hand='right'

        with open(corr_fname, 'rb') as f:
            idxs_data = pickle.load(f)
            if hand == 'both':
                hand_idxs = np.concatenate(
                    [idxs_data['left_hand'], idxs_data['right_hand']]
                )
            else:
                hand_idxs = idxs_data[f'{hand}_hand']

        
        #betas = torch.zeros([1, 10], dtype=torch.float32)
        #expression = torch.zeros([1, 10], dtype=torch.float32)

        #smplx_output = smplx_model(betas=betas, expression=expression,
        #           return_verts=True)

        sbj_parms = params2torch(seq_data.body.params)
        smplx_output = smplx_model(**sbj_parms)
        #verts_sbj = to_cpu(smplx_output.vertices)

        skip_frame = 4
        for frame in range(0,T, skip_frame):

            for cId in range(0, len(dorig['bps_object'])):
                try:
                    from copy import deepcopy
                    meshes = deepcopy(dorig['mesh_object'])
                    obj_mesh = meshes[cId]
                except:
                    obj_mesh = points_to_spheres(to_cpu(dorig['verts_object'][cId]), radius=0.002, vc=name_to_rgb['green'])

                hand_mesh_gen_cnet = Mesh(v=to_cpu(verts_rh_gen_cnet[cId]), f=rh_model.faces, vc=name_to_rgb['pink'])
                hand_mesh_gen_rnet = Mesh(v=to_cpu(verts_rh_gen_rnet[cId]), f=rh_model.faces, vc=name_to_rgb['gray'])

                #################xxx
                smplx_vertices = smplx_output.vertices.detach().cpu().numpy()[frame].squeeze()
                smplx_joints = smplx_output.joints.detach().cpu().numpy().squeeze()

                #smplx_vertices[hand_idxs] = to_cpu(verts_rh_gen_rnet[cId])

                smplx_colors = np.ones_like(smplx_vertices) * [0.3, 0.3, 0.3]
                smplx_colors[hand_idxs] = hand_color

                # mvs[0][cId].set_static_meshes([hand_mesh_gen_cnet] + obj_mesh, blocking=True)
                smplx_mesh = Mesh(v=smplx_vertices, f=smplx_model.faces, vc=smplx_colors)


                ####################



                if 'rotmat' in dorig:
                    rotmat = dorig['rotmat'][cId].T
                    obj_mesh = obj_mesh.rotate_vertices(rotmat)
                    hand_mesh_gen_cnet.rotate_vertices(rotmat)
                    hand_mesh_gen_rnet.rotate_vertices(rotmat)
                    smplx_mesh.rotate_vertices(rotmat)

                hand_mesh_gen_cnet.reset_face_normals()
                hand_mesh_gen_rnet.reset_face_normals()
                smplx_mesh.reset_face_normals()

                if 'offset' in dorig:
                    offset = dorig['offset'][cId]
                    obj_mesh.v += offset
                    hand_mesh_gen_rnet.v += offset
                    smplx_mesh.v += [[ 1.0 -1.00 -1.000]]

                if 'full_rescale' in dorig:
                    full_rescale = dorig['full_rescale'][cId]
                    obj_mesh.v *= full_rescale
                    hand_mesh_gen_rnet.v *= full_rescale
                    smplx_mesh.v *= full_rescale

                R, t = get_rigid_transform(
                    smplx_mesh.v[hand_idxs], hand_mesh_gen_rnet.v)

                smplx_mesh.rotate_vertices(R)
                smplx_mesh.v += t
                

                smplx_mesh.v[hand_idxs] = hand_mesh_gen_rnet.v

                
                mvs[0][cId].set_static_meshes([obj_mesh, smplx_mesh], blocking=True)
                #mvs[0][cId].set_static_meshes([smplx_mesh], blocking=True)

                if save:
                    save_path = os.path.join(save_dir, str(cId))
                    makepath(save_path)
                    hand_mesh_gen_rnet.write_obj(filename=save_path + '/rh_mesh_gen_%d.obj' % cId)
                    obj_mesh.write_obj(filename=save_path + '/obj_mesh_%d.obj' % cId)




def grab_new_objs(grabnet, objs_path, rot=True, n_samples=3, scale=1.):
    
    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()

    rh_model = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(grabnet.device)

    grabnet.refine_net.rhm_train = rh_model

    grabnet.logger(f'################# \n'
                   f'Colors Guide:'
                   f'                   \n'
                   f'Gray  --->  GrabNet generated grasp\n')

    bps = bps_torch(custom_basis = grabnet.bps)

    if not isinstance(objs_path, list):
        objs_path = [objs_path]
        
    for new_obj in objs_path:
        
        rand_rotdeg = np.random.random([n_samples, 3]) * np.array([360, 360, 360])

        rand_rotmat = euler(rand_rotdeg)
        dorig = {'bps_object': [],
                 'verts_object': [],
                 'mesh_object': [],
                 'rotmat':[],
                 'full_rescale':[],
                 'offset':[]}

        for samples in range(n_samples):

            verts_obj, mesh_obj, rotmat, full_rescale, offset = load_obj_verts(new_obj, rand_rotmat[samples], rndrotate=rot, scale=scale)
            
            bps_object = bps.encode(verts_obj, feature_type='dists')['dists']

            dorig['bps_object'].append(bps_object.to(grabnet.device))
            dorig['verts_object'].append(torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0))
            dorig['mesh_object'].append(mesh_obj)
            dorig['rotmat'].append(rotmat)
            dorig['full_rescale'].append(full_rescale)
            dorig['offset'].append(offset)
            obj_name = os.path.basename(new_obj)

        dorig['bps_object'] = torch.cat(dorig['bps_object'])
        dorig['verts_object'] = torch.cat(dorig['verts_object'])

        save_dir = os.path.join(grabnet.cfg.work_dir, 'grab_new_objects')
        grabnet.logger(f'#################\n'
                              f'                   \n'
                              f'Showing results for the {obj_name.upper()}'
                              f'                      \n')

        vis_results(dorig=dorig,
                    coarse_net=grabnet.coarse_net,
                    refine_net=grabnet.refine_net,
                    rh_model=rh_model,
                    save=True,
                    save_dir=save_dir
                    )

def load_obj_verts(mesh_path, rand_rotmat, rndrotate=True, scale=1., n_sample_verts=10000):

    np.random.seed(100)
    obj_mesh = Mesh(filename=mesh_path, vscale=scale)

    obj_mesh.reset_normals()
    #obj_mesh.vc = obj_mesh.colors_like('green')

    ## center and scale the object
    max_length = np.linalg.norm(obj_mesh.v, axis=1).max()

    full_rescale = 1.
    if  max_length > .3:
        re_scale = max_length/.08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh.v = obj_mesh.v/re_scale

        full_rescale *= re_scale

    object_fullpts = obj_mesh.v
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)

    offset = ( maximum + minimum) / 2
    verts_obj = object_fullpts - offset
    obj_mesh.v = verts_obj

    if rndrotate:
        obj_mesh.rotate_vertices(rand_rotmat)
    else:
        rand_rotmat = np.eye(3)

    while (obj_mesh.v.shape[0] < n_sample_verts):        
        mesh = M(vertices=obj_mesh.v, faces = obj_mesh.f)
        mesh = mesh.subdivide()
        obj_mesh = Mesh(v=mesh.vertices, f = mesh.faces, vc=name_to_rgb['green'])

    verts_obj = obj_mesh.v
    verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    verts_sampled = verts_obj[verts_sample_id]

    return verts_sampled, obj_mesh, rand_rotmat, full_rescale, offset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    parser.add_argument('--obj-path', required = True, type=str,
                        help='The path to the 3D object Mesh or Pointcloud')

    parser.add_argument('--rhm-path', required = True, type=str,
                        help='The path to the folder containing MANO_RIHGT model')

    parser.add_argument('--config-path', default= None, type=str,
                        help='The path to the confguration of the trained GrabNet model')

    args = parser.parse_args()

    cfg_path = args.config_path
    obj_path = args.obj_path
    rhm_path = args.rhm_path

    cwd = os.getcwd()
    work_dir = cwd + '/logs'

    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    bps_dir   = 'grabnet/configs/bps.npz'


    if cfg_path is None:
        cfg_path = 'grabnet/configs/grabnet_cfg.yaml'


    config = {
        'work_dir': work_dir,
        'best_cnet': best_cnet,
        'best_rnet': best_rnet,
        'bps_dir': bps_dir,
        'rhm_path': rhm_path
    }

    cfg = Config(default_cfg_path=cfg_path, **config)

    grabnet = Tester(cfg=cfg)
    grab_new_objs(grabnet,obj_path, rot=True, n_samples=3)
