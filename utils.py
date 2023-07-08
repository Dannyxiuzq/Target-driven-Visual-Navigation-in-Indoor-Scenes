import random
import os
import torch
import h5py
import numpy as np
import cv2


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def random_run(prob):
    ls = []
    for i in range(prob):
        ls.append(1)
    for x in range(100 - prob):
        ls.append(0)
    pick = random.choice(ls)
    if pick == 1:
        run = True
    else:
        run = False
    return run


def cal_avr(EPI, ls):
    total = 0
    for i in range(EPI):
        total = total + ls[i]
        ls_avr.append(total / (i+1))
    return ls_avr

def save_epi_pic(i, epi_num, step, current_state_id, scene_scope):
    h5_file = h5py.File('data/%s.h5'%(scene_scope), 'r')
    obs = h5_file['observation'][current_state_id]
    for num in epi_num:
        if i == num:
            if not os.path.exists('./%s_%s/'%(scene_scope, num)):
                os.makedirs('./%s_%s/'%(scene_scope, num))
            cv2.imwrite('./%s_%s/%s_%s.jpg'%(scene_scope, num, step, current_state_id), obs)
    # if current_state_id == 384:
    #     cv2.imwrite('./bedroom_data/bedroom_384.jpg', obs)
    # elif current_state_id == 322:
    #     cv2.imwrite('bedroom_326.jpg', obs)
    # elif current_state_id == 295:
    #     cv2.imwrite('bedroom_295.jpg', obs)
    # elif current_state_id == 327:
    #     cv2.imwrite('bedroom_327.jpg', obs)
    # elif current_state_id == 7:
    #     cv2.imwrite('bedroom_7.jpg', obs)
    # elif current_state_id == 80:
    #     cv2.imwrite('bedroom_80.jpg', obs)


def vis_save_epi_pic(option, i, epi_num, step, current_state_id, scene_scope, A_i, A_c):
    # print(A_i.shape, A_c.shape)
    # to_save = [348, 162, 280, 154, 331, 344]
    to_save = [348, 162]
    h5_file = h5py.File('data/%s.h5'%(scene_scope), 'r')
    obs = h5_file['observation'][current_state_id]
    for num in epi_num:
        if i == num:
            if not os.path.exists('./%s_%s/'%(scene_scope, num)):
                os.makedirs('./%s_%s/'%(scene_scope, num))
            if current_state_id in to_save:
                cv2.imwrite('./%s_%s/%s_%s_%s.jpg'%(scene_scope, num, option, step, current_state_id), obs)
                if option == 'CIVN':
                    torch.save(A_c, './%s_%s/CA_%s_%s.pth'%(scene_scope, num, step, current_state_id))
                    torch.save(A_i, './%s_%s/SA_%s_%s.pth'%(scene_scope, num, step, current_state_id))
                    



