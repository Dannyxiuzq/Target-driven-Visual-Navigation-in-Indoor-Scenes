import argparse
import torch
import matplotlib.pyplot as plt
from train import train_
from test import test_
from visualization import *


if __name__ == '__main__':
    
    '''
    python main.py --DATE_TIME '0907_1539' --train_scene_scope 'bathroom_02' --train_task_scope '26' --test_scene_scope 'bathroom_02' --test_task_scope '26' --device_num 0 --vis False
    python main.py --DATE_TIME '0916_0026' --train_scene_scope 'bedroom_04' --train_task_scope '264' --test_scene_scope 'bedroom_04' --test_task_scope '264' --device_num 1 --vis False
    '''

    parser = argparse.ArgumentParser(description='Causal Intervention Visual Navigation.')
    parser.add_argument('--train', type=str, default=False, help='train_flag')
    parser.add_argument('--DATE_TIME', type=str, help='time that starts running.')
    parser.add_argument('--train_scene_scope', type=str, help='train_scene.')
    parser.add_argument('--train_task_scope', type=str, help='train_id.')
    parser.add_argument('--test_scene_scope', type=str, help='test_scene.')
    parser.add_argument('--test_task_scope', type=str, help='test_id.')
    parser.add_argument('--device_num', type=str, default='0', help='cuda number')
    parser.add_argument('--vis', type=bool, default=False, help='vis or not')
    opt = parser.parse_args()
    
    '''
    ls_scene_scope = ['bathroom_02', 'bedroom_04', 'kitchen_02', 'living_room_08']
    TASK_LIST = {
                    'bathroom_02'    : ['26', '37', '43', '53', '69'],
                    'bedroom_04'     : ['134', '264', '320', '384', '387'],
                    'kitchen_02'     : ['90', '136', '157', '207', '329'],
                    'living_room_08' : ['92', '135', '193', '228', '254']
    }
    '''
    
    # avr_CIVN, collide_CIVN, reward_CIVN, scene_scope, EPI = train_("CIVN",
    #                                                               opt.DATE_TIME,
    #                                                               opt.train_scene_scope,
    #                                                               opt.train_task_scope,
    #                                                               opt.device_num)   
    if opt.train == True:
        avr_base, collide_base, reward_base, scene_scope, EPI = train_("baseline", 
                                                                  opt.DATE_TIME,
                                                                  opt.train_scene_scope,
                                                                  opt.train_task_scope,
                                                                  opt.device_num)
    
    # len_CIVN_, col_CIVN_, r_CIVN_, scene_scope_, EPI_, rate_CIVN_, SPL_CIVN_ = test_("CIVN",
    #                                                                                 opt.DATE_TIME,
    #                                                                                 opt.test_scene_scope,
    #                                                                                 opt.test_task_scope,
    #                                                                                 opt.device_num)
    len_base_, col_base_, r_base_, scene_scope_, EPI_, rate_base_, SPL_base_ = test_("baseline",
                                                                                    opt.DATE_TIME,
                                                                                    opt.test_scene_scope,
                                                                                    opt.test_task_scope,
                                                                                    opt.device_num)
    
    rate_CIVN_, SPL_CIVN_ = 0.0, 0.0
    save_rate(scene_scope_, opt.DATE_TIME, rate_base_, rate_CIVN_, SPL_base_, SPL_CIVN_)
    
    # if opt.vis:
        # plot_length(avr_base, avr_CIVN, EPI, scene_scope, True, opt.DATE_TIME)
        # plot_reward(reward_base, reward_CIVN, EPI, scene_scope, True, opt.DATE_TIME)
        
        # plot_length(len_base_, len_CIVN_, EPI_, scene_scope_, False, opt.DATE_TIME)
        # plot_reward(r_base_, r_CIVN_, EPI_, scene_scope_, False, opt.DATE_TIME)
        # plot_collide(col_base_, col_CIVN_, EPI_, scene_scope_, opt.DATE_TIME)
        
    
   
