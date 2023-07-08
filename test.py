from collections import deque
import torch
import torch.nn.functional as F
from baseline import ActorCritic
from scene_loader import THORDiscreteEnvironment as Environment
from utils import *
from cal_result import *


def test_(option, DATE_TIME, scene_scope, task_scope, device_num,scene_scope2,task_scope2):# 2 means train
    task_scope = task_scope.replace("'", "")
    scene_scope = scene_scope.replace("'", "")
    DATE_TIME = DATE_TIME.replace("'", "")
    scene_scope2=scene_scope2.replace("'", "")
    task_scope2 = task_scope2.replace("'", "")
    EPI = 300
    if option == "baseline":
        MAX_STEP = 500  # fine-tune设置
    else:
        MAX_STEP = 500

    env = Environment({'scene_name': scene_scope, 'terminal_state_id': int(task_scope)})

    if option == 'baseline':
        model = ActorCritic()
        model_type = option
    elif option == 'CIVN':
        model = ActorCritic_()
        model_type = option
        
    model = model.cuda()
    checkpoint = torch.load('./model_save/[%s]%s_%s_param.pth' % (model_type, DATE_TIME, scene_scope2))

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    cnt_fail = 0
    reward_sum_ls = []
    episode_length_ls = []
    collide_sum_ls = []
    min_dist_ls = []

    for i in range(EPI):
        state, state_target, min_dist = env.reset()
        min_dist_ls.append(min_dist)
        state = torch.from_numpy(state)
        state_target = torch.from_numpy(state_target)
        state = state.cuda()
        state_target = state_target.cuda()
        print('================================== %s EPI %s =================================='% (model_type, i + 1))
        actions = deque(maxlen=100)
        episode_length = 0
        reward_sum = 0
        cnt_collide = 0
        for step in range(MAX_STEP):
            print('---------------------  %s EPI %s: step %s ---------------------'% (model_type, i + 1, step + 1))
            episode_length += 1

            with torch.no_grad():
                if torch.is_tensor(state_target) == False:
                    state_target = torch.from_numpy(state_target)
                if torch.is_tensor(state) == False:
                    state = torch.from_numpy(state)
                state = state.cuda()
                state_target = state_target.cuda()

                # A_i, A_c, value, logit = model(state, state_target)
                value, logit = model(state, state_target)
                # print(A_i.shape, A_c.shape)
            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1).detach()

            state, reward, done, current_state_id, collide = env.step(action[0, 0])
            # vis_save_epi_pic(option, i, [10, 20, 50], step, current_state_id, scene_scope, A_i, A_c)

            print('done:',done)
            print('current_state_id:', current_state_id)
            if collide == True:
                cnt_collide += 1

            done = done or episode_length >= 800
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            actions.append(action[0, 0])
            if actions.count(actions[0]) == actions.maxlen:
                done = True

            # if the agent reaches the target successfully
            if done:
                print("episode reward {}, episode length {}".format(reward_sum, episode_length))
                reward_sum_ls.append(reward_sum)
                episode_length_ls.append(episode_length)
                collide_sum_ls.append(cnt_collide)

                reward_sum = 0
                episode_length = 0
                actions.clear()
                state, state_target, _ = env.reset()
            if done:
                break
            
            # if the agent doesn't reach the target within 800 steps
            if step == (MAX_STEP - 1):
                cnt_fail += 1
                reward_sum_ls.append(reward_sum)
                episode_length_ls.append(episode_length)
                collide_sum_ls.append(cnt_collide)

            state = torch.from_numpy(state)
    
    avr_len, avr_col, avr_reward, rate, SPL = cal_test_result(EPI, MAX_STEP, model_type, 
                                                              DATE_TIME, scene_scope, task_scope2,task_scope,
                                                              episode_length_ls, reward_sum_ls, 
                                                              collide_sum_ls, min_dist_ls)

    return avr_len, avr_col, avr_reward, scene_scope, EPI, rate, SPL
