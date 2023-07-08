import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import cv2
from baseline import ActorCritic
from utils import *
from cal_result import *
from scene_loader import THORDiscreteEnvironment as Environment
from PIL import Image


def train_(option, DATE_TIME, scene_scope, task_scope, device_num):
    
    torch.cuda.empty_cache()
    seed_torch(42)

    EPI = 2300
    # 最大步数看起来会影响其收敛速度，对开始阶段影响较大，设为2300的时候明显比3000容易收敛
    # EPI都设为2500，最大步数<=2500
    # 简单场景下差异不大，但是在复杂困难场景下CIVN具有更大的潜力和学习能力
    MAX_STEP = 3000 # fine-tune设置
    # MAX_STEP = 2000
    LR = 0.00026

    env = Environment({'scene_name': scene_scope, 'terminal_state_id': int(task_scope)})

    os.environ['CUDA_VISIBLE_DEVICES'] = device_num
    
    if option == 'baseline':
        model = ActorCritic()
        model_type = option
    elif option == 'CIVN':
        model = ActorCritic_()
        model_type = option

    model = model.cuda()

    optimizer = optim.RMSprop(model.parameters(), LR, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    model.train()
    torch.cuda.empty_cache()

    episode_length = 0
    cnt_epi = 0
    epi_num_step = []
    reward_epi = []
    sum_collide = []

    # f_id = open("[id] [%s]_%s.txt" % (model_type, scene_scope),'a')
    # f_id.truncate(0)
    # f_loop = open("[loop] [%s]_%s.txt" % (model_type, scene_scope),'a')
    # f_loop.truncate(0)
    # f_other = open("[other] [%s]_%s.txt" % (model_type, scene_scope),'a')
    # f_other.truncate(0)

    for i in range(EPI):
        state, state_target, _ = env.reset()
        state = torch.from_numpy(state) 
        state_target = torch.from_numpy(state_target)
        state = state.cuda()
        state_target = state_target.cuda()
        print('==================================  %s EPI %s =================================='% (model_type, cnt_epi))
        cnt_epi += 1
        values = []
        log_probs = []
        rewards = []
        entropies = []
        cnt = 0
        reward_sum = 0

        # 20220828
        prev_location = [0, 0, 0, 0]

        for step in range(MAX_STEP):
            print('--------------------- %s EPI %s: step %s ---------------------'% (model_type, cnt_epi, step + 1))
            episode_length += 1
            
            if torch.is_tensor(state_target) == False:
                state_target = torch.from_numpy(state_target)
            if torch.is_tensor(state) == False:
                state = torch.from_numpy(state)
            state = state.cuda()
            state_target = state_target.cuda()
            
            value, logit = model(state, state_target)  # 1
            
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, current_state_id, collide = env.step(action)  # 2
            # save_epi_pic(i, [20, 50], step, current_state_id, scene_scope)
            
            state = (torch.from_numpy(state)).cuda()
            
            reward = (torch.tensor(reward)).cuda()
            print('done:', done)
            print('current_state_id:',current_state_id)

            # 20220828
            if done:
                reward += 10.0
                reward = (torch.tensor(reward)).cuda()
        
            if collide:
                cnt += 1

            done = done or episode_length >= MAX_STEP 
            reward = max(min(reward, 1), -1)  
            reward_sum += reward
            
            if done:
                episode_length = 0
                state, state_target, _ = env.reset()
                sum_collide.append(cnt)

            if step == MAX_STEP - 1:
                sum_collide.append(cnt)
                
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        # f_id.write('%s' % cnt2)
        # f_id.write('\n')
        
        epi_num_step.append(step)
        reward_epi.append(reward_sum)
        R = torch.zeros(1, 1)
        R = (torch.tensor(R)).cuda()
        if not done:
            value, _ = model(state, state_target)
            R = value.detach()
        values.append(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        gae = (torch.tensor(gae)).cuda()
        for i in reversed(range(len(rewards))):
            R = 0.99 * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + 0.99 * values[i + 1] - values[i]
            gae = gae * 0.99 * 1 + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - 0.01 * entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, './model/[%s]%s_%s_%s_param.pth' % (model_type, DATE_TIME, scene_scope, task_scope))

    avr_len, avr_col, avr_reward = cal_train_result(EPI, MAX_STEP, epi_num_step, 
                                                    reward_epi, sum_collide, option, 
                                                    DATE_TIME, scene_scope)
    
    return avr_len, avr_col, avr_reward, scene_scope, EPI
    