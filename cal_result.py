def cal_train_result(EPI, MAX_STEP, epi_num_step, reward_epi, sum_collide, option, DATE_TIME, scene_scope):
    
    print('****************************** [Train] Calculate Results ******************************')
    avr_len = []
    avr_col = []
    avr_reward = []
    cnt_fail = 0
    total1, total2, total3 = 0, 0, 0
    
    for i in range(EPI):
        if epi_num_step[i] == (MAX_STEP-1):
            cnt_fail += 1
        for ele in range(i, i+1):
            total1 = total1 + epi_num_step[ele]
        avr_len.append(total1 / (i+1))

    for i in range(EPI):
        for ele in range(i, i+1):
            total3 = total3 + sum_collide[ele]
        avr_col.append(total3 / (i+1))

    for i in range(EPI):
        for ele in range(i, i+1):
            total2 = total2 + reward_epi[ele]
        avr_reward.append(total2 / (i+1))

    for i in range(len(avr_reward)):
        avr_reward[i] = (avr_reward[i]).cpu()
    
    rate = 1 - cnt_fail/EPI
    print('Training successful rate: {}%'.format('%.2f' % (100*rate)))

    # 记录训练过程中的数据
    f = open("[Train] avr_%s_%s_%s.txt"%(option, DATE_TIME, scene_scope),'a')
    f.truncate(0)
    for i in range(len(avr_len)):
        f.write('%s'%(avr_len[i])) 
        f.write('\n')
        
    f = open("[Train] col_%s_%s_%s.txt"%(option, DATE_TIME, scene_scope),'a')
    f.truncate(0)
    for i in range(len(avr_col)):
        f.write('%s'%(avr_col[i])) 
        f.write('\n')
        
    return avr_len, avr_col, avr_reward
    

def cal_test_result(EPI, MAX_STEP, model_type, DATE_TIME, scene_scope, task_train_scope,task_test_scope,episode_length_ls, reward_sum_ls, collide_sum_ls, min_dist_ls):
    
    print('****************************** [Test] Calculate Results ******************************')
    f = open("[Test] [%s] %s_%s_%s_%s_rate.txt"%(model_type, DATE_TIME, scene_scope,task_train_scope,task_test_scope),'a')
    mean_epi_len = sum(episode_length_ls)/EPI
    mean_epi_reward = sum(reward_sum_ls)/EPI
    mean_epi_col = sum(collide_sum_ls)/EPI
    f.truncate(0)
    f.write('mean_epi_len: %s\nmean_epi_reward: %s\nmean_epi_col: %s\n'%(mean_epi_len, mean_epi_reward, mean_epi_col))

    avr_len = []
    avr_col = []
    avr_reward = []
    cnt_fail = 0
    total1, total3, total2 = 0, 0, 0
    
    for i in range(EPI):
        if episode_length_ls[i] == MAX_STEP:
            cnt_fail += 1
        for ele in range(i, i+1):
            total1 = total1 + episode_length_ls[ele]
        avr_len.append(total1 / (i+1))

    for i in range(EPI):
        for ele in range(i, i+1):
            total3 = total3 + collide_sum_ls[ele]
        avr_col.append(total3 / (i+1))

    for i in range(EPI):
        for ele in range(i, i+1):
            total2 = total2 + reward_sum_ls[ele]
        avr_reward.append(total2 / (i+1))

    rate = 1 - cnt_fail/EPI
    SPL = 0
    for i in range(EPI):
        if episode_length_ls[i] != MAX_STEP:
            SPL += min_dist_ls[i] / episode_length_ls[i]
    SPL = SPL / EPI
    
    print('Test successful rate: {}%'.format('%.2f' % (100*rate)))
    print('Test SPL: {}%'.format('%.2f' % (100*SPL)))
    f.write('Test successful rate: {}%\n'.format('%.2f' % (100*rate)))
    f.write('Test SPL: {}%'.format('%.2f' % (100*SPL)))
    
    return avr_len, avr_col, avr_reward, rate, SPL