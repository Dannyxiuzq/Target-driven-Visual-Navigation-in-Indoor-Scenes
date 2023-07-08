import matplotlib.pyplot as plt


def plot_length(avr_base, avr_CIVN, EPI, scene_scope, training, DATE_TIME):
    
    if training:
        situation = 'Train'
    else:
        situation = 'Test'
        
    fig1 = plt.figure('Figure1',figsize = (14, 6)).add_subplot(111)
    fig1.grid() 
    fig1.plot(list(range(1, EPI + 1)), avr_base, color='r', label='baseline')
    fig1.plot(list(range(1, EPI + 1)), avr_CIVN, color='g', label='CIVN')
    fig1.set_title('[%s] Average Episode Length' % (situation))
    fig1.set_xlabel('Episodes')
    fig1.set_ylabel('Average Episode Length')
    fig1.title.set_size(20)
    fig1.xaxis.label.set_size(16)
    fig1.yaxis.label.set_size(16)
    fig1.legend(fontsize=14)
    plt.savefig('[%s] len_%s_EPI%s_%s.png' % (situation, DATE_TIME, EPI, scene_scope), dpi=300)
    plt.clf()


def plot_reward(reward_base, reward_CIVN, EPI, scene_scope, training, DATE_TIME):
    
    if training == True:
        situation = 'Train'
    else:
        situation = 'Test'
        
    fig2 = plt.figure('Figure2',figsize = (14, 6)).add_subplot(111)
    fig2.grid() 
    fig2.plot(list(range(1,EPI + 1)), reward_base, color='r', label='baseline')
    fig2.plot(list(range(1,EPI + 1)), reward_CIVN, color='g', label='CIVN')
    fig2.set_title('[%s] Average Episode Reward' % (situation))
    fig2.set_xlabel('Episodes')
    fig2.set_ylabel('Average Episode Reward')
    fig2.title.set_size(20)
    fig2.xaxis.label.set_size(16)
    fig2.yaxis.label.set_size(16)
    fig2.legend(fontsize=14)
    plt.savefig('[%s] reward_%s_EPI%s_%s.png' % (situation, DATE_TIME, EPI, scene_scope), dpi=300)
    plt.clf()


def plot_collide(collide_base, collide_CIVN, EPI, scene_scope, DATE_TIME):
    
    fig3 = plt.figure('Figure3',figsize = (14, 6)).add_subplot(111)
    fig3.grid() 
    fig3.plot(list(range(1, EPI + 1)), collide_base, color='r', label='baseline')
    fig3.plot(list(range(1, EPI + 1)), collide_CIVN, color='g', label='CIVN')
    fig3.set_title('[Test] Average Episode Collision')
    fig3.set_xlabel('Episodes')
    fig3.set_ylabel('Average Episode Collision')
    fig3.title.set_size(20)
    fig3.xaxis.label.set_size(16)
    fig3.yaxis.label.set_size(16)
    fig3.legend(fontsize=14)
    plt.savefig('[Test] collide_%s_EPI%s_%s.png' % (DATE_TIME, EPI, scene_scope), dpi=300)
    plt.clf()


def save_rate(scene_scope_t, DATE_TIME_t, rate_base_t, rate_CIVN_t, SPL_base_t, SPL_CIVN_t):
    
    SR_base_t = '%.2f' % (100 * rate_base_t)
    SR_CIVN_t = '%.2f' % (100 * rate_CIVN_t)
    SPL_base_t = '%.2f' % (100 * SPL_base_t)
    SPL_CIVN_t = '%.2f' % (100 * SPL_CIVN_t)
    
    f = open("[Result] %s_%s.txt"%(scene_scope_t, DATE_TIME_t),'a')
    f.truncate(0)
    f.write('SR_base: %s   SPL_base: %s'%(SR_base_t, SPL_base_t)) 
    f.write('\n')
    f.write('SR_CIVN: %s   SPL_CIVN: %s'%(SR_CIVN_t, SPL_CIVN_t)) 