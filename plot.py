from congestion_games import *
import matplotlib.pyplot as plt
import itertools
import numpy as np
import copy
import statistics
import seaborn as sns; sns.set()
from scipy import optimize 
from time import process_time

def projection_simplex_sort(v, z=1):
	# Courtesy: EdwardRaff/projection_simplex.py
    if v.sum() == z and np.alltrue(v >= 0):
        return v
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

# Define the states and some necessary info
N = 8 #number of agents
harm = - 100 * N # pentalty for being in bad state

safe_state = CongGame(N,1,[[1,0],[2,0],[4,0],[6,0]])
bad_state = CongGame(N,1,[[1,-100],[2,-100],[4,-100],[6,-100]])
state_dic = {0: safe_state, 1: bad_state}

M = safe_state.num_actions 
D = safe_state.m #number facilities
S = 2

# Dictionary to store the action profiles and rewards to
selected_profiles = {}

# Dictionary associating each action (value) to an integer (key)
act_dic = {}
counter = 0
for act in safe_state.actions:
	act_dic[counter] = act 
	counter += 1

RHO = 0.9

def get_next_state(state, actions):
    acts_from_ints = [act_dic[i] for i in actions]
    density = state_dic[state].get_counts(acts_from_ints)
    max_density = max(density)

    if state == 0 and max_density > N/2 or state == 1 and max_density > N/4:
      # if state == 0 and max_density > N/2 and np.random.uniform() > 0.2 or state == 1 and max_density > N/4 and np.random.uniform() > 0.1:
        return 1
    return 0

def pick_action(prob_dist):
    # np.random.choice(range(len(prob_dist)), 1, p = prob_dist)[0]
    acts = [i for i in range(len(prob_dist))]
    action = np.random.choice(acts, 1, p = prob_dist)
    return action[0]

def visit_dist(state, policy, gamma, T,samples):
    # This is the unnormalized visitation distribution. Since we take finite trajectories, the normalization constant is (1-gamma**T)/(1-gamma).
    visit_states = {st: np.zeros(T) for st in range(S)}        
    for i in range(samples):
        curr_state = state
        for t in range(T):
            visit_states[curr_state][t] += 1
            actions = [pick_action(policy[curr_state, i]) for i in range(N)]
            curr_state = get_next_state(curr_state, actions)
    dist = [np.dot(v/samples,gamma**np.arange(T)) for (k,v) in visit_states.items()]
    return dist 

def value_function(policy, gamma, T,samples):
    value_fun = {(s,i):0 for s in range(S) for i in range(N)}
    for k in range(samples):
        for state in range(S):
            curr_state = state
            for t in range(T):
                actions = [pick_action(policy[curr_state, i]) for i in range(N)]
                q = tuple(actions+[curr_state])
                rewards = selected_profiles.setdefault(q,get_reward(state_dic[curr_state], [act_dic[i] for i in actions]))                  
                for i in range(N):
                    value_fun[state,i] += (gamma**t)*rewards[i]
                curr_state = get_next_state(curr_state, actions)
    value_fun.update((x,v/samples) for (x,v) in value_fun.items())
    return value_fun

def Q_function(agent, state, action, policy, gamma, value_fun, samples):
    tot_reward = 0
    for i in range(samples):
        actions = [pick_action(policy[state, i]) for i in range(N)]
        actions[agent] = action
        q = tuple(actions+[state])
        rewards = selected_profiles.setdefault(q,get_reward(state_dic[state], [act_dic[i] for i in actions]))
        tot_reward += rewards[agent] + gamma*value_fun[get_next_state(state, actions), agent]
    return (tot_reward / samples)

def policy_accuracy(policy_pi, policy_star):
    total_dif = N * [0]
    for agent in range(N):
        for state in range(S):
            total_dif[agent] += np.sum(np.abs((policy_pi[state, agent] - policy_star[state, agent])))
	  # total_dif[agent] += np.sqrt(np.sum((policy_pi[state, agent] - policy_star[state, agent])**2))
    return np.sum(total_dif) / N

def policy_gradient(mu, max_iters, gamma, eta, T, samples):

    policy = {(s,i): [1/M]*M for s in range(S) for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]

    for t in range(max_iters):

        #print(t)

        b_dist = M * [0]
        for st in range(S):
            a_dist = visit_dist(st, policy, gamma, T, samples)

            b_dist[st] = np.dot(a_dist, mu)
            
        grads = np.zeros((N, S, M))
        value_fun = value_function(policy, gamma, T, samples)
	
        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    grads[agent, st, act] = b_dist[st] * Q_function(agent, st, act, policy, gamma, value_fun, samples)

        for agent in range(N):
            for st in range(S):
                policy[st, agent] = projection_simplex_sort(np.add(policy[st, agent], eta * grads[agent,st]), z=1)
        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(policy_hist[t], policy_hist[t-1]) < 10e-16:
      # if policy_accuracy(policy_hist[t+1], policy_hist[t]) < 10e-16: (it makes a difference, not when t=0 but from t=1 onwards.)
            return policy_hist

    return policy_hist


def FW(kappa, rho, eta, max_iters, gamma, T, samples, mu = 0.001):

    policy = {(s,i): np.asarray([1/M]*M) for s in range(S) for i in range(N)}
    policy_hat = {(s,i): np.asarray([1/M]*M) for s in range(S) for i in range(N)}
    explore = {(s,i): np.asarray([1/M]*M) for s in range(S) for i in range(N)}
    policy_hist = [copy.deepcopy(policy)]
    d_t = np.zeros((N, S, M))
    a_eq = np.ones_like(policy[0, 0]).T
    a_eq = np.array([a_eq, a_eq])
    for t in range(max_iters):
        b_dist = M * [0]
        for st in range(S):
            a_dist = visit_dist(st, policy, gamma, T, samples)
            b_dist[st] = np.dot(a_dist, kappa)
        
        rho_t = rho/((t+1) ** (3/5))
        eta_t = eta
            
        grads = np.zeros((N, S, M))
        value_fun = value_function(policy, gamma, T, samples)
	
        for agent in range(N):
            for st in range(S):
                for act in range(M):
                    #grads[agent, st, act] =  Q_function(agent, st, act, policy, gamma, value_fun, samples)
                    grads[agent, st, act] = b_dist[st] * Q_function(agent, st, act, policy, gamma, value_fun, samples)
                    d_t[agent, st, act] = (1-rho_t) * d_t[agent, st, act] + rho_t * grads[agent, st, act]
                    #print(d_t[agent, st, :], policy[st, agent])
        for agent in range(N):
            for st in range(S):
                #print(grads[agent, st, :],policy_hat[st, agent])
                #policy_hat[st, agent] = optimize.linprog(-grads[agent, st, :], A_eq = a_eq, b_eq=[1,1], options = {"disp":False, "maxiter":2000,"bland":True} ).x
                policy_hat[st, agent] = optimize.linprog(-d_t[agent, st, :], A_eq = a_eq, b_eq=[1,1], options = {"disp":False, "maxiter":10} ).x
                #print(policy[st, agent], type(policy[st, agent]))
                policy[st, agent] = (1-eta_t) * policy[st, agent] + eta_t * policy_hat[st, agent]
                policy[st,agent] = (1-mu) * policy[st, agent] + mu * explore[st, agent]
                #print(agent, st, grads[agent, st, :], policy_hat[st, agent], policy[st, agent])
        densities = np.zeros((S,M))
        for i in range(N):
            for s in range(S):
                densities[s] += policy[s,i]
        #print(densities)
                #policy[st, agent] = projection_simplex_sort(policy[st,agent], z=1)
        policy_hist.append(copy.deepcopy(policy))

        if policy_accuracy(policy_hist[t], policy_hist[t-1]) < 10e-16:
      # if policy_accuracy(policy_hist[t+1], policy_hist[t]) < 10e-16: (it makes a difference, not when t=0 but from t=1 onwards.)
            return policy_hist

    return policy_hist


def get_accuracies(policy_hist):
    fin = policy_hist[-1]
    accuracies = []
    for i in range(len(policy_hist)):
        this_acc = policy_accuracy(policy_hist[i], fin)
        accuracies.append(this_acc)
    return accuracies


def full_experiment(runs,iters,eta,T,samples):
    densities_sgd = np.zeros((S,M))
    densities_fw = np.zeros((S,M))

    raw_accuracies_sgd = []
    raw_accuracies_fw = []
    myp_start = process_time()
    for k in range(runs):
        #kappa, rho, eta, max_iters, gamma, T, samples
        #policy_hist = FW([0.5, 0.5], RHO, eta,iters,0.99,T,samples)
        # policy_hist_sgd = policy_gradient([0.5, 0.5],iters,0.99,eta,T,samples)
        # raw_accuracies_sgd.append(get_accuracies(policy_hist_sgd))
        # converged_policy_sgd = policy_hist_sgd[-1]

        policy_hist_fw = FW([0.5, 0.5], RHO, 0.1, iters,0.99,T,samples)
        raw_accuracies_fw.append(get_accuracies(policy_hist_fw))
        converged_policy_fw = policy_hist_fw[-1]

    #     for i in range(N):
    #         for s in range(S):
    #             densities_sgd[s] += converged_policy_sgd[s,i]
    #             #densities_fw[s] += converged_policy_fw[s,i]

    # densities_sgd = densities_sgd / runs
    # densities_fw = densities_fw / runs

    myp_end = process_time()
    elapsed_time = myp_end - myp_start
    print(elapsed_time)


    #densities = densities / runs

    # max_length = 0
    # for j in range(runs):
    #     max_length = max(max_length, len(raw_accuracies[j]))

    # plot_accuracies = np.zeros((runs, max_length))

    # for j in range(runs):
    #     j_len = len(raw_accuracies[j])
    #     plot_accuracies[j][:j_len] = raw_accuracies[j]
    
#     plot_accuracies_sgd = np.array(list(itertools.zip_longest(*raw_accuracies_sgd, fillvalue=np.nan))).T
#     plot_accuracies_fw = np.array(list(itertools.zip_longest(*raw_accuracies_fw, fillvalue=np.nan))).T
#     # = sns.color_palette("husl", 3)
    

#     # = plt.figure(figsize=(6,4))
#     #for i in range(len(plot_accuracies)):
#     #print(plot_accuracies.shape)
#     mean_sgd = np.mean(plot_accuracies_sgd, axis=0)
#     mean_fw = np.mean(plot_accuracies_fw, axis=0)
#     plt.plot(mean_sgd[:150],label='SGD', color='g')
#     plt.fill_between(range(150), np.subtract(mean_sgd[:150],np.std(plot_accuracies_sgd, axis=0)[:150]), np.add(mean_sgd[:150],np.std(plot_accuracies_sgd, axis=0)[:150]), alpha=0.1, facecolor='g')
#     plt.plot(mean_fw[:150],label='FW (Ours)', color = 'b')
#     plt.fill_between(range(150), np.subtract(mean_fw[:150],np.std(plot_accuracies_fw, axis=0)[:150]), np.add(mean_fw[:150],np.std(plot_accuracies_fw, axis=0)[:150]), alpha=0.1, facecolor='b')
#     plt.grid(linewidth=0.6)
#     plt.gca().set(xlabel='Iterations',ylabel='L1-distance')
#     #plt.show()
#     plt.legend()
#     plt.savefig('individual_runs_n{}.png'.format(N),bbox_inches='tight')
#     plt.close()
    

#     fig3, ax = plt.subplots()
#     index = np.arange(D)
#     bar_width = 0.0875
#     opacity = 1

#     #print(len(index))
#     #print(len(densities[0]))
#     rects1 = plt.bar(index, densities_fw[0], bar_width,
#     alpha= .7 * opacity,
#     color='b',
#     label='Safe state (Ours)')
#     rects1 = plt.bar(index + bar_width, densities_sgd[0], bar_width,
#     alpha= .7 * opacity,
#     color='g',
#     label='Safe state (SGD)')

#     rects2 = plt.bar(index + bar_width * 2, densities_fw[1], bar_width,
#     alpha= opacity,
#     color='r',
#     label='Distancing state (Ours)')
#     rects2 = plt.bar(index + bar_width * 3, densities_sgd[1], bar_width,
#     alpha= opacity,
#     color='brown',
#     label='Distancing state (SGD)')

#     plt.gca().set(xlabel='Facility',ylabel='Average number of agents')
#     plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
#     plt.legend()
#     plt.savefig('facilities_n_fw{}.png'.format(N),bbox_inches='tight')
#     plt.close()
#     # piters_sgd= list(range(plot_accuracies_sgd.shape[1]))
#     # plot_accuracies_sgd = np.nan_to_num(plot_accuracies_sgd)
#     # pmean_sgd = list(map(statistics.mean, zip(*plot_accuracies_sgd)))
#     # pstdv_sgd = list(map(statistics.stdev, zip(*plot_accuracies_sgd)))

#     # piters_fw= list(range(plot_accuracies_fw.shape[1]))
#     # plot_accuracies_fw = np.nan_to_num(plot_accuracies_fw)
#     # pmean_fw = list(map(statistics.mean, zip(*plot_accuracies_fw)))
#     # pstdv_fw = list(map(statistics.stdev, zip(*plot_accuracies_fw)))
    
#     #fig1 = plt.figure(figsize=(6,4))
#     #clrs = sns.color_palette("husl", 3)
#     #ax = sns.lineplot(piters_sgd, pmean_sgd, color = clrs[0],label= 'Mean L1-accuracy (SGD))')
#     #ax.fill_between(piters_sgd, np.subtract(pmean_sgd,pstdv_sgd = list(map(statistics.stdev, zip(*plot_accuracies_sgd)))), np.add(pmean_sgd,pstdv_sgd), alpha=0.3, facecolor=clrs[0],label="1-standard deviation")
#     #ax.legend()
# #    plt.plot(piters_sgd, pmean_sgd, color = clrs[0],label= 'Mean L1-accuracy (SGD))')
#     #ax = sns.lineplot(piters_fw, pmean_fw, color = clrs[1],label= 'Mean L1-accuracy (FW (Ours)))')
#     #ax.fill_between(piters_fw, np.subtract(pmean_fw,pstdv_fw), np.add(pmean_fw,pstdv_fw), alpha=0.3, facecolor=clrs[1],label="1-standard deviation")

#     #plt.grid(linewidth=0.6)
#     #plt.gca().set(xlabel='Iterations',ylabel='L1-accuracy', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N, runs,eta))
#     #plt.show()
#    #plt.savefig('avg_runs_n_fw{}.png'.format(N),bbox_inches='tight')
#     #plt.close()
    
#     #print(densities)

#     # fig3, ax = plt.subplots()
#     # index = np.arange(D)
#     # bar_width = 0.35
#     # opacity = 1

#     # #print(len(index))
#     # #print(len(densities[0]))
#     # rects1 = plt.bar(index, densities[0], bar_width,
#     # alpha= .7 * opacity,
#     # color='b',
#     # label='Safe state')

#     # rects2 = plt.bar(index + bar_width, densities[1], bar_width,
#     # alpha= opacity,
#     # color='r',
#     # label='Distancing state')

#     # plt.gca().set(xlabel='Facility',ylabel='Average number of agents', title='Policy Gradient: agents = {}, runs = {}, $\eta$ = {}'.format(N,runs,eta))
#     # plt.xticks(index + bar_width/2, ('A', 'B', 'C', 'D'))
#     # plt.legend()
#     # fig3.savefig('facilities_n{}.png'.format(N),bbox_inches='tight')
#     plt.close()
    #plt.savefig('results_n{}.png'.format(N),bbox_inches='tight')


#full_experiment(10,1000,0.0001,20,10)
full_experiment(1,200,0.0001,20,10)

