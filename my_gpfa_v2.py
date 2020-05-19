import scipy.io
import pandas as pd
from elephant.gpfa import GPFA
import numpy as np
import quantities as pq
import neo
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

## load spikeTime data as neo.spiketrains
with open('data/spikeTimes.pickle', 'rb') as filename:
    st = pickle.load(filename)
    
## load behavior data
behav = pd.read_csv('data/RT_cue_choice.csv',header=None, names=['RT','cue','choice'])
# add a movement onset column
behav['mvmt'] = behav['RT'] + 400

## load latent dynamics from elephant.gpfa with 4 dims
with open('data/gpfa_trajectories_4d.pickle', 'rb') as filename:
    trajectories_all = pickle.load(filename)

# number of trials and neurons
numTrials = len(st)
numNeurons = len(st[0])

# raster plot

fig = plt.figure(figsize=(10,6))
for i, spiketrain in enumerate(st[0]):
    plt.plot(spiketrain, np.ones_like(spiketrain)*i, ls='', marker='|')

plt.title('Raster plot of trial 1')
plt.xlabel('time (ms)')
plt.ylabel('Neuron')
# plt.savefig('images/gpfa_data_raster.png')
plt.show()

# specify fitting parameters
bin_size = 20 * pq.ms
latent_dimensionality = 4

gpfa = GPFA(bin_size=bin_size, x_dim=latent_dimensionality)

# apply GPFA
# trajectories_all = gpfa.fit_transform(st)

# function to make plots look cleaner
def transBackground(ax, dim):
    if dim==3:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.grid(False)
    if dim==2:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.grid(False)

# function to get index of times vector where movement onset occurs
def mvmtIdx(trial_idx):
    m = behav['mvmt'][trial_idx]
    m = np.floor(m)
    for i in range(20):
        if m % 20 == 0:
            break
        m += 1
    try:
        t_idx = np.where(times==m)[0][0]
    except:
        t_idx = len(times)
    return t_idx

# function that returns index of times vector, provided RT as index
def RT_idx(m):
    m = np.floor(m)
    for i in range(20):
        if m % 20 == 0:
            break
        m += 1
    try:
        t_idx = np.where(times==m)[0][0]
    except:
        t_idx = len(times)
    return t_idx

# generate a time vector compatible with neo.spiketrains
times = np.arange(len(trajectories_all[0][0])) * bin_size.rescale('ms')

# checkerboard onset idx
checkOnIdx = np.where(times==400)[0][0]

# generate random trial vector for plotting ax1
ct = 500 # number of trials to plot
possible = np.arange(1877)
np.random.shuffle(possible)
picks = possible[:ct]

# get average trajectories for left and right reaches
right_trials = behav[behav['choice']==2] # green
left_trials = behav[behav['choice']==1]   # red
right_trials_idx = list(right_trials.index)
left_trials_idx = list(left_trials.index)
r_traj = trajectories_all[right_trials_idx]
l_traj = trajectories_all[left_trials_idx]
r_avg_traj = np.mean(r_traj, axis=0)
l_avg_traj = np.mean(l_traj, axis=0)

# get times index for when reach happens for averaged trials
r_avg_mvmt_idx = RT_idx(np.mean(right_trials['mvmt']))
l_avg_mvmt_idx = RT_idx(np.mean(left_trials['mvmt']))

## plot a single trial's trajectories until movement onset
f = plt.figure(figsize=(10, 8))
ax2 = f.add_subplot(1, 1, 1)

trial_to_plot = 0
ax2.set_title(f'Trajectory for trial {trial_to_plot}')
ax2.set_xlabel('Time [ms]')
# get mvmt onset index
t_idx = mvmtIdx(trial_to_plot)
ax2.plot(times[:t_idx], trajectories_all[0][0][:t_idx], c='C0', label="Dim 1")
ax2.plot(times[:t_idx], trajectories_all[0][1][:t_idx], c='C1', label="Dim 2")
ax2.plot(times[:t_idx], trajectories_all[0][2][:t_idx], c='C2', label="Dim 3")
ax2.plot(times[:t_idx], trajectories_all[0][3][:t_idx], c='C3', label="Dim 4")
ax2.legend(loc='best')

plt.show()

## plot setup

f = plt.figure(figsize=(10, 8))
ax1 = f.add_subplot(1, 1, 1, projection='3d')

## plot GPFA latent dynamics by choice

ax1.set_title('Latent dynamics by choice')
ax1.set_xlabel('Dim 1')
ax1.set_ylabel('Dim 2')
ax1.set_zlabel('Dim 3')
        
for pick in picks:
    trial = trajectories_all[pick]
    if behav['choice'][pick] == 1:
        # left reaches
        ax1.plot(trial[0], trial[1], trial[2], '-', lw=0.5, c='C0', alpha=0.3)
    else:
        # reach reaches
        ax1.plot(trial[0], trial[1], trial[2], '-', lw=0.5, c='C1', alpha=0.3)

# avg right reaches (green dot)
ax1.plot(r_avg_traj[0], r_avg_traj[1], r_avg_traj[2], '-', lw=2, c='C5')
ax1.scatter(r_avg_traj[0][checkOnIdx], r_avg_traj[1][checkOnIdx], r_avg_traj[2][checkOnIdx], s=50, c='C5', alpha=0.8)
# ax1.scatter(r_avg_traj[0][r_avg_mvmt_idx], r_avg_traj[1][r_avg_mvmt_idx], r_avg_traj[2][r_avg_mvmt_idx], s=50, c='C5', alpha=0.8)

# avg left reaches
ax1.plot(l_avg_traj[0], l_avg_traj[1], l_avg_traj[2], '-', lw=2, c='C7')
ax1.scatter(l_avg_traj[0][checkOnIdx], l_avg_traj[1][checkOnIdx], l_avg_traj[2][checkOnIdx], s=100, c='C7', alpha=0.5)
# ax1.scatter(r_avg_traj[0][l_avg_mvmt_idx], r_avg_traj[1][l_avg_mvmt_idx], r_avg_traj[2][l_avg_mvmt_idx], s=100, c='C3', alpha=0.8)
transBackground(ax1,3)
# ax1.legend(['right(red)','left(green)','avg'])

plt.show()

# get trials by cue == {214, 180} and then use those trials for plots in this section (3.2)
# reduce fast and slow trials to those with cues of 
cue214 = behav[behav['cue']==214]
cue180 = behav[behav['cue']==180]
cueBehav = pd.concat([cue214, cue180])

# get fast and slow reach trials
quantiles = cueBehav['RT'].quantile([0.25,0.75])
fast_RT_trials = cueBehav[cueBehav['RT'] <= quantiles[0.25]]
slow_RT_trials = cueBehav[cueBehav['RT'] >= quantiles[0.75]]
fastRTidx = list(fast_RT_trials.index)
slowRTidx = list(slow_RT_trials.index)

# separate fast and slow trials by choice
fast_left = fast_RT_trials[fast_RT_trials['choice']==1]
fast_right = fast_RT_trials[fast_RT_trials['choice']==2]
slow_left = slow_RT_trials[slow_RT_trials['choice']==1]
slow_right = slow_RT_trials[slow_RT_trials['choice']==2]

fl_idx = list(fast_left.index)
fr_idx = list(fast_right.index)
sl_idx = list(slow_left.index)
sr_idx = list(slow_right.index)

fl_traj = trajectories_all[fl_idx]
fr_traj = trajectories_all[fr_idx]
sl_traj = trajectories_all[sl_idx]
sr_traj = trajectories_all[sr_idx]

# average trajectoroes for fl, fr, sl, sr
fl_avg = np.mean(fl_traj, axis=0)
fr_avg = np.mean(fr_traj, axis=0)
sl_avg = np.mean(sl_traj, axis=0)
sr_avg = np.mean(sr_traj, axis=0)

# avg time to mvmt onset
fl_avg_idx = RT_idx(np.mean(behav['mvmt'][fl_idx]))
fr_avg_idx = RT_idx(np.mean(behav['mvmt'][fr_idx]))
sl_avg_idx = RT_idx(np.mean(behav['mvmt'][sl_idx]))
sr_avg_idx = RT_idx(np.mean(behav['mvmt'][sr_idx]))

##  plot setup

f = plt.figure(figsize=(10,8))
ax1 = f.add_subplot(1, 1, 1, projection='3d')

## ax1 - plot by reach direction

ax1.set_title('Reach Direction, left==teal, right==yellow, slow==dashed, fast==line')

# plot fast left
ct = 0
for traj in fl_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(fl_idx[ct])
    # plot
    ax1.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-', lw=1, c='C0', alpha=0.7)
    ax1.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C0', alpha=0.8)
    ct += 1
    
# plot slow left
ct = 0
for traj in sl_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(sl_idx[ct])
    # plot
    ax1.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-.', lw=1, c='C0', alpha=0.5)
    ax1.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C0', alpha=0.8)
    ct += 1
    
# plot fast right
ct = 0
for traj in fr_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(fr_idx[ct])
    # plot
    ax1.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-', lw=1, c='C1', alpha=0.5)
    ax1.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C1', alpha=0.8)
    ct += 1

# plot slow right
ct = 0
for traj in sr_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(sr_idx[ct])
    # plot
    ax1.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-.', lw=1, c='C1', alpha=0.5)
    ax1.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C1', alpha=0.8)
    ct += 1

transBackground(ax1,3)
plt.show()

##  plot setup

f = plt.figure(figsize=(10,8))
ax2 = f.add_subplot(1,1,1, projection='3d')

## ax2 - plot by RT

ax2.set_title('RT, slow==red, fast==purple, left==line, right==dashed')

# plot fast left
ct = 0
for traj in fl_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(fl_idx[ct])
    # plot
    ax2.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-', lw=1, c='C2', alpha=0.5)
    ax2.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C2', alpha=0.8)
    ct += 1
    
# plot slow left
ct = 0
for traj in sl_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(sl_idx[ct])
    # plot
    ax2.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-', lw=1, c='C3', alpha=0.5)
    ax2.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C3', alpha=0.8)
    ct += 1
    
# plot fast right
ct = 0
for traj in fr_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(fr_idx[ct])
    # plot
    ax2.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-.', lw=1, c='C2', alpha=0.5)
    ax2.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C2', alpha=0.8)
    ct += 1

# plot slow right
ct = 0
for traj in sr_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(sr_idx[ct])
    # plot
    ax2.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-.', lw=1, c='C3', alpha=0.5)
    ax2.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C3', alpha=0.8)
    ct += 1

transBackground(ax2,3)
plt.show()

##  plot setup

f = plt.figure(figsize=(10,8))
ax3 = f.add_subplot(1, 1, 1, projection='3d')

## ax3 - plot by RT and reach direction

ax3.set_title('FL==teal, SL==yellow, FR==purple, SR==green')

# plot fast left
ct = 0
for traj in fl_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(fl_idx[ct])
    # plot
    ax3.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-', lw=1, c='C0', alpha=0.5)
    ax3.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C0', alpha=0.6)
    ct += 1
    
# plot slow left
ct = 0
for traj in sl_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(sl_idx[ct])
    # plot
    ax3.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-', lw=1, c='C1', alpha=0.5)
    ax3.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C1', alpha=0.6)
    ct += 1
    
# plot fast right
ct = 0
for traj in fr_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(fr_idx[ct])
    # plot
    ax3.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-', lw=1, c='C7', alpha=0.5)
    ax3.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C7', alpha=0.6)
    ct += 1

# plot slow right
ct = 0
for traj in sr_traj:
    # get mvmt onset index
    t_idx = mvmtIdx(sr_idx[ct])
    # plot
    ax3.plot(traj[0][:t_idx],traj[1][:t_idx],traj[2][:t_idx],'-', lw=1, c='C8', alpha=0.5)
    ax3.scatter(traj[0][checkOnIdx],traj[1][checkOnIdx],traj[2][checkOnIdx], lw=0.5, c='C8', alpha=0.6)
    ct += 1

transBackground(ax3,3)
plt.show()

##  plot setup

f = plt.figure(figsize=(10,8))
ax4 = f.add_subplot(1, 1, 1, projection='3d')

## ax4 - plot ax3 but averaged trajectories

ax4.set_title('FL==teal, SL==yellow, FR==purple, SR==green')

# plot fast left
ax4.plot(fl_avg[0][:fl_avg_idx],fl_avg[1][:fl_avg_idx],fl_avg[2][:fl_avg_idx],'-', lw=2, c='C0', alpha=1)
ax4.scatter(fl_avg[0][checkOnIdx],fl_avg[1][checkOnIdx],fl_avg[2][checkOnIdx], s=100, lw=0.5, c='C0', alpha=0.6)
    
# plot slow left
ax4.plot(sl_avg[0][:sl_avg_idx],sl_avg[1][:sl_avg_idx],sl_avg[2][:sl_avg_idx],'-', lw=2, c='C1', alpha=1)
ax4.scatter(sl_avg[0][checkOnIdx],sl_avg[1][checkOnIdx],sl_avg[2][checkOnIdx], s=100, lw=0.5, c='C1', alpha=0.6)
    
# plot fast right
ax4.plot(fr_avg[0][:fr_avg_idx],fr_avg[1][:fr_avg_idx],fr_avg[2][:fr_avg_idx],'-', lw=2, c='C7', alpha=1)
ax4.scatter(fr_avg[0][checkOnIdx],fr_avg[1][checkOnIdx],fr_avg[2][checkOnIdx], s=100, lw=0.5, c='C7', alpha=0.6)

# plot slow right
ax4.plot(sr_avg[0][:sr_avg_idx],sr_avg[1][:sr_avg_idx],sr_avg[2][:sr_avg_idx],'-', lw=2, c='C8', alpha=1)
ax4.scatter(sr_avg[0][checkOnIdx],sr_avg[1][checkOnIdx],sr_avg[2][checkOnIdx], s=100, lw=0.5, c='C8', alpha=0.6)

transBackground(ax4,3)
plt.show()

# define RT bins
RT_bins = {}
ct = 0
for i in range(min(behav['RT']), max(behav['RT']), 100):
    RT_bins[ct] = [i,i+100]
    ct += 1
    
# get trials for each bin
binned_trials = {}
for i, binArray in RT_bins.items():
    binned_trials[i] = behav[(behav['RT'] >= binArray[0]) & (behav['RT'] < binArray[1])]
    
# delete empty bins
del binned_trials[10]
del RT_bins[10]

# get mvmt idx for each of the averaged trajectories
mvmtIdx = {}
for i, trials in binned_trials.items():
    mvmtIdx[i] = RT_idx(np.mean(trials['mvmt']))
    if mvmtIdx[i] > 57:
        mvmtIdx[i] = 57

# get left and right trials for each bin
temp = {}
for i, trials in binned_trials.items():
    temp[i] = {}
    temp[i]['left'] = binned_trials[i][binned_trials[i]['choice']==1]
    temp[i]['right'] = binned_trials[i][binned_trials[i]['choice']==2]
binned_trials = temp

# delete empty bins
del binned_trials[11]['left']

# average the binned trials
avg_traj = {}
for i in binned_trials.keys():
    avg_traj[i] = {}
    for dir in binned_trials[i].keys():
        trials = binned_trials[i][dir]
        idx = list(trials.index)
        traj = trajectories_all[idx]
        avg_traj[i][dir] = np.mean(traj, axis=0)
            
##  plot setup

f = plt.figure(figsize=(10,8))
ax1 = f.add_subplot(1, 1, 1, projection='3d')

## ax1 - RT-binned and averaged trajectories

ax1.set_title('RT binned and averaged, left==dashed, right==line')

cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, len(avg_traj.keys()))]
colors[0] = [0.7,0.7,0.7,1]

# average the binned trials
ct = 0
for i in avg_traj.keys():
    for dir in avg_traj[i].keys():
        traj = avg_traj[i][dir]
        if dir=='right':
            ax1.plot(traj[0][:mvmtIdx[i]], traj[1][:mvmtIdx[i]], traj[3][:mvmtIdx[i]],'-', c=colors[ct], label=str(RT_bins[i]) + ' ms')
            ax1.scatter(traj[0][checkOnIdx], traj[1][checkOnIdx], traj[3][checkOnIdx], s=30, c=np.array(colors[ct][:3]).reshape(1,3))
        else:
            ax1.plot(traj[0][:mvmtIdx[i]], traj[1][:mvmtIdx[i]], traj[3][:mvmtIdx[i]],'--', c=colors[ct])
            ax1.scatter(traj[0][checkOnIdx], traj[1][checkOnIdx], traj[3][checkOnIdx], s=30, c=np.array(colors[ct][:3]).reshape(1,3))
    ct += 1
    if i==5:
        break

transBackground(ax1,3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()