#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:54:13 2021

Makes figure 4 from Thomas et al. 2021

@author: amt
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from obspy.core import Stream
import matplotlib.patches as patches


# SET THESE PARAMETERS HOW YOU WISH
drop=0 # drop layer, 1 if you want to include, 0 if not
large=0.5 # model size (can be 0.5, 1, 2)
std=0.2 # how wide do you want the gaussian STD to be in seconds?

# LEAVE THESE
day=5 # August 5th is the day
thresh=0.1 # decision threshold

fig = plt.figure(constrained_layout=True,figsize=(18,12))
gs = fig.add_gridspec(3, 6)   

# make plots
label='ABCDEFGHIJ'
fac=10
# load data
for jj, sta in enumerate(['THIS','40','B079']): #,'B901']):

    # load picks
    file='picks-'+str(drop)+'-'+str(large)+'-'+str(std)+'_'+sta+'-'+str(2018)+'-'+str(8).zfill(2)+'-'+str(day).zfill(2)+'.pkl'
    picks=pickle.load( open(file, "rb" ) )
    
    # load data
    st=Stream()
    if sta=='40':
        st=read("/Users/amt/Documents/parkfield_nodal_deployment/data/1B.40..DP2.2018-08-0"+str(day)+"-00-00-00.ms")
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/1B.40..DP1.2018-08-0"+str(day)+"-00-00-00.ms")
    elif sta=='B079' or sta=='B901':
        st=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".PB."+sta+".EH1.ms")
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".PB."+sta+".EH2.ms")
    else:
        st=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".BK."+sta+".HHE.ms")
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".BK."+sta+".HHN.ms")

    st.filter("highpass", freq=1.0, zerophase=True) 
    t=np.arange(st[0].stats.npts)*st[0].stats.delta
    data1=st[0].data
    data2=st[1].data
    
    ax0=fig.add_subplot(gs[jj, 0:4])
    srange=24300,24900 
    picks_small=picks[(picks[:,0]>srange[0]) & (picks[:,0]<=srange[1])]
    inds=np.where((t>=srange[0]) & (t<srange[1]))[0]
    ax0.plot(t[inds],data1[inds]/(fac*np.median(np.abs(data1[inds])))+1,color=(0.5,0.5,0.5))
    ax0.plot(t[inds],data2[inds]/(fac*np.median(np.abs(data2[inds])))-1,color=(0.5,0.5,0.5))
    cx1=ax0.scatter(picks_small[:,0],-0.1*np.ones(len(picks_small)),c=picks_small[:,1],s=100,vmin=0.1,vmax=0.3,cmap='viridis',label='CNN Picks',edgecolors='b')
    ax0.set_xlim((srange))
    ax0.set_ylim((-2.1,2.1))
    ax0.set_yticks([])
    if jj==0:
        legend=ax0.legend(loc="lower left",fontsize=14)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
    ax0.tick_params(axis="x", labelsize=12)
    ax0.set_title(sta,fontsize=14)
    ax0.text(srange[0]+4,1.5,label[int(2*jj)],fontsize=28,fontweight='bold')
    
    # Create a Rectangle patch
    rect = patches.Rectangle((24825,-2.1), 8, 4.2, linewidth=2, edgecolor='k', facecolor='none')
    
    # Add the patch to the Axes
    ax0.add_patch(rect)
    
    #-------- PANEL C
    
    ax2=fig.add_subplot(gs[jj, 4:])
    sranges=24825,24833
    # bpinds=np.where((bpt>=srange[0]) & (bpt<=srange[1]))[0]
    picks_small=picks[(picks[:,0]>sranges[0]) & (picks[:,0]<=sranges[1])]
    #atinds=np.where((arrival_time_small[:,0]>=srange[0]) & (arrival_time_small[:,0]<=srange[1]))[0]
    inds=np.where((t>=sranges[0]) & (t<sranges[1]))[0]
    ax2.plot(t[inds],data1[inds]/(fac*np.median(np.abs(data1[inds])))+1,color=(0.5,0.5,0.5))
    ax2.plot(t[inds],data2[inds]/(fac*np.median(np.abs(data2[inds])))-1,color=(0.5,0.5,0.5))
    ax2.scatter(picks_small[:,0],-0.1*np.ones(len(picks_small)),c=picks_small[:,1],s=100,vmin=0.1,vmax=0.3,cmap='viridis',label='CNN Picks',edgecolors='b')
    ax2.set_xlim((sranges))
    # xticks = ax2.xaxis.get_major_ticks()
    # xticks[-1].set_visible(False)
    ax2.set_ylim((-2.1,2.1))
    ax2.set_yticks([])

    if jj==2:
        ax2.yaxis.set_major_locator(plt.NullLocator())
        ax2.set_xlabel("Second of day on Aug. 5 2018",fontsize=14,labelpad=15)
    ax2.tick_params(axis="x", labelsize=12)
    ax2.text(sranges[0]+0.5,1.5,label[int(2*jj)+1],fontsize=28,fontweight='bold')
    ax2.xaxis.set_major_locator(plt.MaxNLocator(4))

ax0.set_xlabel("Second of day on Aug. 5 2018",fontsize=14,labelpad=15)
cbaxes = fig.add_axes([0.675, 0.06, 0.2, 0.02]) 
cb = plt.colorbar(cx1, cax = cbaxes, orientation="horizontal")
cb.set_label(label='CNN Amplitude', size='large')
cb.patch.set_facecolor((0.2, 0.2, 0.2, 1.0))
cb.outline.set_linewidth(2)
cb.ax.tick_params(labelsize=12)

fig.savefig("figure4.png",bbox_inches='tight')