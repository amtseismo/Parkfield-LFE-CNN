#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:08:45 2020

Script to run CNN on continuous data

@author: amt
"""

from obspy.core import read, Stream
import numpy as np
import matplotlib.pyplot as plt
import lfe_unet_tools
from scipy.signal import find_peaks
import pickle
import datetime

# SET THESE PARAMETERS HOW YOU WISH
thresh=0.1 # minimum decision threshold to log detections
shift=15 # time window step size in seconds
drop=0 # drop layer, 1 if you want to include, 0 if not
large=0.5 # model size (can be 0.5, 1, 2)
std=0.2 # how wide do you want the gaussian STD to be in seconds?
plots=0 # do you want plots?

# LEAVE THESE
sr=100 # sample rate in Hz, network was trained on 100Hz data so best keep this as is
winlen=15 # window length in seconds 
nwin=int(sr*winlen) # leave this
nshift=int(sr*shift) # leave this
epsilon=1e-6 # leave this

# MODEL NAME
model_save_file="large_"+'{:3.1f}'.format(large)+"_unet_lfe_std_"+str(std)+".tf"

if drop:
    model_save_file="drop_"+model_save_file

# BUILD THE MODEL
print("BUILD THE MODEL")
if drop:
    model=lfe_unet_tools.make_large_unet_drop(large,sr,ncomps=3)
else:
    model=lfe_unet_tools.make_large_unet(large,sr,ncomps=3)

# LOAD THE MODEL PARAMETERS
print('Loading training results from '+model_save_file)
model.load_weights("./models_v2/"+model_save_file)

# LOAD DATA
day=5
for sta in ['B079','40','THIS']: 
    print(sta+' '+str(day))
    st=Stream()
    if sta=='40':
        st=read("/Users/amt/Documents/parkfield_nodal_deployment/data/1B.40..DPZ.2018-08-0"+str(day)+"-00-00-00.ms")
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/1B.40..DP1.2018-08-0"+str(day)+"-00-00-00.ms")
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/1B.40..DP2.2018-08-0"+str(day)+"-00-00-00.ms")
    elif sta=="THIS":
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".BK."+sta+".HHZ.ms")
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".BK."+sta+".HHE.ms")
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".BK."+sta+".HHN.ms")
    elif sta=="B079":
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".PB."+sta+".EHZ.ms")
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".PB."+sta+".EH1.ms")
        st+=read("/Users/amt/Documents/parkfield_nodal_deployment/data/2018-08-0"+str(day)+".PB."+sta+".EH2.ms")    
    
    # PROCESS DATA
    st.detrend(type='simple')
    st.filter("highpass", freq=1.0, zerophase=True)
    
    # ENFORCE COMPONENTS HAVE SAME START AND END TIME
    start=st[0].stats.starttime
    finish=st[0].stats.endtime
    for ii in range(1,len(st)):
        if start<st[ii].stats.starttime:
            start=st[ii].stats.starttime
        if finish>st[ii].stats.endtime:
            finish=st[ii].stats.endtime
    st.trim(starttime=start, endtime=finish,nearest_sample=1,pad=1,fill_value=0)
    
    # INTERPOLATE IF SR != 100
    for tr in st:
        if sr != tr.stats.sampling_rate:
            tr.interpolate(sampling_rate=sr, starttime=start)
            
    # APPLY CNN TO DATA
    nn=(len(st[0].data)-nwin)//nshift # number of windows
    sdects=[] # intialize detection structure
    codestart=datetime.datetime.now()
    for ii in range(nn+1): # loop through windows
        data0s=lfe_unet_tools.simple_detrend(st[0].data[ii*nshift:ii*nshift+nwin])
        data1s=lfe_unet_tools.simple_detrend(st[1].data[ii*nshift:ii*nshift+nwin])
        data2s=lfe_unet_tools.simple_detrend(st[2].data[ii*nshift:ii*nshift+nwin])
        snip=np.concatenate((data0s,data1s,data2s))
        sign=np.sign(snip)
        val=np.log(np.abs(snip)+epsilon)
        cnninput=np.hstack( [val[:1500].reshape(-1,1), sign[:1500].reshape(-1,1), val[1500:3000].reshape(-1,1), sign[1500:3000].reshape(-1,1), val[3000:].reshape(-1,1), sign[3000:].reshape(-1,1)] )
        cnninput=cnninput[np.newaxis,:,:]
        # make s predictions
        stmp=model.predict(cnninput)
        stmp=stmp.ravel()
        spk=find_peaks(stmp, height=thresh, distance=200)    
        if plots:
            fig, ax = plt.subplots(4,1,figsize=(8,12))
            nl=len(snip)//3
            for jj in range(3):
                ax[jj].plot(snip[jj*nl:(jj+1)*nl])
            ax[3].plot(stmp,color=(0.25,0.25,0.25))
            for jj in range(len(spk[0])):
                ax[3].axvline(spk[0][jj],color='b') 
            ax[3].set_ylim((0,1))
        if len(spk[0]>0):     
            for kk in range(len(spk[0])):
                sdects.append([(spk[0][kk]+ii*nshift)/sr, spk[1]['peak_heights'][kk]])
    codestop=datetime.datetime.now()
    
    # SAVE DETECTION FILES
    dects=np.asarray(sdects)            
    file='picks-'+str(drop)+'-'+str(large)+'-'+str(std)+'_'+sta+'-'+str(2018)+'-'+str(8).zfill(2)+'-'+str(day).zfill(2)+'.pkl'
    with open(file, "wb") as fp:   #Pickling
        pickle.dump(dects, fp)