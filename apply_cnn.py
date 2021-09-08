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

def ZEN2inp(Z,E,N,epsilon):
    '''
    Apply feature normalizations to raw timeseries and convert
    to CNN input format

    Parameters:
        Z (float array): The vertical component data.
        E (float array): The east component data.
        N (float array): The north component data.
        epsilon (float): constant value added to input data to avoid zeros.

    Returns:
        data_inp (float array): CNN input
    '''
    data_Z_sign = np.sign(Z)
    data_E_sign = np.sign(E)
    data_N_sign = np.sign(N)
    data_Z_val = np.log(np.abs(Z)+epsilon)
    data_E_val = np.log(np.abs(E)+epsilon)
    data_N_val = np.log(np.abs(N)+epsilon)
    data_inp = np.hstack([data_Z_val.reshape(-1,1),data_Z_sign.reshape(-1,1),
                          data_E_val.reshape(-1,1),data_E_sign.reshape(-1,1),
                          data_N_val.reshape(-1,1),data_N_sign.reshape(-1,1),])
    return data_inp

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
    
    # ENFORCE COMPONENTS TO HAVE SAME START AND END TIME
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
    sav_data = []
    codestart=datetime.datetime.now() # start timer

    for ii in range(nn+1):
        # print(ii)
        data0s=lfe_unet_tools.simple_detrend(st[0].data[ii*nshift:ii*nshift+nwin])
        data1s=lfe_unet_tools.simple_detrend(st[1].data[ii*nshift:ii*nshift+nwin])
        data2s=lfe_unet_tools.simple_detrend(st[2].data[ii*nshift:ii*nshift+nwin])
        snip=np.concatenate((data0s,data1s,data2s))
        data_inp=ZEN2inp(data0s,data1s,data2s,epsilon)
        sav_data.append(data_inp)  # run model prediction in batch
    sav_data = np.array(sav_data)
    # make s predictions
    stmp=model.predict(sav_data)
    stmp=stmp.ravel()
    spk=find_peaks(stmp, height=thresh, distance=200)    
    sdects=np.hstack((spk[0].reshape(-1,1)/sr,spk[1]['peak_heights'].reshape(-1,1)))
    codestop=datetime.datetime.now()
    
    # SAVE DETECTION FILES
    dects=np.asarray(sdects)            
    file='picks-'+str(drop)+'-'+str(large)+'-'+str(std)+'_'+sta+'-'+str(2018)+'-'+str(8).zfill(2)+'-'+str(day).zfill(2)+'.pkl'
    with open(file, "wb") as fp:   #Pickling
        pickle.dump(dects, fp)