# Parkfield-LFE-CNN

## Description

This repository contains many variations of a Convolutional Neural Network trained to detect low-frequency earthquakes on the San Andreas fault.  The full descrition of the model, training data, and performance can be found in.

Thomas, A. M., A. Inbal, J. Searcy, D. R. Shelly, and R. Burgmann (202?) Identification of low-frequency earthquakes on the San Andreas fault with deep learning. Submitted to Geophysical Research Letters.

BibTeX:

    @article{thomas2020lfe,
        title={Identification of low-frequency earthquakes on the San Andreas fault with deep learning},
        author={Thomas, AM and Inbal, Asaf, and Searcy, Jacob and Shelly, David R and B{\"u}rgmann, R},
        journal={Geophysical Research Letters},
    }
	
## Requirements

In order to run the software you will need both [Obspy](https://docs.obspy.org/) (I have version 1.2.2) and [Tensorflow](https://www.tensorflow.org/) (I have version 1.14.0).  I recommend creating a [conda](https://docs.conda.io/en/latest/) environment and installing both packages into it.    

## Make LFE detections

To make detections with the CNN, you should select which version you'd like.  You need to choose from the paper your desired target gaussian standard deviation (std=0.05, 0.1, 0.2, or 0.4 seconds), your model size (fac=0.5, 1, or 2), whether you want a drop layer or not (drop=0 or 1).  If you just want me to pick the best one for you do nothing; I've already set the scripts up to use my preferred model.  The first script you will need is **apply_cnn.py**.  This script loads the data, processes it (e.g. filters and decimates/interpolates to 100 Hz) and runs the detector on 15 s windows that are not overlapping (i.e. shift=15).  It registers any detections that exceed a decision threshold of 0.1 (thresh=0.1).  If you want to make some plots of the windows and CNN output you can set plots=1.  This writes out a pickle file called picks_[station]-[year]-[month]-[day].pkl.

The second script makes Figure 4 from the paper.  I have included detection files (and data) from the three stations shown in the manuscript.  You are welcome to make your own detection files but I have included mine so it works.  