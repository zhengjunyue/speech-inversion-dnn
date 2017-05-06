# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:31:30 2016

@author: ganesh
"""

import struct
import numpy as np
import sys
import os

def writehtk(data, speriod, opfnm):
    # Please supply the data in the format of (nsamples x nfeatures)
    # speriod must be in milliseconds
    speriod_htk = speriod*1e4 # Converting this to 100ns format
    nsamples = data.shape[0]
    featdim = data.shape[1]
    h = struct.pack(
          '>iihh', # the beginning '>' says write big-endian
          nsamples,# nSamples
          speriod_htk, #samplePeriod
          4*featdim, # 2 floats per feature
          9) # user features
    assert len(h) == 12
    s = ''
#    for row in data:
#        for col in row:
#            s += struct.pack('>f', col)
    data_flat = data.flatten()    
    fmt = '>'+str(len(data_flat))+'f'
    s = struct.pack(fmt, *data_flat)
    
    # write it to an output file
    dnm=os.path.split(opfnm)[0]
    if not os.path.isdir(dnm):
        os.makedirs(dnm)
    outFile = open(opfnm, 'wb')
    outFile.write(h)
    outFile.write(s)
    outFile.close()
#    
#if __name__ == "__main__":
#    a = np.round(np.random.rand(20,4)*100)
#    print(a)
#    writehtk(a, 10, './test_writehtk.htk')
