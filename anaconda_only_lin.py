import os
import sys
import numpy as np
from numpy import *
import scipy.interpolate
import pandas as pd
import time
import easygui
import originpro as op
import re

print("starting")
file_path_short = easygui.fileopenbox( "choose an original spectrum", "Video", "F:\\qLILBID\\Kinetik\\quatsch")
print("Only choose indices of a single sample")
start_number = input("Start number: ")
end_number = input("End number: ")
sample = input("Sample name: ")
path_array =  file_path_short.rsplit('\\', 3)[0]
print(path_array)
file_path = file_path_short.rsplit('\\', 1)[0] + "\\"
edit_path = path_array + "\\Spektren\\bearbeitet\\"
# smooth_factor = input("Smooth factor (multiple of 2)")
# soll1 = int(input("Soll 1"))
# soll2 = int(input("Soll 2"))
# ist1 = int(input("Ist 1"))
# ist2 = int(input("Ist 2"))
# linfactor = int(input("Linearization factor"))
##limitsfile = input("Path incl. file name for limits file")

t1 = time.time()

##start_number = "022"
##end_number = "022"
# file_path = "F:\qLILBID\Kinetik\160321\Spektren\original\\"
smooth_factor = 50
soll1 = 42500
soll2 = 21000
ist1 = 8188
ist2 = 4020
linfactor = 10
##limitsfile = "G:\programming and other brainstorming\python\python for me\peaks.txt"

##file_path = "G:\programming and other brainstorming\python\python for me\output videos\\"
setsfilelist = os.listdir(file_path)

#create a list with the relevant files
relevantfilelist = []
for set in setsfilelist:
    if int(start_number) <= int(set[-7:-4]) <= int(end_number):
        full_path = file_path + set
        if os.path.getsize(full_path)>0:
            relevantfilelist.append(set)

#create data array for spectral data
numsets = len(relevantfilelist)
data = np.zeros((numsets,12000,99))

#add the relevant files to the data array
print(relevantfilelist)

for set in relevantfilelist:
    full_path = file_path + set
    readfile = loadtxt(full_path, delimiter="\t", skiprows=3, usecols = range(3,200,2))
    index = relevantfilelist.index(set)
    np.copyto(data[index,:,:],readfile[0:12000,0:99])

#calibrate data
full_path = file_path + relevantfilelist[0]
xaxis_raw = loadtxt(full_path, delimiter="\t", skiprows=3, usecols = 0)
a = (soll2-soll1)/(ist2-ist1)
b = soll1-(a*ist1)
xaxis_calib = (a*xaxis_raw)+b
xaxis_calib = xaxis_calib[0:12000]
x_file1 = edit_path + "xaxiscalib.txt"
np.savetxt(x_file1,xaxis_calib)

#smooth data
smoothed = np.zeros((numsets,12000,99))

def smooth(data, output, smoothfactor):
    halffactor = int(smoothfactor)/2
    it = np.nditer([output], op_flags=['readwrite'], flags=['multi_index'])
    while not it.finished:
        if it.multi_index[1] > halffactor:
            start = it.multi_index[1]-halffactor
            end = it.multi_index[1]+halffactor+1
            relevantarray = data[it.multi_index[0],int(start):int(end),it.multi_index[2]]
            it[0] = np.mean(relevantarray)
        else:
            end = it.multi_index[1]+halffactor+1
            relevantarray = data[it.multi_index[0],:int(end),it.multi_index[2]]
            it[0] = np.mean(relevantarray)
        it.iternext()
    return output

smoothed = (smooth(data,smoothed,smooth_factor))

#export smoothed data
for file in relevantfilelist:
    exportname = edit_path + str(int(file[-7:-4])) + "smoothed.txt"
    print(exportname)
    index = relevantfilelist.index(file)
    np.savetxt(exportname,smoothed[index,:,:],delimiter='\t')

#prep to linearize data
max_x = int(xaxis_calib[-1])
min_x = int(xaxis_calib[0])
goal_x = range(min_x, max_x, linfactor)
print("goal_x length is " + str(len(goal_x)))
x_file2 = edit_path + "xaxislin.txt"
np.savetxt(x_file2,goal_x)

##linearized = np.zeros((numsets,len(goal_x),5))
##linearized = np.zeros((numsets,len(goal_x),5))
linearized = np.zeros((numsets,len(goal_x),99))
print("shape of linearized is " + str(linearized.shape))

def find_nearest(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

startindexes = []
stopindexes = []

for x in goal_x:
    start_x = find_nearest(xaxis_calib,(x - (linfactor/2)))
    startindexes.append(start_x)
    stop_x = find_nearest(xaxis_calib,(x + (linfactor/2)))
    stopindexes.append(stop_x)

print("start_indexes reached! First 6 are:" + str(startindexes[0:5]))

#define function for linearization
def linearize(smoothed, linearized):
    it = np.nditer([linearized], op_flags=['readwrite'], flags=['multi_index'])
    while not it.finished:
        start = startindexes[it.multi_index[1]]
##        print("start is " + str(start))
        stop = stopindexes[it.multi_index[1]]
##        print("stop is " + str(stop))
        tempy = np.array(smoothed[it.multi_index[0],start:(stop+1),it.multi_index[2]])
        xtempx = np.array(xaxis_calib[start:(stop+1)])
        tempx = xtempx.astype(int)
        # print("y:" + str(tempy))
        # print("x:" + str(tempx))
##        print("tempy is " + str(tempy))
##        print("tempx is " + str(tempx))
        index = it.multi_index[1]
        if len(tempx)>1:
            interp = scipy.interpolate.interp1d(tempx, tempy)
            it[0] = interp.__call__(goal_x[index])
            it[0] = it[0] - np.nanmean(smoothed[it.multi_index[0],20:70,it.multi_index[2]])
        else:
            try:
                it[0] = linearized[it.multi_index[0],(it.multi_index[1]-1),it.multi_index[2]]+(smoothed[it.multi_index[0],start,it.multi_index[2]]-smoothed[it.multi_index[0],start+1,it.multi_index[2]])*(goal_x[index]-goal_x[index-1])/(xaxis_calib[start]-xaxis_calib[start+1])
            except:
                it[0] = np.nan
        it.iternext()
    return linearized

linearized = linearize(smoothed, linearized)

#export linearized data
for file in relevantfilelist:
    exportname = edit_path + str(int(file[-7:-4])) + "linearized.txt"
    print(exportname)
    index = relevantfilelist.index(file)
    np.savetxt(exportname,linearized[index,:,:],delimiter='\t')