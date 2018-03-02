#!/usr/bin/python
"""
This script calculates the stdev for the error of ORB-SLAM2.
"""

import sys
import numpy as np
import argparse
import os

ORB_PATH = '/home/joost/Documents/Master_Thesis/vio/ORB_SLAM2/'
GT_PATH = '/home/joost/Documents/Master_Thesis/vio/gt/'

SCALES = {'00.txt': 16.21, '02.txt': 15.72, '04.txt': 27.04, '05.txt': 24.12,
          '06.txt': 24.84, '07.txt': 10.98, '08.txt': 10.55, '09.txt': 20.32, '10.txt': 16.67}

def diff(seq):
  errors = []
  gt = {}
  gt_raw = np.loadtxt(os.path.join(GT_PATH, seq))
  for pose in gt_raw:
    gt[round(pose[0],6)] = np.asarray(pose[1:])
  orb = np.loadtxt(os.path.join(ORB_PATH, seq))
  for pose in orb:
    error = np.append((np.asarray(pose[1:4]) * SCALES[seq]), np.asarray(pose[4:])) - gt[pose[0]]
    errors.append(error)
  return errors


if __name__=="__main__":
  total_error = []
  for seq in ['00.txt', '02.txt', '04.txt', '05.txt', '06.txt',
              '07.txt', '08.txt', '09.txt', '10.txt']:
    error = diff(seq)
    total_error.extend(error)
  print np.std(total_error, axis=0)
