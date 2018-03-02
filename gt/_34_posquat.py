#!/usr/bin/python
"""
This script transforms gt in 3x4 matrix format to pos-quaternion format.
"""

import sys
import numpy as np
import argparse
import os

BASE_PATH = '/home/joost/Documents/Master_Thesis/vio/gt/'

def _34_posquat(mat):
  # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
  tr = mat[0] + mat[5] + mat[10]
  if (tr > 0):
    S = np.sqrt(tr + 1.0) * 2
    qw = 0.25 * S
    qx = (mat[9] - mat[6]) / S
    qy = (mat[2] - mat[8]) / S
    qz = (mat[4] - mat[1]) / S
  elif ((mat[0] > mat[5]) and (mat[0] > mat[10])):
    S = np.sqrt(1.0 + mat[0] - mat[5] - mat[10]) * 2
    qw = (mat[9] - mat[6]) / S
    qx = 0.25 * S
    qy = (mat[1] + mat[4]) / S
    qz = (mat[2] + mat[8]) / S
  elif (mat[5] > mat[10]):
    S = np.sqrt(1.0 + mat[5] - mat[0] - mat[10]) * 2
    qw = (mat[2] - mat[8]) / S
    qx = (mat[1] + mat[4]) / S
    qy = 0.25 * S
    qz = (mat[6] + mat[9]) / S
  else:
    S = np.sqrt(1.0 + mat[10] - mat[0] - mat[5]) * 2
    qw = (mat[4] - mat[1]) / S
    qx = (mat[2] + mat[8]) / S
    qy = (mat[6] + mat[9]) / S
    qz = 0.25 * S
  return [mat[3], mat[7], mat[11], qx, qy, qz, qw]


def convert(data, times):
  results = []
  for time, point in zip(times, data):
      posquat = _34_posquat(point.flatten())
      results.append(np.append(time, posquat))
  return results


if __name__=="__main__":
  # parse command line
  parser = argparse.ArgumentParser(description='''
  This script transforms gt in 3x4 matrix format to pos-quaternion format. 
  ''')
  parser.add_argument('seq', help='sequence')
  args = parser.parse_args()
  gt = os.path.join(BASE_PATH, args.seq + '_original.txt')
  times = os.path.join(BASE_PATH, args.seq + '_times.txt')

  data = np.loadtxt(gt)
  timestamps = np.loadtxt(times)
  results = convert(data, timestamps)
  np.savetxt(os.path.join(BASE_PATH, args.seq + '.txt'), results)
