#!/usr/bin/python
"""
This script transforms ORB-SLAM2 data from posquat to displacement.
"""

import sys
import numpy as np
import argparse
import os

BASE_PATH = '/home/joost/Documents/Master_Thesis/vio/ORB_SLAM2/'

SCALES = {'00': 16.21, '02': 15.72, '04': 27.04, '05': 24.12,
          '06': 24.84, '07': 10.98, '08': 10.55, '09': 20.32, '10': 16.67}

def convert(data, SCALE):
  results = []
  prev_point = None
  for point in data:
    if prev_point is not None:
      time = point[0]
      mov = (point[1:4] - prev_point[1:4]) * SCALE
      rot = quat_diff(prev_point[4:], point[4:])
      results.append(np.append(np.append(time, mov), rot))
    prev_point = point
  return results


def quat_diff(q1, q2):
  # https://www.cprogramming.com/tutorial/3d/quaternions.html
  q1_inv = [-q1[0], -q1[1], -q1[2], q1[3]]
  x = (q2[3]*q1_inv[0]) + (q2[0]*q1_inv[3]) + (q2[1]*q1_inv[2]) - (q2[2]*q1_inv[1])
  y = (q2[3]*q1_inv[1]) + (q2[0]*q1_inv[2]) + (q2[1]*q1_inv[3]) - (q2[2]*q1_inv[0])
  z = (q2[3]*q1_inv[2]) + (q2[0]*q1_inv[1]) + (q2[1]*q1_inv[0]) - (q2[2]*q1_inv[3])
  w = (q2[3]*q1_inv[3]) + (q2[0]*q1_inv[0]) + (q2[1]*q1_inv[1]) - (q2[2]*q1_inv[2])
  rot = [x, y, z, w]
  return rot / np.linalg.norm(rot)


if __name__=="__main__":
  # parse command line
  parser = argparse.ArgumentParser(description='''
  This script transforms ORB-SLAM2 data from posquat to displacement. 
  ''')
  parser.add_argument('seq', help='sequence')
  args = parser.parse_args()
  ORB_SLAM2 = os.path.join(BASE_PATH, args.seq + '_original.txt')

  data = np.loadtxt(ORB_SLAM2)
  results = convert(data, SCALES[args.seq])
  np.savetxt(os.path.join(BASE_PATH, args.seq + '.txt'), results)
