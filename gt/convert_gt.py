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


def quat_multiply(q1, q2):
  x = (q1[3]*q2[0]) + (q1[0]*q2[3]) + (q1[1]*q2[2]) - (q1[2]*q1[1])
  y = (q1[3]*q2[1]) + (q1[0]*q2[2]) + (q1[1]*q2[3]) - (q1[2]*q2[0])
  z = (q1[3]*q2[2]) + (q1[0]*q2[1]) + (q1[1]*q2[0]) - (q1[2]*q2[3])
  w = (q1[3]*q2[3]) + (q1[0]*q2[0]) + (q1[1]*q2[1]) - (q1[2]*q2[2])
  return [x, y, z, w] / np.linalg.norm([x, y, z, w])


def quat_diff(q1, q2):
  # https://www.cprogramming.com/tutorial/3d/quaternions.html
  q1_inv = [-q1[0], -q1[1], -q1[2], q1[3]]
  rot = quat_multiply(q2, q1_inv)
  return rot


def convert(data, times):
  results = []
  prev_pose = None
  for time, pose in zip(times, data):
    pose = _34_posquat(pose)
    if prev_pose is not None:
      mov = np.asarray(pose[0:3]) - np.asarray(prev_pose[0:3])
      rot = quat_diff(prev_pose, pose)
      results.append(np.append(time, np.append(mov, rot)))
    prev_pose = pose
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
