#!/usr/bin/python
"""
This script calculates the variance of the error of ORB-SLAM2.
"""

import sys
import numpy as np
import argparse
import os

ORB_PATH = '/home/joost/Documents/Master_Thesis/vio/ORB_SLAM2/'
GT_PATH = '/home/joost/Documents/Master_Thesis/vio/gt/'


def diff(seq):
  errors = []
  gt = {}
  gt_raw = np.loadtxt(os.path.join(GT_PATH, seq))
  prev_pose = None
  for pose in gt_raw:
    if prev_pose is not None:
      mov = pose[1:4] - prev_pose[1:4]
      rot = quat_diff(prev_pose[4:], pose[4:])
      gt[round(pose[0],6)] = np.append(mov, rot)
    prev_pose = pose
  orb = np.loadtxt(os.path.join(ORB_PATH, seq))
  prev_pose = None
  for pose in orb:
    if prev_pose is not None:
      gt_pose = gt[round(pose[0],6)]
      mov_error = pose[1:4] - gt_pose[0:3]
      rot_error = quat_diff(gt_pose[3:], pose[4:])
      errors.append(np.append(mov_error, rot_error))
    prev_pose = pose
  return errors


def quat_multiply(q1, q2):
  x = (q1[3]*q2[0]) + (q1[0]*q2[3]) + (q1[1]*q2[2]) - (q1[2]*q1[1])
  y = (q1[3]*q2[1]) + (q1[0]*q2[2]) + (q1[1]*q2[3]) - (q1[2]*q2[0])
  z = (q1[3]*q2[2]) + (q1[0]*q2[1]) + (q1[1]*q2[0]) - (q1[2]*q2[3])
  w = (q1[3]*q2[3]) + (q1[0]*q2[0]) + (q1[1]*q2[1]) - (q1[2]*q2[2])
  return [x, y, z, w]


def quat_diff(q1, q2):
  # https://www.cprogramming.com/tutorial/3d/quaternions.html
  q1_inv = [-q1[0], -q1[1], -q1[2], q1[3]]
  rot = quat_multiply(q2, q1_inv)
  return rot / np.linalg.norm(rot)


if __name__=="__main__":
  total_error = []
  for seq in ['00.txt', '02.txt', '04.txt', '05.txt', '06.txt',
              '07.txt', '08.txt', '09.txt', '10.txt']:
    error = diff(seq)
    total_error.extend(error)
  print np.std(total_error, axis=0)**2
