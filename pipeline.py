"""
This module implements the pipeline as described in Figure 1 in:

Learning to fuse: A deep learning approach to visual-inertial camera pose estimation,
Rambach, J. R., Tewari, A., Pagani, A., & Stricker, D. (2016, September).

Joost Hoppenbrouwer
"""

import sys
import argparse
import os
import numpy as np

import modules
from LKF import LKF

DEFAULT_SEQUENCE = '00'
TIMES_PATH = '/home/joost/Documents/Master_Thesis/vio/gt/'

class pipeline():
  """
  This class implements the full pipeline.
  """
  
  def __init__(self, mov_noise_std=0.0, rot_noise_std=0.0, dt_th=1.0, dq_th=5.0):
    """
    Set up the pipeline.
    """
    self.INERTIAL_MODULE = modules.noisy_gt(mov_noise_std, rot_noise_std)
    self.VISUAL_MODULE = modules.ORB_SLAM2()
    #self.ERROR_MODULE = modules.error_module(dt_th, dq_th)
    self.KALMAN_FILTER = LKF(mov_noise_std, rot_noise_std)


  def update(self, seq, timestamp):
    """
    Retrieve next pose from input data.
    NOTE: loads pre-generated data.
    """
    gt, inertial_pose = self.INERTIAL_MODULE.get_pose(seq, timestamp)
    visual_pose = self.VISUAL_MODULE.get_pose(seq, timestamp)
    #filtered = self.ERROR_MODULE.filter(inertial_pose, visual_pose)
    return gt, self.KALMAN_FILTER.step(inertial_pose, visual_pose)


if __name__=="__main__":
  # parse command line
  parser = argparse.ArgumentParser(description='''This script estimates trajectories using fused visual and inertial data.''')
  parser.add_argument('--seq', type = str, default = DEFAULT_SEQUENCE, help='sequence')
  args = parser.parse_args()

  # set up pipeline
  p = pipeline(mov_noise_std=0.5, rot_noise_std=0.01)

  # load timestamps for sequence
  with open(os.path.join(TIMES_PATH, args.seq + '_times.txt')) as f:
    times = map(float, f.read().splitlines())

  # get pose for every timestamp
  for t in times:
    gt, pose = p.update(args.seq, t)
    print gt
    print pose
    print '----------------------------------------------------------'

