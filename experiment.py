"""
This module runs experiments using the pipeline as described in Figure 1 in:

Learning to fuse: A deep learning approach to visual-inertial camera pose estimation,
Rambach, J. R., Tewari, A., Pagani, A., & Stricker, D. (2016, September).

Joost Hoppenbrouwer
"""

import sys
import argparse
import os
import numpy as np

from pipeline import pipeline
import modules
from LKF import LKF

TIMES_PATH = '/home/joost/Documents/Master_Thesis/vio/gt/'
STORE_PATH = '/home/joost/Documents/Master_Thesis/vio/KITTI_results/'

if __name__=="__main__":
  # parse command line
  parser = argparse.ArgumentParser(description='''This script estimates trajectories using fused visual and inertial data.''')
  parser.add_argument('--seq', type = str, default = '00', help='sequence')
  parser.add_argument('--mov_std', type = float, default = 0.00, help='std for movement noise')
  parser.add_argument('--rot_std', type = float, default = 0.00, help='std for rotation noise')
  parser.add_argument('--store', type = bool, default = False, help='store gt and orb trajectories')
  args = parser.parse_args()

  # set up pipeline
  p = pipeline(mov_noise_std=args.mov_std, rot_noise_std=args.rot_std)

  # load timestamps for sequence
  with open(os.path.join(TIMES_PATH, args.seq + '_times.txt')) as f:
    times = map(float, f.read().splitlines())

  # get poses for every timestamp
  inertial_poses = []
  visual_poses = []
  poses = []
  for t in times:
    inertial_pose, visual_pose, pose = p.update(args.seq, t)
    inertial_poses.append(np.append(t, inertial_pose))
    visual_poses.append(np.append(t, visual_pose))
    poses.append(np.append(t, pose))
  # store
  if args.store:
    np.savetxt(os.path.join(STORE_PATH, 'gt/', args.seq + '.txt'), inertial_poses)
    np.savetxt(os.path.join(STORE_PATH, 'ORB_SLAM2/', args.seq + '.txt'), visual_poses)
  np.savetxt(os.path.join(STORE_PATH, 'vio/', args.seq + '.txt'), poses)
