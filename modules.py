"""
This module implements all modules used for the visual-inertial pipeline.
It contains the following modules:
- Inertial Module for generating poses from inertial data (pre-generated data in this case)
- Visual Module for generating poses from visual data (pre-generated data in this case)
- Error Module as described in Learning to fuse: A deep learning approach to visual-inertial camera pose estimation
- Kalman Filter as described in Learning to fuse: A deep learning approach to visual-inertial camera pose estimation

Joost Hoppenbrouwer
"""

import os.path
import numpy as np

DEFAULT_GT_PATH = '/home/joost/Documents/Master_Thesis/visual_inertial/gt/'
DEFAULT_ORB_SLAM2_PATH = '/home/joost/Documents/Master_Thesis/visual_inertial/ORB_SLAM2/'

class noisy_gt():
  """
  Class used as inertial module. Provides ground thruth data with chosen amount of error.
  """

  def __init__(self, noise, gt_path=DEFAULT_GT_PATH):
    self.noise = noise
    self.data = self.load_KITTI_gt(gt_path)


  def load_KITTI_gt(self, gt_path):
    """
    Loads ground truth for KITTI dataset.
    """
    data = {}
    # ORB-SLAM2 fails on sequence 01, sequence 03 has no inertial data
    for seq in ['00', '02', '04', '05', '06', '07', '08', '09', '10']:
      data[seq] = {}
      # load times
      with open(os.path.join(gt_path, seq + '_times.txt')) as f:
        times = map(float, f.read().splitlines())
      # load gt
      with open(os.path.join(gt_path, seq + '.txt')) as f:
        gt = [map(float, line.split()) for line in f.read().splitlines()]
      # convert gt to [x,y,z,qx,qy,qz,qw] and store
      for i, t in enumerate(times):
        data[seq][t] = np.asarray(_34_posquat(gt[i]))
    return data


  def get_pose(self, seq, timestamp):
    """
    Return pose at timestamp for given sequence.
    """
    return self.data[seq][timestamp]


class ORB_SLAM2():
  """
  Class used as visual module. Provides ORB-SLAM2 output.
  """

  def __init__(self, data_path=DEFAULT_ORB_SLAM2_PATH):
    self.SCALES = {'00': 16.21, '02': 15.72, '04': 27.04, '05': 24.12,
      '06': 24.84, '07': 10.98, '08': 10.55, '09': 20.32, '10': 16.67}
    self.data = self.load_ORB_SLAM2_data(data_path)


  def load_ORB_SLAM2_data(self, data_path):
    """
    Loads pre-generated ORB-SLAM2 output data.
    """
    data = {}
    # ORB-SLAM2 fails on sequence 01, sequence 03 has no inertial data
    for seq in ['00', '02', '04', '05', '06', '07', '08', '09', '10']:
      data[seq] = {}
      # load data
      with open(os.path.join(data_path, seq + '.txt')) as f:
        traj = [map(float, line.split()) for line in f.read().splitlines()]
      # store
      for pose in traj:
        pose_arr = np.asarray(pose)
        pose_arr[1:4] = pose_arr[1:4] * self.SCALES[seq]
        data[seq][pose[0]] = pose_arr[1:]
    return data


  def get_pose(self, seq, timestamp):
    """
    Return pose at timestamp for given sequence.
    """
    try:
      pose = self.data[seq][timestamp]
    except KeyError:
      pose = None
    return pose


class error_module():
  """
  Implementation of the Error Module as described.
  """

  def __init__(self, dt_th, dq_th):
    self.dt_th = dt_th
    self.dq_th = dq_th


  def filter(self, inertial_pose, visual_pose):
    """
    Ignore visual pose if error is too big.
    """
    out = [inertial_pose]
    if visual_pose is None:
      return out
    # Euclidean Distance
    print inertial_pose[0:3]
    print visual_pose[0:3]
    dt = np.linalg.norm(inertial_pose[0:3]-visual_pose[0:3])
    print dt
    if (dt > self.dt_th):
      return #out
    # https://math.stackexchange.com/questions/90081/quaternion-distance
    dq = np.arccos(2 * ((np.sum(np.multiply(inertial_pose[3:], visual_pose[3:])))**2) - 1)
    if (dq > self.dq_th):
      return out
    out.append(visual_pose)
    return out


"""
Helper functions
"""

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

