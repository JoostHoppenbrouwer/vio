"""
This module implements the Linear Kalman Filter as described in:
Learning to fuse: A deep learning approach to visual-inertial camera pose estimation

Joost Hoppenbrouwer
"""

import sys
import numpy as np

ORB_VAR = [0.0878775315, 0.000597190384, 0.110587757, 0.00000108883552,
           0.00253643937, 0.00000254709930, 0.000853188441]

class LKF():
  """
  Implementation of a Linear Kalman Filter used for visual-inertial pose estimation.
  """

  def __init__(self, mov_noise_std=0.0, rot_noise_std=0.0):
    self.state = np.append(np.zeros(6), 1)  # initial state
    self.p = np.zeros(7)      # predicted state
    self.S = np.zeros([7,7])  # state covariance matrix
    self.F = np.eye(7)        # state transition matrix
    # process noise covariance matrix --> should match the variance of the Gaussian noise in ground truth
    Q_mov = np.asarray([mov_noise_std**2]*3)
    Q_rot = np.asarray([rot_noise_std**2]*4)
    self.Q = np.diag(np.append(Q_mov, Q_rot)**2)
    self.R = np.diag(ORB_VAR)
    self.H = np.eye(7)        # observation matrix


  def get_state(self):
    return self.state


  def predict(self, inertial_pose):
    """
    Prediction step of the filter.
    """
    self.p = np.dot(self.F, inertial_pose)
    self.S = np.dot(self.F, np.dot(self.S, self.F.T)) + self.Q


  def update(self, visual_pose):
    """
    Update step of the filter.
    """
    y = visual_pose - (np.dot(self.H, self.p))              # innovation
    E = np.dot(self.H, np.dot(self.S, self.H.T)) + self.R   # innovation covariance
    # try to inverse E, catch the case where E is np.zeros([7,7])
    try:
      E_inv = np.linalg.inv(E)
    except np.linalg.linalg.LinAlgError as err:
      if 'Singular matrix' in err.message:
        print "!!! Innovation Covariance is singular !!!"
        E_inv = np.zeros([7,7])
      else:
        raise
    G = np.dot(self.S, np.dot(self.H.T, E_inv))             # Kalman gain
    self.state = self.p + np.dot(G, y)                      # final state
    self.S = np.dot((np.eye(7) - np.dot(G, self.H)), self.S)


  def step(self, inertial_pose, visual_pose):
    """
    Full step of the filter including prediction and update.
    """
    self.predict(inertial_pose)
    if visual_pose is None:
      self.state = self.p
    else:
      self.update(visual_pose)
    return self.state
