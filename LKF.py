"""
This module implements the Linear Kalman Filter as described in:
Learning to fuse: A deep learning approach to visual-inertial camera pose estimation

Joost Hoppenbrouwer
"""

import numpy as np

class LKF():
  """
  Implementation of a Linear Kalman Filter used for visual-inertial pose estimation.
  """

  def __init__(self):
    self.state = np.zeros(7)
    self.p = np.zeros(7) # predicted state
    #self.S =  #TODO state covariance matrix
    self.F = np.eye(7) # state transition matrix
    #self.Q =  #TODO process noise covariance matrix
    self.H = np.eye(7) # observation matrix


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
    y = visual_pose - (np.dot(self.H, self.p)) # innovation
    E = np.dot(self.H, np.dot(self.S, self.H.T)) # innovation covariance
    G = np.dot(self.S, np.dot(self.H.T, np.linalg.inv(E))) # Kalman gain
    self.state = self.p + np.dot(G, y)
    self.S = np.dot((np.eye(7) - np.dot(G, self.H)), self.S)


  def step(self, poses):
    """
    Full step of the filter including prediction and update.
    """
    self.predict(poses[0])
    if poses[1] is None:
      self.state = self.p
    else:
      self.update(poses[1])
    return self.state
