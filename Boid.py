import numpy as np
from sklearn.neighbors import BallTree


class Boid():

  def main(flock_pos, target, FieldOfView, WeightCenterOfMass, WeightSeparation, MinSeparation, WeightAvoidance, AvoidanceTolerance, FeatureImportance, ObedienceValues):

    def get_same_class_idx(inst,boid):
      return inst[np.where(target[inst]==boid)[0]]

    def get_diff_class_idx(inst,boid):
      return inst[np.where(target[inst]!=boid)[0]]

    def get_same_class_dist(inst,dist,boid):
      return dist[np.where(target[inst]==boid)[0]]

    def get_diff_class_dist(inst,dist,boid):
      return dist[np.where(target[inst]!=boid)[0]]

    tree = BallTree(flock_pos, leaf_size=10)
    inst_dist, inst_idx = tree.query(flock_pos, k=int(flock_pos.shape[0]*FieldOfView))
    inst_idx = inst_idx[:,1:]
    inst_dist = inst_dist[:,1:]

    same_class_idx = list(map(get_same_class_idx, inst_idx, target))
    diff_class_idx = list(map(get_diff_class_idx, inst_idx, target))
    same_class_dist = list(map(get_same_class_dist, inst_idx, inst_dist, target))
    diff_class_dist = list(map(get_diff_class_dist, inst_idx, inst_dist, target))

    for boid in range(flock_pos.shape[0]):

      # Rule 1: Cohesion
      rule_1 = (np.mean(flock_pos[same_class_idx[boid],:],axis=0) - flock_pos[boid,:])*((FeatureImportance*WeightCenterOfMass)*ObedienceValues[boid])

      # Rule 2: Seperation
      idx = np.where(same_class_dist[boid]<MinSeparation)[0]
      differences = flock_pos[same_class_idx[boid][idx],:] - flock_pos[boid,:]
      distances = same_class_dist[boid][idx]
      rule_2 = np.zeros(flock_pos.shape[1]) - sum((differences/distances[:,np.newaxis])/distances[:,np.newaxis])
      rule_2 = rule_2 * ((FeatureImportance*WeightSeparation)*ObedienceValues[boid])

      # Rule 3: Avoidance
      idx = np.where(diff_class_dist[boid]<AvoidanceTolerance)[0]
      differences = flock_pos[diff_class_idx[boid][idx],:] - flock_pos[boid,:]
      distances = diff_class_dist[boid][idx]
      rule_3 = np.zeros(flock_pos.shape[1]) - sum((differences/distances[:,np.newaxis])/distances[:,np.newaxis])
      rule_3 = rule_3 * ((FeatureImportance*WeightAvoidance)*ObedienceValues[boid])

      flock_vel = rule_1 + rule_2 + rule_3
      flock_pos[boid,:] = flock_pos[boid,:] + flock_vel
    return flock_pos