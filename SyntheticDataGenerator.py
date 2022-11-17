import numpy as np
from Objectives import Objectives
from Boid import Boid
from pymop.problem import Problem
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skewnorm


class SyntheticDataGenerator(Problem):

  def __init__(self, X, y, NumberOfBoids, Dimensions, Classes, F1_Score, N1_Score,
               Mimic_Classes, Mimic_DataTypes, Mimic_Dataset):
    
    self.NumberOfBoids = X.shape[0]
    self.Dimensions = X.shape[1]
    self.Classes = np.unique(y).shape[0]

    self.F1_Score = F1_Score
    self.N1_Score = N1_Score

    self.X = X
    self.y = y

    self.Mimic_Classes = Mimic_Classes
    self.Mimic_DataTypes = Mimic_DataTypes
    self.Mimic_Dataset = Mimic_Dataset

    super().__init__(n_var=8, n_obj=2, n_constr=0, 
                     xl=np.array([0,0,0,0,0,0,-100,-100]), 
                     xu=np.array([1,1,1,0.50,1,0.50,100,100]), 
                     evaluation_of="auto")

  def _evaluate(self, x, out, *args, **kwargs):
    
    x = x[0]
    ll=np.array([0,0,0,0,0,0,-100,-100])
    ul=np.array([1,1,1,0.50,1,0.50,100,100])

    x[3] = ll[3] + (ul[3] * x[3])
    x[5] = ll[5] + (ul[5] * x[5])
    x[6] = ll[6] + (ul[6] * x[6])
    x[7] = ll[7] + (ul[7] * x[7])

    # Simulation Parameters
    timesteps = 15

    #initialise X and y
    if self.Mimic_Dataset:
      positions = np.zeros((self.NumberOfBoids,self.Dimensions,timesteps))
      positions[:,:,0] = self.X
      target = self.y
    elif self.Mimic_Classes:
      np.random.seed(0)
      positions = np.zeros((self.NumberOfBoids,self.Dimensions,timesteps))
      positions[:,:,0] = np.random.uniform(0,1,(self.NumberOfBoids,self.Dimensions))
      target = self.y
    else:
      np.random.seed(0)
      positions = np.zeros((self.NumberOfBoids,self.Dimensions,timesteps))
      positions[:,:,0] = np.random.uniform(0,1,(self.NumberOfBoids,self.Dimensions))
      target = np.random.randint(0,self.Classes,self.NumberOfBoids)
    
    if self.Mimic_DataTypes:
      correct_discrete = np.sum(self.X,axis=0) % 1 == 0
    
    #initialise obedience values
    ObedienceValues = skewnorm.rvs(x[6], size=self.NumberOfBoids, random_state=42)
    ObedienceValues = 1 - ((ObedienceValues-np.min(ObedienceValues))/((np.max(ObedienceValues))-np.min(ObedienceValues)))

    #Inititalise Feature Importance
    FeatureImportance = skewnorm.rvs(x[7], loc=1, scale=0.5, size=self.Dimensions, random_state=42)
    FeatureImportance[FeatureImportance<0] = 0
    FeatureImportance[FeatureImportance>1] = 1

    try:
      # simulate boids
      for i in range(1,timesteps):
        positions[:,:,i] = Boid.main(positions[:,:,i-1], target, x[0],x[1],x[2],x[3],x[4],x[5],FeatureImportance,ObedienceValues)
      
      if self.Mimic_DataTypes:
        for i in range(correct_discrete.shape[0]):
          if correct_discrete[i]:
            X_fake = MinMaxScaler().fit_transform(positions[:,i,-1].reshape(-1, 1))
            ul = max(self.X[:,i])
            ll = min(self.X[:,i])
            scaled = ll + (ul * X_fake).reshape(1,-1)
            positions[:,i,-1] = np.round(scaled,0)
      
      # evaluate simulation
      f1_ = Objectives.F1(positions[:,:,-1], target)
      n1_ = Objectives.N1(positions[:,:,-1], target)
      f1 = abs(self.F1_Score - f1_)
      n1 = abs(self.N1_Score - n1_)

      if np.nan in [f1,n1]:
        f1 = 1
        n1 = 1
          
    except:
      f1 = 1
      n1 = 1

    out["F"] = np.column_stack([f1,n1])