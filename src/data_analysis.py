from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import pandas as pd

NUM_FEATURE = 11

def get_data():
  wine_data = np.genfromtxt('../data/winequality-red.csv',delimiter=';')
  #return np.delete(wine_data,(0),axis=0)
  return wine_data

def get_correlation_matrix(data):
  data = np.delete(data,-1,axis=1)
  
  df = pd.DataFrame(data=data)
  
  fig = plt.figure()
  ax = fig.add_subplot(111)  
  cax = ax.matshow(df.corr())
  fig.colorbar(cax)
  return

def correlation_plot(x,y):
    

def visualise_data(x,y):
  sz = np.array([0.2 for x in range(len(y))])
  fig = plt.figure()
  for i in range(NUM_FEATURE):
    ax = fig.add_subplot(111)
    plt.scatter(np.transpose(x[:,i]),y,sz)  
  return

def get_data_plot(data):
  #remove 
  data = np.delete(data,(0),axis=0)
  x,y = np.hsplit(data,[NUM_FEATURE])
  visualise_data(x,y)
  return

def get_data_output_correlation_plot(data):

if __name__=="__main__":
  data = get_data()
  get_correlation_matrix(data)
  get_data_plot(data)

  plt.show()

