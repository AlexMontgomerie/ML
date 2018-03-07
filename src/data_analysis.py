from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUM_FEATURE = 11

def get_data():
  wine_data = np.genfromtxt('../data/winequality-red.csv',delimiter=';')
  #return np.delete(wine_data,(0),axis=0)
  return wine_data

def normalise(data):
  return (data - data.mean())/data.std() 

def normalise_all(data):
  for i in range(NUM_FEATURE):
    data[:,i] = normalise(data[:,i])
  return data

def get_correlation_matrix(data,fig_num):
  data = np.delete(data,-1,axis=1)
  
  df = pd.DataFrame(data=data)
  
  fig = plt.figure(fig_num)
  ax = fig.add_subplot(111)  
  cax = ax.matshow(df.corr())
  fig.colorbar(cax)
  return

def visualise_data(x,y,fig_num):
  sz = np.array([0.2 for x in range(len(y))])
  fig = plt.figure(fig_num)
  for i in range(NUM_FEATURE):
    ax = fig.add_subplot(2,6,i+1)
    x_norm = normalise(x[:,i])
    ax.scatter(x_norm,y,sz)  
    ax.set_ylim([0,10])
    ax.set_xlim(xmin=0)
        
  return

def get_data_plot(data,fig_num):
  data = np.delete(data,(0),axis=0)
   
  x,y = np.hsplit(data,[NUM_FEATURE])
  
  visualise_data(x,y,fig_num)
  return


def get_corr_coef(x,y):
  coef_array = np.corrcoef(np.transpose(x),np.transpose(y)) 
  return coef_array[0,1]
  

def get_data_output_correlation_plot(data,fig_num):
  data = np.delete(data,(0),axis=0)
  x,y = np.hsplit(data,[NUM_FEATURE])
  x = normalise_all(x)
  corr = np.array([get_corr_coef(x[:,i],y) for i in range(NUM_FEATURE)])
  x_axis = np.arange(NUM_FEATURE)

  plt.figure(fig_num)
  plt.bar(x_axis,corr)
  return

def get_output_dist(data,fig_num):
  data = np.delete(data,(0),axis=0)
  x,y = np.hsplit(data,[NUM_FEATURE])
 
  fig = plt.figure(fig_num)
  sns.distplot(np.transpose(y))

  return

def get_feature_dist(data,fig_num):
  data = np.delete(data,(0),axis=0)
  x,y = np.hsplit(data,[NUM_FEATURE])

  x = normalise_all(x)

  fig = plt.figure(fig_num)
  
  for i in range(NUM_FEATURE):
    ax = fig.add_subplot(2,6,i+1)
    sns.distplot(np.transpose(x[:,i]))
  
  return 

#TODO: find a way of removing outliers whilst preserving largest data set
def remove_outliers(data):
  pass

'''  
  x_ = [[x] for i in range(11)]
  for j in range(11): 
    for i in range(len(y)):
      if int(y[i]) != j:
        x_[j].delete(i,axis=0)
      x_ = np.append(x_[int(y[i])],x[i])
'''

if __name__=="__main__":
  data = get_data()
  #get_lin_regr(data)
  
  get_correlation_matrix(data,1)
  get_data_plot(data,2)
  get_data_output_correlation_plot(data,3)
  get_output_dist(data,4)  
  get_feature_dist(data,5)

  plt.show()

