from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns

NUM_FEATURE = 11

def get_data():
  wine_data = np.genfromtxt('../data/winequality-red.csv',delimiter=';')
  return np.delete(wine_data,(0),axis=0)

#######################

### Normalise data ###
def normalise(data):
  data = np.transpose(data)

  #iterate through each column
  for i in range(NUM_FEATURE):
    #remove bias, and only leave unit variance
    mean  = np.mean(data[i])
    std   = np.std(data[i])
    normaliser = lambda x : (x-mean)/std
    vfunc = np.vectorize(normaliser)
    #update with normalised feature
    data[i] = vfunc(data[i])

  return np.transpose(data)

def normalise_single(data):
  data = np.transpose(data)

  #remove bias, and only leave unit variance
  mean  = np.mean(data)
  std   = np.std(data)
  normaliser = lambda x : (x-mean)/std
  vfunc = np.vectorize(normaliser)
  #update with normalised feature
  data = vfunc(data)

  return np.transpose(data)


######################

### Split Training, Test and Validation sets ###
def split_data(data,test_split=0.1,val_split=0.1,train_split=0.8):
  #train,test,val
  train,test,val = np.split(data, [int(train_split*data.shape[0]),int(test_split*data.shape[0])])

  train_x,train_y = np.hsplit(train,[NUM_FEATURE])
  test_x,test_y   = np.hsplit(test, [NUM_FEATURE])
  val_x,val_y     = np.hsplit(val,  [NUM_FEATURE])
  
  return train_x,train_y,test_x,test_y,val_x,val_y
#sns.distplot(np.transpose(train_x[:,0]))
#sns.distplot(np.transpose(train_y))

################################################

### Compute Linear Regression Weights ###
def fit(data_x,data_y):
  #normalise y
  data_y = normalise_single(data_y)
  pinv_x = np.linalg.pinv(data_x)
  return  np.matmul(pinv_x, data_y)

#########################################


### Get discrete values for output ###

def predict(x,y,weights):
  y_mean  = np.mean(y)
  y_std   = np.std(y)
  h_y = np.zeros(len(y))
  for i in range(len(y)):
    #get temporary result
    tmp = np.dot(x[i],np.transpose(weights))
    #loop through all output classes
    tmp = tmp*y_std + y_mean
    for j in range(11):
      if tmp > (j-0.5) and tmp <= (j+0.5):
        h_y[i] = j
    
    if h_y[i] <0 or h_y[i] > 10:
      h_y[i] = 0
    
  return np.transpose(h_y)

def classify(y):
  for i in range(len(y)):
    for j in range(11):
      if y[i] > (j-0.5) and y[i] <= (j+0.5):
        y[i] = j
    if y[i] < 0:
        y[i] = 0
    if y[i] > 10:
        y[i] = 10
  return y

def visualise_data(x,y):
  sz = np.array([0.2 for x in range(len(y))])
  for i in range(NUM_FEATURE):
    plt.scatter(np.transpose(x[:,i]),y,sz)  
  return

def get_cov_matrix(x):
  return np.cov(np.transpose(train_x))

def sk_lin_regr(data):

  data = normalise(data)

  train_x,train_y,test_x,test_y,val_x,val_y = split_data(data)

  regr = linear_model.LinearRegression(fit_intercept=False) 

  regr.fit(train_x,train_y)

  pred_y = regr.predict(val_x)

  pred_y = classify(pred_y)

  print(np.max(pred_y))
  
  print(accuracy_score(val_y,pred_y))
 
if __name__=="__main__":
  data = get_data()
  #data = normalise(data)
  #train_x,train_y,test_x,test_y,val_x,val_y = split_data(data)
  #weights = fit(train_x,train_y)
  #predict_y = predict(val_x,train_y,weights)

  sk_lin_regr(data)

  #plt.show()
