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
  return (data - data.mean())/data.std() 

def normalise_all(data):
  for i in range(NUM_FEATURE):
    data[:,i] = normalise(data[:,i])
  return data

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

  print(np.split(data, [int(train_split*data.shape[0]),int(test_split*data.shape[0]),int(val_split*data.shape[0])]))
  train,test,val = np.split(data, [int(train_split*data.shape[0]),int(test_split*data.shape[0]),int(val_split*data.shape[0])])
  
  print(test)
  
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
  pinv_x = np.linalg.pinv(data_x)
  return  np.matmul(pinv_x, data_y)

#########################################


### Get discrete values for output ###

def predict(x,weights):
  y = np.dot(x,weights)
  return  classify(y)

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

def normalise_y(data):
  mean = np.mean(data)
  std  = np.std(data)
  normaliser = lambda x: (x-mean)/std
  vfunc = np.vectorize(normaliser)
  return vfunc(data)
  
def un_normalise_y(data):
  mean  = np.mean(data)
  std   = np.std(data)
  normaliser = lambda x: x*std+mean
  vfunc = np.vectorize(normaliser)
  return vfunc(data)

def svd_reduction(data):
  u,s,vh = np.linalg.svd(data,full_matrices=False)

  #only take largest eigenvalue
  eig_max = np.max(s)
  
  print(eig_max)
  #print(s)

  for i in range(len(s)):
    if s[i] !=eig_max:
      s[i] = 0;

  #print(np.diag(s))
  
  print(u.shape)
  print(np.diag(s).shape)
  print(vh.shape)

  data =  np.matmul(np.matmul(u,np.diag(s)),vh)
  
  print(data)
  return data
 
def sk_lin_regr(data):

  #data = normalise(data) 

  train_x,train_y,test_x,test_y,val_x,val_y = split_data(data)

  train_x = svd_reduction(train_x)
  val_x   = svd_reduction(val_x)
  #train_y = normalise_y(train_y)

  regr = linear_model.LinearRegression(fit_intercept=True,normalize=True) 

  regr.fit(train_x,train_y)

  pred_y = regr.predict(val_x)
  #pred_y = un_normalise_y(pred_y)

  pred_y = classify(pred_y)

  print("linear regression accuracy: {}",accuracy_score(val_y,pred_y))

def add_bias(x):
  ones = np.array([[1] for i in range(len(x.shape(0)))])
  return np.concatenate((x,ones),axis=1)

def my_lin_regr(data):
  data = normalise(data)
  train_x,train_y,test_x,test_y,val_x,val_y = split_data(data)
  
  train_x = add_bias(train_x)
  val_x   = add_bias(val_x)
    
  weights = fit(train_x,train_y)
  pred_y = predict(val_x,train_y,weights)
  
  print("linear regression accuracy: {}",accuracy_score(val_y,pred_y))

def my_lin_regr(data):
  
  train_x,train_y,test_x,test_y,val_x,val_y = split_data(data)
  
  #normalise all x
  train_x = normalise_all(train_x)
  test_x  = normalise_all(test_x)
  val_x   = normalise_all(val_x)

  print('train_x shape: ',train_x.shape)
  print('test_x shape:  ',test_x.shape)
  print('val_x shape:   ',val_x.shape)

  #bias all x
  train_x = add_bias(train_x)   
  test_x  = add_bias(test_x)   
  val_x   = add_bias(val_x)   

  #get learned weight
  weights = fit(train_x,train_y)

  pred_y  = predict(val_x,weights)

  print("linear regression accuracy: {}",accuracy_score(val_y,pred_y))


if __name__=="__main__":
  data = get_data()
  #data = normalise(data)
  #train_x,train_y,test_x,test_y,val_x,val_y = split_data(data)
  #weights = fit(train_x,train_y)
  #predict_y = predict(val_x,train_y,weights)

  #sk_lin_regr(data)
  my_lin_regr(data)
  #plt.show()
