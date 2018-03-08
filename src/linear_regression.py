from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import KFold

NUM_FEATURE = 11

def get_data():
  wine_data = np.genfromtxt('../data/winequality-red.csv',delimiter=';')
  return np.delete(wine_data,(0),axis=0)
  #return np.random.shuffle(wine_data)

#######################

### Normalise data ###
def normalise(data):
  return (data - data.mean())/data.std() 

def normalise_all(data):
  for i in range(NUM_FEATURE):
    data[:,i] = normalise(data[:,i])
  return data

######################

### Split Training, Test and Validation sets ###
def split_data(data,train_split=0.8):
  #train,test,val

  train,test = np.split(data, [int(train_split*data.shape[0])])
  
  train_x,train_y = np.hsplit(train,[NUM_FEATURE])
  test_x,test_y   = np.hsplit(test, [NUM_FEATURE])
  
  return train_x,train_y,test_x,test_y

##### Loss Functions #####
def square_loss(y,pred_y):
  return ((y-pred_y)**2).mean()
  #N = len(y)
  #error_sum = 0
  #for i in range(N):
  #  error_sum += (y[i]-pred_y[i])**2
  #return error/N
  
def identity_loss(y,pred_y):
  N = len(y)
  error_sum = 0
  for i in range(N):
    if y[i] != pred_y[i]:
      error_sum += 1
  return error_sum/N

##########################

################################################
class lin_regr:
  def __init__(self,data,reguliser):
    self.data = data
    self.train_x = np.array([])
    self.train_y = np.array([])
    self.test_x  = np.array([]) 
    self.test_y  = np.array([])
    self.weights = np.array([])
    self.reguliser = 'ridge'
    self.init_data()
  
  def split_data(self,train_split=0.8):
    #TODO: randomise data split
    train,test = np.split(self.data, [int(train_split*self.data.shape[0])])
    #update datasets
    self.train_x,self.train_y = np.hsplit(train,[NUM_FEATURE])
    self.test_x,self.test_y   = np.hsplit(test, [NUM_FEATURE])
  
  def init_data(self):
    self.split_data()
    #normalise all x
    self.train_x = normalise_all(self.train_x)
    self.test_x  = normalise_all(self.test_x)

    #bias all x
    self.train_x = self.add_bias(self.train_x)   
    self.test_x  = self.add_bias(self.test_x)   

  def fit(self,x,y,l=0.001):
    if self.reguliser=='ridge':
      tmp = np.matmul(x.T,x) + np.diag([l for i in range(NUM_FEATURE+1)])
      tmp = np.linalg.pinv(tmp)
      tmp = np.matmul(tmp,x.T)
      self.weights = np.matmul(tmp,y)
    else:
      pinv_x = np.linalg.pinv(x)
      self.weights = np.matmul(pinv_x,y)

  def add_bias(self,x):
    ones = np.array([[1] for i in range(len(x[:,0]))])
    return np.concatenate((x,ones),axis=1)

  def predict(self,x):
    y = np.dot(x,self.weights)
    return  classify(y)

  def classify(self,y):
    for i in range(len(y)):
      for j in range(11):
        if y[i] > (j-0.5) and y[i] <= (j+0.5):
          y[i] = j
        if y[i] < 0:
          y[i] = 0
        if y[i] > 10:
          y[i] = 10
    return y

  def cross_validation(self,k=10,loss=square_loss):
    #split into K folds
    data = np.concatenate((self.train_x,self.train_y),axis=1)
    kf = KFold(n_splits=k)

    error_sum = 0
    for train,test in kf.split(data):
      train_data  = np.array(data)[train]  
      test_data   = np.array(data)[test]  
      train_x, train_y  = np.hsplit(train_data,[NUM_FEATURE+1])
      test_x , test_y   = np.hsplit(test_data,[NUM_FEATURE+1])
      self.fit(train_x,train_y)
      pred_y = self.predict(test_x)
      pred_y = self.classify(pred_y) 
      error_sum += loss(test_y,pred_y)

    print("cross validation error: ",error_sum/k)
    return error_sum/k    


### Compute Linear Regression Weights ###
def fit(x,y,ridge=False,l=0.001):
  if ridge:
    tmp = np.matmul(np.transpose(x),x) + np.diag([l for i in range(NUM_FEATURE+1)])
    tmp = np.linalg.pinv(tmp)
    tmp = np.matmul(tmp,x.T)
    return np.matmul(tmp,y)
  else:
    pinv_x = np.linalg.pinv(x)
    return  np.matmul(pinv_x,y)

def add_bias(x):
  ones = np.array([[1] for i in range(len(x[:,0]))])
  return np.concatenate((x,ones),axis=1)

def my_lin_regr(data):
  
  train_x,train_y,test_x,test_y = split_data(data)
  
  #normalise all x
  train_x = normalise_all(train_x)
  test_x  = normalise_all(test_x)

  #train_x = train_x[:,[1,10]]
  #test_x  = test_x[:,[1,10]]

  #bias all x
  train_x = add_bias(train_x)   
  test_x  = add_bias(test_x)   

  #get learned weight
  weights = fit(train_x,train_y,ridge=True)

  cross_validation(train_x,train_y,weights)

  pred_y  = predict(train_x,weights)

  print("(my) linear regression accuracy: ",accuracy_score(train_y,pred_y))

def bin_classify_y(y,val):
  
  tmp = np.array([[0] for i in range(len(y))])

  for i in range(len(y)):
    if y[i][0] >= val:
      tmp[i][0] =  1
    else:
      tmp[i][0] = -1
  
  return tmp

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

def svd_reduction(data):
  u,s,vh = np.linalg.svd(data,full_matrices=False)
  data =  np.matmul(np.matmul(u,np.diag(s)),vh)
  return data
 
#########################################

def sk_lin_regr(data):

  #data = normalise(data) 

  train_x,train_y,test_x,test_y = split_data(data)

  regr = linear_model.LinearRegression(fit_intercept=True,normalize=True) 

  regr.fit(train_x,train_y)

  pred_y = regr.predict(test_x)

  pred_y = classify(pred_y)

  print("(sk) linear regression accuracy: {}",accuracy_score(test_y,pred_y))

def sk_svm(data):
  #split the data  
  train_x,train_y,test_x,test_y = split_data(data)

  #normalise all x
  train_x = normalise_all(train_x)
  test_x  = normalise_all(test_x)

  print(bin_classify_y(test_y,6))

  #define SVR 
  clf = SVR(C=1.0, epsilon=0.2)
  clf.fit(train_x,train_y)

  #predict y using SVC  
  pred_y = clf.predict(test_x)

  pred_y = classify(pred_y)

  print("(sk) support vector machine for regression accuracy: {}",accuracy_score(test_y,pred_y))

  return

##### Validation #####

#TODO: create K-fold cross-validation function
def cross_validation(x,y,weights,k=10,loss=square_loss):
  #split into K folds
  data = np.concatenate((x,y),axis=1)
  kf = KFold(n_splits=k)

  for train,test in kf.split(data):
    train_data = np.array(data)[train]  

if __name__=="__main__":
  data = get_data()
  #sk_lin_regr(data)
  my_lin_regr(data)
  #sk_svm(data)
  tmp = lin_regr(data,'ridge')
  tmp.cross_validation(loss=identity_loss)

  train_x,train_y,test_x,test_y = split_data(data)
