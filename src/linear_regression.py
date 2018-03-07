from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
from sklearn.svm import SVR

NUM_FEATURE = 11

def get_data():
  wine_data = np.genfromtxt('../data/winequality-red.csv',delimiter=';')
  wine_data = np.delete(wine_data,(0),axis=0)
  return np.random.shuffle(wine_data)

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
def split_data(data,test_split=0.1,val_split=0.1,train_split=0.8):
  #train,test,val

  train,test = np.split(data, [int(train_split*data.shape[0])])
  
  train_x,train_y = np.hsplit(train,[NUM_FEATURE])
  test_x,test_y   = np.hsplit(test, [NUM_FEATURE])
  
  return train_x,train_y,test_x,test_y

################################################

### Compute Linear Regression Weights ###
def fit(data_x,data_y):
  #normalise y
  pinv_x = np.linalg.pinv(data_x)
  return  np.matmul(pinv_x, data_y)

def add_bias(x):
  ones = np.array([[1] for i in range(len(x[:,0]))])
  return np.concatenate((x,ones),axis=1)


def my_lin_regr(data):
  
  train_x,train_y,test_x,test_y = split_data(data)
  
  #normalise all x
  train_x = normalise_all(train_x)
  test_x  = normalise_all(test_x)

  train_x = train_x[:,[1,10]]
  test_x  = test_x[:,[1,10]]

  #bias all x
  train_x = add_bias(train_x)   
  test_x  = add_bias(test_x)   

  #get learned weight
  weights = fit(train_x,train_y)

  pred_y  = predict(test_x,weights)

  print("(my) linear regression accuracy: {}",accuracy_score(test_y,pred_y))

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

if __name__=="__main__":
  data = get_data()
  sk_lin_regr(data)
  my_lin_regr(data)
  #sk_svm(data)
