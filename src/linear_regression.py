from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

np.random.seed(0)

NUM_FEATURE = 12

def get_data(red=True):
  #get red wine data  
  red_data = np.genfromtxt('../data/winequality-red.csv',delimiter=';')
  red_data = np.delete(red_data,(0),axis=0)
  red_feature = np.array([[1 for i in range(red_data.shape[0])]])
  red_y = np.array([red_data[:,11]])
  red_data = np.append(red_data[:,0:11],red_feature.T,axis=1)
  red_data = np.append(red_data,red_y.T,axis=1)
  if red:
    return red_data
  #get white wine data
  white_data = np.genfromtxt('../data/winequality-white.csv',delimiter=';')
  white_data = np.delete(white_data,(0),axis=0)
  white_feature = np.array([[-1 for i in range(white_data.shape[0])]])
  white_y = np.array([white_data[:,11]])
  white_data = np.append(white_data[:,0:11],white_feature.T,axis=1)
  white_data = np.append(white_data,white_y.T,axis=1)
  #combine the 2
  data = np.append(red_data,white_data,axis=0)
  np.random.shuffle(data)
  return data


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
 
  for i in range(NUM_FEATURE):
    mean  = train_x[:,i].mean()
    std   = train_x[:,i].std() 
    
    #normalise training data
    
    #normalise test data  
    test[:,i] 
 
  return train_x,train_y,test_x,test_y

##### Loss Functions #####
def square_loss(y,pred_y):
  return ((y-pred_y)**2).mean()
  #N = len(y)
  #error_sum = 0
  #for i in range(N):
  #  error_sum += (y[i]-pred_y[i])**2
  #return error/N
def mae_loss(y,pred_y):
  return (y-pred_y).mean()
  
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

  print(train_x[0])

  regr = linear_model.LinearRegression() 
  regr.fit(train_x,train_y)
  pred_y = regr.predict(test_x)
  print("(sk) linear regression MAE: ",mean_absolute_error(test_y,pred_y))
  return regr

def sk_lasso_regr(data,alpha=0.1):
  #data = normalise(data) 
  train_x,train_y,test_x,test_y = split_data(data)
  regr = linear_model.Lasso(alpha) 
  regr.fit(train_x,train_y)
  pred_y = regr.predict(test_x)
  print("(sk) linear regression Lasso MAE: ",mean_absolute_error(test_y,pred_y))
  return regr

def sk_ridge_regr(data,alpha=0.1):
  #data = normalise(data) 
  train_x,train_y,test_x,test_y = split_data(data)
  regr = linear_model.Ridge(alpha) 
  regr.fit(train_x,train_y)
  pred_y = regr.predict(train_x)
  ''' 
  #tuning hyperparameter
  lmbda = 50 
  prev_err = cross_validation(train_x,train_y,regr);
  print("Previous ERROR: ",prev_err)
  alpha += lmbda*0.00001 
  for i in range(200):
    regr = linear_model.Ridge(alpha) 
    regr.fit(train_x,train_y)
    curr_err = cross_validation(train_x,train_y,regr);
    err_gradient = curr_err - prev_err
    print("Error: ",curr_err,", Alpha: ",alpha,", gradient: ",err_gradient)
    alpha = alpha - lmbda*err_gradient
  '''
  pred_y = regr.predict(test_x)
  print("(sk) linear regression Ridge MAE: ",mean_absolute_error(test_y,pred_y))
  return regr

def sk_elastic_regr(data,alpha=0.1, l1_ratio=0.5):
  #data = normalise(data) 
  train_x,train_y,test_x,test_y = split_data(data)
  regr = linear_model.ElasticNet(alpha, l1_ratio) 
  regr.fit(train_x,train_y)
  pred_y = regr.predict(test_x)
  pred_y = classify(pred_y)
  print("(sk) linear regression Elastic Net MAE: ",mean_absolute_error(test_y,pred_y))
  return regr

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
def cross_validation(train_x,train_y,model,k=10,loss=square_loss):
  #split into K folds
  data = np.concatenate((train_x,train_y),axis=1)
  kf = KFold(n_splits=k)

  error_sum = 0
  for train,test in kf.split(data):
    train_data  = np.array(data)[train]  
    test_data   = np.array(data)[test]  
    train_x, train_y  = np.hsplit(train_data,[NUM_FEATURE])
    test_x , test_y   = np.hsplit(test_data,[NUM_FEATURE])
    model.fit(train_x,train_y)
    pred_y = model.predict(test_x)
    error_sum += loss(test_y,pred_y)

  return error_sum/k    

if __name__=="__main__":
  data = get_data()

  model = sk_lin_regr(data) 
  #sk_ridge_regr(data,0.05)

'''   
  for i in range(20):
    alpha = pow(10,-(i-5))
    print("Alpha: ", alpha)
    #sk_lasso_regr(data,alpha)
    sk_ridge_regr(data,alpha)

  print("\n ////////////////////// \n")

  for i in range(20):
    for j in range(10):
      alpha = pow(10,-i)
      l1_ratio = 0.1*(j+1) 
      print("Alpha: ", alpha,", L1 Ratio: ",l1_ratio)
      sk_elastic_regr(data,alpha,l1_ratio)

  #my_lin_regr(data)
  #sk_svm(data)
  #tmp = lin_regr(data,'ridge')
  #tmp.cross_validation(loss=identity_loss)
'''  
  
