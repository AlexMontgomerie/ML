from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

NUM_FEATURE = 11

def get_data():
  wine_data = np.genfromtxt('../data/winequality-red.csv',delimiter=';')
  #return np.delete(wine_data,(0),axis=0)
  return wine_data

def bin_classify_y(y,val):
  
  tmp = np.array([[0] for i in range(len(y))])

  for i in range(len(y)):
    if y[i][0] >= val:
      tmp[i][0] =  1
    else:
      tmp[i][0] = -1
  
  return tmp

def predict_y(y,clf):
  #split into 2 branches
  
  #if result of predictor 5 > 1, check 5+3, other wise 3

  #if result of 8 or 3 > 1, check 8+1 or 3+1, or 8 - 2, or 3 - 2

def get_SVM(data):
  #split data appropriately
  X,Y_top = np.hsplit(data,[NUM_FEATURE])
  Y_top = [bin_classify_y(Y_top,i) for i in range(11)]
  #Y = bin_classify_y(Y,5.0)
  for Y in Y_top:
    print(Y)
    Y = Y.ravel()
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2,random_state=0)
    #define model parameters
  
    parameters = [{'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[0.001,0.01,0.1,1]}] 
  #create the model
    clf = GridSearchCV(SVC(),parameters,cv=10,scoring='accuracy',verbose=3)
  #fit parameters
    clf.fit(train_x,train_y)
    print(clf.best_params_)
  #predict y
    pred_y = clf.predict(test_x)
    print(classification_report(test_y, pred_y))
  return

if __name__ == "__main__":
  data = get_data()
  data = np.delete(data,(0),axis=0)
  get_SVM(data)
