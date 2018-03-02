from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

### Get Input Data ###
wine_data = np.genfromtxt('../data/winequality-red.csv',delimiter=';')
#remove first row
wine_data = np.delete(wine_data,(0),axis=0)

#######################

### Normalise data ###

wine_data = np.transpose(wine_data)

#iterate through each column
for i in range(11):
  print(i)
  #remove bias, and only leave unit variance
  mean  = np.mean(wine_data[i])
  std   = np.std(wine_data[i])
  normaliser = lambda x : (x-mean)/std
  vfunc = np.vectorize(normaliser)
  wine_data[i] = vfunc(wine_data[i])

wine_data = np.transpose(wine_data)

######################

### Split Training, Test and Validation sets ###

test_split = 0.1
val_split = 0.1
train_split = 1 - test_split - val_split

train,test,val = np.split(wine_data, [int(train_split*wine_data.shape[0]),int(test_split*wine_data.shape[0])])

train_x,train_y = np.hsplit(train,[11])
test_x,test_y   = np.hsplit(test,[11])
val_x,val_y     = np.hsplit(val,[11])

#sns.distplot(np.transpose(train_x[:,0]))
#sns.distplot(np.transpose(train_y))

################################################

### Compute Linear Regression Weights ###

pinv_x = np.linalg.pinv(train_x)
weights = np.matmul(pinv_x, train_y)

#########################################

### Get discrete values for output ###

h_y = np.zeros(len(train_y))

for i in range(len(train_y)):
  #get temporary result
  tmp = np.dot(np.transpose(weights),train_x[i])
  h_y[i] = np.dot(np.transpose(weights),train_x[i])
  #loop through all output classes
  for j in range(11):
    if tmp > (j-0.5) and tmp <= (j+0.5):
      pass
      #h_y[i] = j

h_y = np.transpose(h_y)

sz = np.array([0.2 for x in range(len(h_y))])

print(np.cov(np.transpose(train_x)))

plt.scatter(np.transpose(train_x[:,0]),h_y,sz)
plt.scatter(np.transpose(train_x[:,1]),h_y,sz)
plt.scatter(np.transpose(train_x[:,2]),h_y,sz)
plt.scatter(np.transpose(train_x[:,3]),h_y,sz)
plt.scatter(np.transpose(train_x[:,4]),h_y,sz)
plt.scatter(np.transpose(train_x[:,5]),h_y,sz)
plt.scatter(np.transpose(train_x[:,6]),h_y,sz)
plt.scatter(np.transpose(train_x[:,7]),h_y,sz)
plt.scatter(np.transpose(train_x[:,8]),h_y,sz)
plt.scatter(np.transpose(train_x[:,9]),h_y,sz)
plt.scatter(np.transpose(train_x[:,10]),h_y,sz)

######################################

### Get Errors ###

val_error = np.zeros(len(val_y))

for i in range(len(val_y)):
  val_error[i] = np.linalg.norm(np.dot(np.transpose(weights),val_x[i]) - val_y[i])

##################

plt.show()

print('error: ',np.mean(val_error)*100)

#iterate thro

