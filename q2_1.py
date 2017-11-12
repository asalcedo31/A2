'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import preprocessing

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        self 
        test point ia a 1 x 64 numpy array of a new observation to classify
        k is the number of closest trianing points that vote on the class for the test point
        '''
        #get distance to the training points
        dist = self.l2_distance(test_point)
        labeled_dist = np.zeros(self.train_labels.shape[0],dtype={'names':('labels','dist'), 'formats':('int8','f4')})
        labeled_dist['labels']= self.train_labels
        labeled_dist['dist'] = dist
        #sort by distance
        labeled_dist = np.sort(labeled_dist,order='dist')
     #   if k = 1 just get the closest test point and return
        if k == 1:
            digit = labeled_dist['labels'][0]     
            return(digit)
        # if k >1 then get the top k points and count their class labels
        top_k_lab = labeled_dist['labels'][0:k]  
        unique_counts = np.asarray(np.unique(top_k_lab,return_counts=True))
         #   Assign the test point the class with the most votes, if tie then choose randomly from the tied classes
        top_counts = unique_counts[:,np.random.choice(np.ndarray.flatten(np.argwhere(unique_counts[1,:] == np.amax(unique_counts[1,:]))),size=1)]
           
        digit = top_counts[0]
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16),folds=10):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    train data is an n x 64 numpy array of observations
    train labels is a n x 1 array of labels for each observations
    k range is a tuple of the minimum and maximum+1 values of k to test
    folds is the number of fold to use
    '''
    #split data
    kf = KFold(n_splits=folds)
    k_acc = np.zeros((k_range.shape[0],folds))
    f = 0;
    # Loop over folds
    for train_index,test_index in kf.split(train_data):
        x_f_train, x_f_test = train_data[train_index], train_data[test_index]
        y_f_train, y_f_test = train_labels[train_index], train_labels[test_index]
        knn_f = KNearestNeighbor(x_f_train, y_f_train)
        print("running fold ", f)
        for k in k_range:
            k_acc[k-1,f-1]=classification_accuracy(knn_f,k,x_f_test,y_f_test)
        f +=1
        
    # obtain accuracy across folds
    k_acc_mean = np.mean(k_acc,axis=1)
   # best k has the highest accuracy
    best_k = np.argmax(k_acc_mean)+1
    print("best k", best_k)
    acc_df = pd.DataFrame({'k': k_range, 'Accuracy':k_acc_mean})
    acc_df = acc_df[['k','Accuracy']]
    acc_df_tex = acc_df.to_latex(index=False)
    print(acc_df)
    return(best_k)

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    knn is a knn object
    k is the desired number of k to use in the knn classifier
    eval data is a n x 64 numpy matrix of observations to classify
    eval_labels is a n x 1 vector of true labels
    '''
    # obtain predictions from knn classifier
    predicted_labels=knn_classify(knn,eval_data,k)
    # combine prediction into a single array
    pred_truth = np.zeros(predicted_labels.shape[0],dtype={'names':('predicted','truth'),'formats':('int8','int8')})
    pred_truth['predicted'] = predicted_labels
    pred_truth['truth']= eval_labels
    #count the rows where predicted and true labels match
    correct = pred_truth[pred_truth['predicted'] == pred_truth['truth']]
    return(correct.shape[0]/pred_truth.shape[0])

def knn_classify(knn,test_data,k):
    '''
    Assigns a label to test data using a knn model
    knn is a knn object
    test_data is a n x 64 numpy array of observations
    k is the desired k to use to assign labels in the knn object
    '''
    predicted_labels=np.zeros(test_data.shape[0])
    #iterate over test data
    for i in np.arange(0,test_data.shape[0]-1):
        predicted_labels[i]=knn.query_knn(test_data[i],k)
    return(predicted_labels)
def normalize(data,scaler=None): 
   '''
   normalize features so noisy features don't 
   influence results. If a scaling object is provided it will use it to scale, otherwise it will
   scale based on the mean and variance of the input dataset.
   Inputs: dataset is a N x k matrix to scale
   scaler: a sklearn standard scaler object
   Outputs: scaled data, if scaler not provided,  also outputs the sklearn scaler object used to scale the data
   '''
   if scaler == None:
        #scale data and save scaling parameters
        scaler = preprocessing.StandardScaler().fit(data)
        return(scaler.transform(data),scaler)
   else:  #if using input scaling
        print("using input scaling\n")
        return(scaler.transform(data))
    
def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, scaler = normalize(train_data)
    test_data = normalize(test_data,scaler)
    knn = KNearestNeighbor(train_data, train_labels)
    # accuracy for train and test data when k=1 and k=15
    accuracy_test_k1 = classification_accuracy(knn,1,test_data,test_labels)
    accuracy_train_k1 = classification_accuracy(knn,1,train_data,train_labels)
    accuracy_test_k15 = classification_accuracy(knn,15,test_data,test_labels)
    accuracy_train_k15 = classification_accuracy(knn,15,train_data,train_labels)
    print("test k1:", accuracy_test_k1, "train k1:", accuracy_train_k1, "test k15:", accuracy_test_k15, "train k15:", accuracy_train_k15)
   #find the optimal k with cross valudation 
    best_k = cross_validation(train_data,train_labels,np.arange(1,16),10)
    #report testing and training accuracy for the best k
    accuracy_test_best = classification_accuracy(knn,best_k,test_data,test_labels)
    accuracy_train_best = classification_accuracy(knn,best_k,train_data,train_labels)
    print("test k best:", accuracy_test_best, "train k best:", accuracy_train_best)
  #  print(predicted_labels[1:5])

if __name__ == '__main__':
    main()