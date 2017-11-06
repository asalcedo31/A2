'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

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
        '''
        dist = self.l2_distance(test_point)
        labeled_dist = np.zeros(7000,dtype={'names':('labels','dist'), 'formats':('int8','f4')})
        labeled_dist['labels']= self.train_labels
        labeled_dist['dist'] = dist
        labeled_dist = np.sort(labeled_dist,order='dist')
     #   print(labeled_dist[0])
        if k == 1:
            digit = labeled_dist['labels'][0]     
            return(digit)
        top_k_lab = labeled_dist['labels'][0:k]
     #   print(top_k_lab)    
        if k == 1:
            digit = labeled_dist['labels':0]        
        unique_counts = np.asarray(np.unique(top_k_lab,return_counts=True))
      #  unique_counts = np.sort(unique_counts.T,axis=1)
        print(unique_counts)
        print(np.ndarray.flatten(np.argwhere(unique_counts[1,:] == np.amax(unique_counts[1,:]))))
        top_counts = unique_counts[:,np.random.choice(np.ndarray.flatten(np.argwhere(unique_counts[1,:] == np.amax(unique_counts[1,:]))),size=1)]
       # print(top_counts.shape)
        print(top_counts)
     #   if top_counts.shape[0] > 1:
            
        digit = top_counts[0]
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        pass

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    predicted_labels=knn_classify(knn,eval_data,k)
    pred_truth = np.zeros(predicted_labels.shape[0],dtype={'names':('predicted','truth'),'formats':('int8','int8')})
    pred_truth['predicted'] = predicted_labels
    pred_truth['truth']= eval_labels
    print(pred_truth.shape)
    correct = pred_truth[pred_truth['predicted'] == pred_truth['truth']]
    return(correct.shape[0]/pred_truth.shape[0])

def knn_classify(knn,test_data,k):
    predicted_labels=np.zeros(test_data.shape[0])
    for i in np.arange(0,test_data.shape[0]-1):
        predicted_labels[i]=knn.query_knn(test_data[i],k)
    return(predicted_labels)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)
    np.random.seed()
    # Example usage:
    predicted_label = knn.query_knn(test_data[0], 2)
    print("predicted_label:",predicted_label)
    predicted_label = knn.query_knn(test_data[9], 2)
    print("predicted_label:",predicted_label)
 #   accuracy_test_k1 = classification_accuracy(knn,1,test_data,test_labels)
   # accuracy_train_k1 = classification_accuracy(knn,1,train_data,train_labels)
    #accuracy_test_k15 = classification_accuracy(knn,15,test_data,test_labels)
    #accuracy_train_k15 = classification_accuracy(knn,15,train_data,train_labels)
    #print("test k1:", accuracy_test_k1, "train k1:", accuracy_train_k1, "test k15:", accuracy_test_k15, "train k15:", accuracy_train_k15)

  #  print(predicted_labels[1:5])

if __name__ == '__main__':
    main()