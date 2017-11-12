'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import math as math
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def obs_in_class(data,labels,k):
     class_idx = np.where(labels == k)
     class_labels = labels[class_idx]
     class_data = data[class_idx,][0]
     return(class_labels,class_data)
 
def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    train_data is a n x 64 array of data
    train_labels is a n  vector of training labels
    '''
    means = np.zeros((10, 64))
    for k in np.unique(train_labels):
        k= int(k)
        #get observations in class k
        class_labels , class_data = obs_in_class(train_data,train_labels,k)
        sum_k = np.sum(class_data,axis=0)
        avg_k = sum_k/class_data.shape[0] #compute the average feature value for points in the class 
        means[0,:] = avg_k

    return means

def compute_sigma_mles(train_data, train_labels, means):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    train_data is a n x 64 array of data
    train_labels is a n  vector of training labels
    means is a 10 x 64 numpy array of class means from compute_means_mles
    '''
    covariances = np.zeros((10, 64, 64))
    #compute separately for each k
    for k in np.unique(train_labels):
        k=int(k)
        #pull out the observations in class k
        class_labels , class_data = obs_in_class(train_data,train_labels,k)
        obs_diff = np.zeros((class_data.shape)).transpose()
        class_mean = means[k-1,:]
        #for each feature compute the difference between observations and the class mean 
        for i in np.arange(0,class_data.shape[0]-1):
            obs_data = class_data[i,:].transpose()
            diff = obs_data-class_mean
            obs_diff[:,i]=diff
        cov = np.dot(obs_diff,obs_diff.transpose())/class_data.shape[0]
        cov = cov + np.identity(64)*0.01 #add 0.01 to the diagonal for numerical stability
        covariances[k-1]=cov
    return covariances

def plot_cov_diagonal(covariances):
    '''
    covariances is a (10,64,64) numpy array of covariances from compute_sigma_mles
    '''
    # Plot the log-diagonal of each covariance matrix side by side
    cov_mat = np.zeros((10,8,8))
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        cov_diag_log = np.log(cov_diag)
        cov_diag_log = cov_diag_log.reshape(8,8)
        cov_mat[i] = cov_diag_log
    all_concat_cov = np.concatenate(cov_mat, 1)
    plt.figure(figsize=(20, 5))
    plt.imshow(all_concat_cov, cmap='gray')
    plt.savefig('q2_2_eta.png')

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    means is a 10 x 64 numpy array
    digits is a n by 64 numpy array
    covarainces is a 10 x 64 x 64 numpy array
    '''
    #first constant in Gaussian equation, common across classes
    const1 = (2*math.pi)**(-covariances[0].shape[0]/2)
    cond_lik = np.zeros((digits.shape[0],10))
   
    for k in range((10)):
        k = int(k)
        det = np.linalg.det(covariances[k])
        #second term in multivariate Gaussian
        const2 = 1/math.sqrt(det)
        cov_inv = np.linalg.inv(covariances[k])
        class_mean = means[k,:]
        for i in range(digits.shape[0]):
            data = digits[i,:]    
            diff = data-class_mean
            in_exp = -1/2*np.dot(np.dot(diff.transpose(),cov_inv),diff)
       #   sum the log of the constants and the likelihood
            likelihood = np.log(const1)+np.log(const2)+in_exp
            cond_lik[i,k] = likelihood
    return cond_lik

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    means is a 10 x 64 numpy array
    digits is a n by 64 numpy array
    covarainces is a 10 x 64 x 64 numpy array
    '''
    prior = np.log(1/10)
    gen_lik = generative_likelihood(digits,means,covariances)
  #  evidence is the sum of the likelihoods cross all classes
    evidence = np.sum(np.exp(gen_lik)*np.exp(prior),axis=1) #exponentiate the prior
    evidence_mat = np.repeat(evidence,10,axis=0)
    evidence_mat = evidence_mat.reshape(digits.shape[0],10)
    out = (gen_lik +prior)  - np.log(evidence_mat)
    return(out)
    

def right_class(x,k):
    k = int(k)
    if k == 0:
        k = 10
    return(x[k-1])

def avg_conditional_likelihood(digits, labels,means,covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    means is a 10 x 64 numpy array
    digits is a n by 64 numpy array
    covarainces is a 10 x 64 x 64 numpy array
    labels is a n x 1 vector of data labels
    '''
    #get log conditional likelihoods for all classes in all datapoints
    cond_likelihood = conditional_likelihood(digits, means,covariances)
    right_cond_lik = np.zeros((digits.shape[0]))
    for i in range(digits.shape[0]):
    #Only sum the likelihoods for the correct class
        right_cond_lik[i] = right_class(cond_likelihood[i,:],labels[i])
    right_cond_sum = np.sum(right_cond_lik)
    right_cond_mean = right_cond_sum/digits.shape[0]
    return right_cond_mean


def classification_accuracy(predicted_labels,eval_labels):
    '''
    computes accuracy given a set of prediced labels and truth labels
    predicted_labels is an n x 1 vector of predictions
    eval_labels is an n x 1 vector of true labels against which to evaluate the predictors
    '''
    #combine both into a data frame
    pred_truth = np.zeros(predicted_labels.shape[0],dtype={'names':('predicted','truth'),'formats':('int8','int8')})
    pred_truth['predicted'] = predicted_labels
    pred_truth['truth']= eval_labels
    #count rows where the labels match
    correct = pred_truth[pred_truth['predicted'] == pred_truth['truth']]
    return(correct.shape[0]/pred_truth.shape[0])

def best_class (x):
    best_k = np.argmax(x)+1
    if (best_k == 10):
        best_k = 0
    return(best_k)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    means is a 10 x 64 numpy array
    digits is a n by 64 numpy array
    covarainces is a 10 x 64 x 64 numpy array
    '''
    #get conditional likelihood
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    #select the most likely posterior class
    pred = np.apply_along_axis(best_class, 1 , cond_likelihood)
    
    return(pred)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
     # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels,means)
    plot_cov_diagonal(covariances)
    #get the average training and testing likelihoods for the correct class
    avg_train = avg_conditional_likelihood(train_data,train_labels,means,covariances)
    print("average training  conditional log likelihood", avg_train, "average training conditional likelihood", np.exp(avg_train) )
    avg_test = avg_conditional_likelihood(test_data,test_labels,means,covariances)
    print("average testing conditional log likelihood", avg_test, "average testing conditional likelihood", np.exp(avg_test))
    #Predict training labels and compute accuracy
    pred_train = classify_data(train_data,means,covariances)
    acc_train = classification_accuracy(pred_train,train_labels)
    #Predict testing labels and compute accuracy
    pred_test = classify_data(test_data,means,covariances)
    acc_test = classification_accuracy(pred_test,test_labels)
    print("training accuracy: ", acc_train, "testing accuract: ", acc_test)


if __name__ == '__main__':
    main()