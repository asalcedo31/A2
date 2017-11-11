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
    '''
    means = np.zeros((10, 64))
    for k in np.unique(train_labels):
        k= int(k)
        class_labels , class_data = obs_in_class(train_data,train_labels,k)
        sum_k = np.sum(class_data,axis=0)
        avg_k = sum_k/class_data.shape[0]
        means[0,:] = avg_k
        #   means=np.insert(means,0,avg_k,axis=0)
        print(avg_k)
        print(means[(k-1),:])

    # Compute means
    return means

def compute_sigma_mles(train_data, train_labels, means):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    for k in np.unique(train_labels):
        print("k: ", k)
        k=int(k)
        class_labels , class_data = obs_in_class(train_data,train_labels,k)
        print(class_labels.shape)
        print(class_data.shape)
        obs_diff = np.zeros((class_data.shape)).transpose()
        print(obs_diff.shape)
        class_mean = means[k-1,:]
        print(class_mean.shape)
        for i in np.arange(0,class_data.shape[0]-1):
            obs_data = class_data[i,:].transpose()
            diff = obs_data-class_mean
            obs_diff[:,i]=diff
        print(obs_diff[0:3,])
     #   cov = np.cov(class_data,rowvar=False)
        cov = np.dot(obs_diff,obs_diff.transpose())/class_data.shape[0]
        cov = cov + np.identity(64)*0.01
     ##   print(cov.shape)
        covariances[k-1]=cov
            
            
    # Compute covariances
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov_mat = np.zeros((10,8,8))
    for i in range(10):
        print(i)
        cov_diag = np.diag(covariances[i])
        print(covariances[i][5:9,5:9])
        cov_diag_log = np.log(cov_diag)
        cov_diag_log = cov_diag_log.reshape(8,8)
        cov_mat[i] = cov_diag_log
    all_concat_cov = np.concatenate(cov_mat, 1)
    plt.imshow(all_concat_cov, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    const1 = (2*math.pi)**(-covariances[0].shape[0]/2)
 #   print(const1)
  #  cond_lik = np.zeros((12,10))
    cond_lik = np.zeros((digits.shape[0],10))
    const2 = np.zeros((10))
    cov_inv = np.zeros((10))
    for k in range((10)):
  #  for k in range(10):
        k = int(k)
       # print("class ", k)
        det = np.linalg.det(covariances[k])
        const2 = 1/math.sqrt(det)
        cov_inv = np.linalg.inv(covariances[k])
        class_mean = means[k,:]
      #  for i in range(12):
        for i in range(digits.shape[0]):
            data = digits[i,:]    
            diff = data-class_mean
            in_exp = -1/2*np.dot(np.dot(diff.transpose(),cov_inv),diff)
       #     print("in_exp",in_exp)
            likelihood = np.log(const1)+np.log(const2)+in_exp
            cond_lik[i,k] = likelihood
       #     print(likelihood)
    return cond_lik

def conditional_likelihood(digits, means, covariances,labels=None):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    prior = np.log(1/10)
    gen_lik = generative_likelihood(digits,means,covariances)
  #  print(np.exp(gen_lik))
    evidence = np.sum(np.exp(gen_lik)*np.exp(prior),axis=1)
    evidence_mat = np.repeat(evidence,10,axis=0)
 #   evidence_mat = evidence_mat.reshape(12,10)
    evidence_mat = evidence_mat.reshape(digits.shape[0],10)
    out = (gen_lik +prior)  - np.log(evidence_mat)
 #   out = np.log(evidence_mat)
  #  print("log_cond_new", out,out.shape)
    return(out)
    

def conditional_likelihood_old(digits, means, covariances,labels=None):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    prior = 1/10
    
    const1 = (2*math.pi)**(-covariances[0].shape[0]/2)
    print(const1)
    cond_lik = np.zeros((digits.shape[0],1))
  #  for i in range(12):
    for i in range(digits.shape[0]):
        data = digits[i,:]
        evidence = np.zeros((10,1))
        reshaped = data.reshape(8,8)
        if (labels != None):
            lab = int(labels[i])
       #     print("lab: ", lab)
        else:
            lab = None
        for k in range(10):
    #    for k in range(10):
     #       print("class: ", k)
            class_mean = means[k,:]
    #        print("means", class_mean.shape)
            det = np.linalg.det(covariances[k])
            const2 = 1/math.sqrt(det)
  #          print("const2:", const2)
            
       #     print("data", data.shape)
        #    print("data", data.shape)
            diff = data-class_mean
            cov_inv = np.linalg.inv(covariances[k])
       #     print("diff:", min(diff),max(diff))
       ##     print("inv ",cov_inv)
            in_exp = -1/2*np.dot(np.dot(diff.transpose(),cov_inv),diff)
        #    print("exp",math.exp(in_exp))
            likelihood = const1*const2*math.exp(in_exp)
        #    print("likelihood: ", likelihood)
            evidence[k]=likelihood
       
        cond_lik_data = prior*evidence /np.sum(evidence*prior,axis=0)
      #  cond_lik_data = np.sum(evidence*prior,axis=0)
    #    print("old ", np.log(cond_lik_data))
   #     plt.imshow(reshaped,cmap='gray')
    #    plt.show()
        if (lab == None):
            best_lik = np.max(cond_lik_data)
            class_pred = np.where(cond_lik_data==best_lik)[0]+1
            if (class_pred==10):
                class_pred = 0
            cond_lik[i] = class_pred
        else:
            if(lab ==10):
                lab = 0
            best_lik = cond_lik_data[lab-1]
            cond_lik[i] = np.log(best_lik)
    #    print("best:",best_lik,np.where(cond_lik_data==best_lik)[0]+1, np.log(best_lik))
        
            
        
    return cond_lik

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
    '''
    cond_likelihood = conditional_likelihood(digits, means,covariances)
    right_cond_lik = np.zeros((digits.shape[0]))
    for i in range(digits.shape[0]):
        right_cond_lik[i] = right_class(cond_likelihood[i,:],labels[i])
    right_cond_sum = np.sum(right_cond_lik)
    right_cond_mean = right_cond_sum/digits.shape[0]
    print("average coditional likelihood ", right_cond_mean, np.exp(right_cond_mean))
    # Compute as described above and return
    return right_cond_mean

def avg_conditional_likelihood_old(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood_old(digits, means, covariances,labels)
    print("cond like shape",cond_likelihood.shape)
    avg = np.sum(cond_likelihood)/cond_likelihood.shape[0]

    # Compute as described above and return
    return avg

def classification_accuracy(predicted_labels,eval_labels):
    pred_truth = np.zeros(predicted_labels.shape[0],dtype={'names':('predicted','truth'),'formats':('int8','int8')})
    pred_truth['predicted'] = predicted_labels
    pred_truth['truth']= eval_labels
    print(pred_truth[0:3])
    correct = pred_truth[pred_truth['predicted'] == pred_truth['truth']]
    return(correct.shape[0]/pred_truth.shape[0])

def best_class (x):
    best_k = np.argmax(x)+1
    if (best_k == 10):
        best_k = 0
    return(best_k)

def classify_data(digits, means, covariances,labels):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    print(cond_likelihood.shape)
    pred = np.apply_along_axis(best_class, 1 , cond_likelihood)
    print(pred[0:10])
    print(labels[0:10])
    print(pred.shape)
    # Compute and return the most likely class
    return(pred)

def classify_data_old(digits, means, covariances,labels):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
 #   train_labels = train_labels[np.where(train_labels<)]
    # Fit the model
    print(train_data.shape)
    means = compute_mean_mles(train_data, train_labels)
  #  print(means)
    covariances = compute_sigma_mles(train_data, train_labels,means)
    plot_cov_diagonal(covariances)
    avg_train = avg_conditional_likelihood(train_data,train_labels,means,covariances)
   # avg_train_old = avg_conditional_likelihood_old(train_data,train_labels,means,covariances)
   # print("avg_train old", avg_train_old)
    print("avg_train", avg_train)
    avg_test = np.exp(avg_conditional_likelihood(test_data,test_labels,means,covariances))
    print("avg_test", avg_test)
    pred_train = classify_data(train_data,means,covariances,train_labels)
    acc_train = classification_accuracy(pred_train,train_labels)
    
    pred_test = classify_data(test_data,means,covariances,test_labels)
 #   print(pred_test[0:5])
    acc_test = classification_accuracy(pred_test,test_labels)
    print("train: ", acc_train, "test: ", acc_test)
#    print(acc_test)
 #   conditional_likelihood_old(train_data,means,covariances)
  #  conditional_likelihood(train_data,means,covariances)
  #  generative_likelihood(train_data,means,covariances)
    #conditional_likelihood(train_data,means,covariances,train_labels)
    # Evaluation

if __name__ == '__main__':
    main()