# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:48:43 2017

@author: Dr_Salcedo
"""



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

def classify_data_old(digits, means, covariances,labels):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass
