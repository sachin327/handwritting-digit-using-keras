 Loss function
=
  cost = -1/m *np.sum(np.sum(((Y*np.log(A2)+(1-Y)*np.log(1-A2))))) + (lambda_/(2*m))*(np.sum(np.sum(W1*W1))+np.sum(np.sum(W2*W2))) 
  
  
  


 
 

m is number of examples
 

Y is actual output

 
A2 is predicted output

 
lambda_ is a regularization term to reduce over fitting

 
W1 and W2 are weights of neural network

  


# It uses keras library to train the model.So it is more accurate on test data.
