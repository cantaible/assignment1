# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    std是所初始化得到的初始w缩小的比例
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    h1 = np.maximum(0,np.dot(X, W1) + b1) # 计算第一个隐层的激活数据(NxH)
    scores=np.dot(h1, W2) + b2 # 神经元输出(NxC)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    scores = scores - np.reshape(np.max(scores,axis=1),(N,-1))#NxC
    #scores中的每个元素减去这行的最大值
    #axis=1每行，
    p = np.exp(scores)/np.reshape(np.sum(np.exp(scores),axis=1),(N,-1))#NxC
    #scoes中e每个元素除以e每行元素之和
    loss = -sum(np.log(p[np.arange(N),y]))/N
    #loss是一个数，取对数之后求和
    loss += 0.5*reg*np.sum(W1*W1)+0.5*reg*np.sum(W2*W2)
    #正则化

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dscores = p
    dscores[range(N),y]-=1.0#NxC
    #这个有公式推导的，就是这个样子，p中这个类别中的元素减去这个类别的
    dscores/=N#loss中除以了n所以这里也要除
    dW2 = np.dot(h1.T,dscores)#HxC
    dh2 = np.sum(dscores,axis=0,keepdims=False)#C*1
    #对h2[i]求导的时候，因为scores中的每一列都包含了h2[i]，所以得把这一列累加起来
    #然后弄成一个列向量
    da2 = np.dot(dscores,W2.T)#NxH
    #此时是经过relu之后的
    da2[h1<=0]=0#NxH
    #relu'=max(0,1)
    dW1 = np.dot(X.T,da2)#DxH
    dh1 = np.sum(da2,axis=0,keepdims=False)#Hx1
    #隐藏层的偏置项
    dW2 += reg*W2#正则化
    dW1 += reg*W1#正则化
    grads['W1']=dW1
    grads['b1']=dh1
    grads['W2']=dW2
    grads['b2']=dh2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.验证数据？
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.正则化强度
    - num_iters: Number of steps to take when optimizing.优化时采取的步骤数
    - batch_size: Number of training examples to use per step.每批数据的数据量大小
    - verbose: boolean; if true print progress during optimization.如果该值为true则
      在优化过程中打印进度正常
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    #训练数据的总量/每批数据量

    # Use SGD to optimize the parameters in self.model
    #利用随机梯度下降来优化模型的参数
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      indices=np.random.choice(num_train,batch_size)
      #从num_train中返回batch_size个随机采样
      X_batch=X[indices]#batch_size x D
      y_batch=y[indices]#batch_size x 1
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      #调用完这个函数之后，可以完成一次前向传播，计算出损失值并算出梯度
      loss_history.append(loss)
      #记录下该损失

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      #参数更新
      W1 = grads['W1']
      b1 = grads['b1']
      W2 = grads['W2']
      b2 = grads['b2']

      self.params['W1'] -= learning_rate*W1
      self.params['b1'] -= learning_rate*b1
      self.params['W2'] -= learning_rate*W2
      self.params['b2'] -= learning_rate*b2
      
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        #每迭代100次就打印一次

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        #
        # Check accuracy
        #预测精度，记录
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        #改变学习率
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None#1xC

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    h1 = np.maximum(0, np.dot(X, self.params['W1']) + self.params['b1'])
    scores = np.dot(h1, self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


