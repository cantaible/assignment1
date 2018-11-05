#!/usr/bin/env python
# coding: utf-8

# # Implementing a Neural Network
# In this exercise we will develop a neural network with fully-connected layers to perform classification, and test it out on the CIFAR-10 dataset.

# In[ ]:


# A bit of setup
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet



get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


#相对误差
def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# We will use the class `TwoLayerNet` in the file `cs231n/classifiers/neural_net.py` to represent instances of our network. The network parameters are stored in the instance variable `self.params` where keys are string parameter names and values are numpy arrays. Below, we initialize toy data and a toy model that we will use to develop your implementation.

# In[ ]:


# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    #生成随机数，每次生成的随机数都一样
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)
#应该是验证参数用的

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()


# # Forward pass: compute scores
# Open the file `cs231n/classifiers/neural_net.py` and look at the method `TwoLayerNet.loss`. 
#This function is very similar to the loss functions you have written for the SVM and Softmax exercises: 
#It takes the data and weights and computes the class scores, the loss, and the gradients on the parameters. 
# 
# Implement the first part of the forward pass which uses the weights and biases to compute the scores for all inputs.

# In[ ]:


scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small. We get < 1e-7
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))

#检查梯度算错没有
# # Forward pass: compute loss
# In the same function, implement the second part that computes the data and regularizaion loss.

# In[ ]:


loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))

#检查损失函数
# # Backward pass
# Implement the rest of the function. This will compute the gradient of the loss with respect to the variables `W1`, `b1`, `W2`, and `b2`. Now that you (hopefully!) have a correctly implemented forward pass, you can debug your backward pass using a numeric gradient check:

# In[ ]:


from cs231n.gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.05)

# these should all be less than 1e-8 or so
for param_name in grads:
#对grad中每种梯度
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    #获取loss
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
	#param_grad_num是数值梯度吗
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))
	#计算解析梯度和数值梯度之间的误差
    


# # Train the network
# To train the network we will use stochastic gradient descent (SGD), similar to the SVM and Softmax classifiers. Look at the function `TwoLayerNet.train` 
#and fill in the missing sections to implement the training procedure. 
#This should be very similar to the training procedure you used for the SVM and Softmax classifiers. 
#You will also have to implement `TwoLayerNet.predict`, as the training process periodically performs prediction to keep track of accuracy over time while the network trains.
# 
# Once you have implemented the method, run the code below to train a two-layer network on toy data. You should achieve a training loss less than 0.2.

# In[ ]:


net = init_toy_model()
#初始化一个模型，主要是初始w权重偏置等信息
#x是用init_toy_data生成的数据
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()


# # Load the data
# Now that you have implemented a two-layer network that passes gradient checks and works on toy data, it's time to load up our favorite CIFAR-10 data so we can use it to train a classifier on a real dataset.

# In[ ]:


from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test
	#X_train训练集  X_val验证集  X_test测试集


# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# # Train a network
# To train our network we will use SGD. In addition, we will adjust the learning rate with an exponential learning rate schedule as optimization proceeds; after each epoch, we will reduce the learning rate by multiplying it by a decay rate.

# In[ ]:


input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)


# # Debug the training
# With the default parameters we provided above, you should get a validation accuracy of about 0.29 on the validation set. This isn't very good.
# 
# One strategy for getting insight into what's wrong is to plot the loss function and the accuracies on the training and validation sets during optimization.
# 
# Another strategy is to visualize the weights that were learned in the first layer of the network. In most neural networks trained on visual data, the first layer weights typically show some visible structure when visualized.

# In[ ]:


# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.legend()
plt.show()


# In[ ]:


from cs231n.vis_utils import visualize_grid

# Visualize the weights of the network

def show_net_weights(net):
    W1 = net.params['W1']
	#将W1权重可视化
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(net)


# # Tune your hyperparameters
# 调整超参数
# **What's wrong?**. Looking at the visualizations above, we see that the loss is decreasing more or less linearly, 
#从上面的可视化来看，损失或多或少是线性减少的
#which seems to suggest that the learning rate may be too low. Moreover, 
#这似乎表明学习率可能太低了
#there is no gap between the training and validation accuracy, 
#训练和验证的准确性之间没有差距
#suggesting that the model we used has low capacity, and that we should increase its size. 
#这表明我们使用的模型容量低，并且我们应该增加它的尺寸
#On the other hand, with a very large model we would expect to see more overfitting, 
#另一方面，对于一个非常大的模型，我们期望看到更多的过度拟合。
#which would manifest itself as a very large gap between the training and validation accuracy.
# 这将表现为训练和验证准确性之间的很大差距。
# **Tuning**. Tuning the hyperparameters and developing intuition开发直觉 for how they affect the final performance is a large part of using Neural Networks,
# so we want you to get a lot of practice. Below, you should experiment with different values of the various hyperparameters, 
# including hidden layer size, learning rate, numer of training epochs, and regularization strength. 
# You might also consider tuning the learning rate decay, but you should be able to get good performance using the default value.
# 
# **Approximate results近似结果**. You should be aim to achieve a classification accuracy of greater than 48% on the validation set. Our best network gets over 52% on the validation set.
# 
# **Experiment**: You goal in this exercise is to get as good of a result on CIFAR-10 as you can, with a fully-connected Neural Network. 
#Feel free implement your own techniques (e.g. PCA to reduce dimensionality, or adding dropout, or adding features to the solver, etc.).

# In[ ]:


best_net = None # store the best model into this 

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
# 你hyperparameters using the验证集。你的最佳训练模型的盲最佳_网。              #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#    为了帮助调试您的网络，使用类似于我们上面使用的可视化可能会有所帮助；对于调优不佳的网络，这些可视化与我们上面看到的有明显的定性差异。                                                                           #
# Tweaking扭捏 hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep打扫清除 through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.   超参数的可能组合                       #
#################################################################################

best_valacc=-1.0#最好的精度
input_size = 32 * 32 * 3
num_classes = 10
#hidden_size = 50
hidden_size = 32 * 32 * 3
learn_rate =[7.2e-4]#学习率
#learning_rate_decay=[0.94,0.95,0.93]
reg=[1e-3]#正则化强度
results = {}
params = [x1 for x1 in learn_rate ]
#          for x3 in learning_rate_decay for x4 in reg]
#调整learn_rate
for learn_rate in params:
    net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=6400, batch_size=128,
            learning_rate=7.2e-4, learning_rate_decay=0.95,
            reg=1e-3, verbose=True)

# Predict on the validation set
    val_acc = np.mean(net.predict(X_val) == y_val)
    results[learn_rate] =val_acc 
    if val_acc>best_valacc:
        best_valacc = val_acc
        best_net = net


for learn_rate in sorted(results):
    val_accuracy = results[(learn_rate)]
    print ('learn_rate %e val accuracy: %f' % (learn_rate,val_accuracy))

print ('best validation accuracy achieved during cross-validation: %f' % best_valacc)

#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################


# In[ ]:


# visualize the weights of the best network
show_net_weights(best_net)


# # Run on the test set
# When you are done experimenting, you should evaluate your final trained network on the test set; you should get above 48%.

# In[ ]:


test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)


# **Inline Question**
# 
# Now that you have trained a Neural Network classifier, you may find that your testing accuracy is much lower than the training accuracy. In what ways can we decrease this gap? Select all that apply.
# 1. Train on a larger dataset.
# 2. Add more hidden units.
# 3. Increase the regularization strength.
# 4. None of the above.
# 
# *Your answer*:
# 
# *Your explanation:*
