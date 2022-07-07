# **Journey of 66DaysOfData in Machine Learning**

**Day1 of 66DaysOfData!**
  
  **💡 Logistic Regression:**
  - Logistic Regression is the appropriate regression analysis to conduct when the dependent variable is binary. It is used to describe data and to explain the relationship between one dependent binary varible and one or more nominal, ordinal, interval or ratio-level varaibles.
  - Binary or Binomial Logistic Regression can be understood as the type of Logistic Regression that deals with scenarios wherein the observed outcomes for dependent variables can be only in binary, i.e., it can have only two possible types.
  - Multinomial Logistic Regression works in scenarios where the outcome can have more than two possible types – type A vs type B vs type C – that are not in any particular order.
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/01_logisticregression.png)

**Day2 of 66DaysOfData!**
  
  **💡 Gradient Descent:**
  - It is an algorithm to find the minimum of a convex function.  It is used in algorithm, for example, in linear regression.
Gradient descent is an iterative optimization algorithm that is popular and it is a base for many other optimization techniques, which tries to obtain minimal loss in a model by tuning the weights/parameters in the objective function.
    
      There are threee types of Gradient Descent:
            i. Batch Gradient Descent
            ii. Stochastic Gradient Descent
            iii. Mini Batch Gradient Descent
            
       Steps to achieve minimal loss:
            1. Decide your cost function.
            2. Choose random initial values for parameters θ, 
            3. Find derivative of your cost function, 
            4. Choosing appropriate learning rate, 
            5. Update your parameters till you converge. This is where, you have found optimal θ values where your cost function, is minimum.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/02_gradientdescent.png)

**Day3 of 66DaysOfData!**
  
  **💡 Perceptron Algorithm:**
  - The Perceptron is one of the simplest ANN architectures, invented by Frank Rosenblatt. It is based on a slightly different artificial neuron called a threshold logic unit (TLU).
  - Perceptron algorithm is a simple classification method that plays an important role in development of the much more felxible neural network and are trained using the stochastic gradient descent optimization algorithm.
  - It consists of single node or neuron that takes a row of data as input and predicts a class label. This is achieved by calculating the weighted sum of the inputs and a bias (set to 1). The weighted sum of the input is called activation.
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/03_perceptron.png)

**Day4 of 66DaysOfData!**
  
  **💡 K Nearest Neighbor:**
  - K-Nearest Neighbor is a Supervised Machine Learning Algorithm that is used to solve classificaiton as well as regression problems. 
  - It is probably the first machine leanring algorithm developed and due to its simple nature, it is still widely accepted in solving many industrial problems. 
  - Whenever new test sample comes, it tries to verify the similarity of the test sample with its training sample
  
          Properties which might define KNN well:
          1. Lazy learning algorithm − KNN is a lazy learning algorithm because it does not have a specialized training phase and uses all the data for training while classification.
          2. Non-parametric learning algorithm − KNN is also a non-parametric learning algorithm because it doesn’t assume anything about the underlying data.
          
          
          Steps to be carried out during the KNN algorithm are as follow:
          1. First we need to select the number of neighbours we want to consider. 
          2. We need to find the K-Neighbours based on any distance metric, that can be Euclidean/Manhatten/or custom distance metric.
          [The most commonly used method to calculate distance is Euclidean.]
          3. Among selected K - neighbours, we need to count how many neighbours are form the different classes
          4. Assign the test data sample to the class for which the count of neighbours was maximum
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/04_knn.png)
