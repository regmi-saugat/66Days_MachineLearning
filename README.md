# **Journey of 66DaysOfData in Machine Learning**

**Day1 of 66DaysOfData!**
  
  **💡 Logistic Regression:**
  - Logistic Regression is the appropriate regression analysis to conduct when the dependent variable is binary. It is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level variables.
  - Binary or Binomial Logistic Regression can be understood as the type of Logistic Regression that deals with scenarios wherein the observed outcomes for dependent variables can be only in binary, i.e., it can have only two possible types.
  - Multinomial Logistic Regression works in scenarios where the outcome can have more than two possible types – type A vs type B vs type C – that are not in any particular order.
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/01_logisticregression.png)

**Day2 of 66DaysOfData!**
  
  **💡 Gradient Descent:**
  - It is an algorithm to find the minimum of a convex function.  It is used in algorithm, for example, in linear regression.
Gradient descent is an iterative optimization algorithm that is popular and it is a base for many other optimization techniques, which tries to obtain minimal loss in a model by tuning the weights/parameters in the objective function.
    
        There are three types of Gradient Descent:
              i. Batch Gradient Descent
              ii. Stochastic Gradient Descent
              iii. Mini Batch Gradient Descent

         Steps to achieve minimal loss:
              1. Decide your cost function.
              2. Choose random initial values for parameters θ, 
              3. Find derivative of your cost function, 
              4. Choosing appropriate learning rate, 
              5. Update your parameters till you converge. This is where, you have found optimal θ values where your cost function, is minimum.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/02_grad.descent.png)

**Day3 of 66DaysOfData!**
  
  **💡 Perceptron Algorithm:**
  - The Perceptron is one of the simplest ANN architectures, invented by Frank Rosenblatt. It is based on a slightly different artificial neuron called a threshold logic unit (TLU).
  - Perceptron algorithm is a simple classification method that plays an important role in development of the much more flexible neural network and are trained using the stochastic gradient descent optimization algorithm.
  - It consists of single node or neuron that takes a row of data as input and predicts a class label. This is achieved by calculating the weighted sum of the inputs and a bias (set to 1). The weighted sum of the input is called activation.
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/03_perceptron.png)

**Day4 of 66DaysOfData!**
  
  **💡 K Nearest Neighbor:**
  - K-Nearest Neighbor is a Supervised Machine Learning Algorithm that is used to solve classification as well as regression problems. 
  - It is probably the first machine learning algorithm developed and due to its simple nature, it is still widely accepted in solving many industrial problems. 
  - Whenever new test sample comes, it tries to verify the similarity of the test sample with its training sample
  
          Properties which might define KNN well:
          1. Lazy learning algorithm − KNN is a lazy learning algorithm because it does not have a specialized training phase and uses all the data for training while classification.
          2. Non-parametric learning algorithm − KNN is also a non-parametric learning algorithm because it doesn’t assume anything about the underlying data.
          
          Steps to be carried out during the KNN algorithm are as follow:
          1. First we need to select the number of neighbors we want to consider. 
          2. We need to find the K-Neighbors based on any distance metric, that can be Euclidean/Manhattan/or custom distance metric.
          [The most commonly used method to calculate distance is Euclidean.]
          3. Among selected K - neighbors, we need to count how many neighbors are form the different classes
          4. Assign the test data sample to the class for which the count of neighbors was maximum
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/04_knn.png)

**Day5 of 66DaysOfData!**
  
  **💡 Decision Tree:**
  - Decision tree is the powerful and popular tool for classification and regression that splits data-feature values into branches at decision nodes (eg, if a feature is a color, each possible color becomes a new branch) until a final decision output is made.
  - Generally, Decision tree are nothing but a giant structure of nested if-else condition. Mathematically, decision tree use hyperplanes which run parallel to any one of the axes to cut coordinate system into hyper cuboids.
  - Also, I learned about Entropy, GINI impurity, information gain, hyperparameters, overfitting, underfitting in decision tree. 
  - For regression, purity means the first child should have observations with high values of the target variable and the second should have observations with low values and similarly, for classification, purity means the first child should have observations primarily of one class and the second should have observations primarily of another.

**Day6 of 66DaysOfData!**

  **💡 Ensemble Voting Classifier:**
  - A voting ensemble is an ensemble machine learning model that combines the prediction from multiple other models. It implements hard and soft voting. In voting classifier, a hard voting ensemble picks class label that has the prediction with the highest number of votes, whereas soft voting classifies input data based on the probabilities of all the predictions made by different classifiers. Weights applied to each classifier get applied appropriately based on the given equation. I have presented the Implementation Voting Classifier using the Iris dataset here in the Snapshot. Excited about the days ahead!
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/06_voting_classifier.png)

**Day7 of 66DaysOfData!**

  **💡 Bagging Ensemble:**
  - Bagging, also known as bootstrap aggregation, is the ensemble learning method that is commonly used to reduce variance within a noisy dataset. In bagging a random sample data in a training set is selected with replacement (individual datapoints can be chosen more than once). Similarly, when sampling is performed without replacement is called pasting. In bagging method weak learners are trained in parallel which exhibit high variance and low bias. Today I read and implemented about Bagging Ensemble such as Bagging Classifier, Bagging Regressor, and also revised some previous theories which I learned like Decision Tree, Gradient Descent and others. Here, I have presented the implementation of Bagging Classifier here in the snapshot. Excited about the days ahead!
           
          Some Bagging tips:
          - Bagging generally gives better results than Pasting
          - Good results around the 25% - 50% row sampling marks
          - Random Patches and Subspaces should be used while dealing with high dimensional data.
          - To find correct hyperparameter values we can do GridSearchCV or RandomSearchCV.
          
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/07_bagging_classifier.png)

**Day8 of 66DaysOfData!**

  **💡 Random Forest:**
  - A random forest is a slight extension to the bagging approach for decision trees that can further decrease overfitting and improve out-of-sample precision. Unlike bagging, random forest are exclusively designed for decision trees. Like bagging, a random forest combinesthe predictions of several base learners, each trained on a bootstrapped sample of the original training set. Random forests, however, add one additional regulatory step: at each split within each tree, we only consider splitting a randomly-chosen subset of the predictors. Hence, random forests average the results of several decision trees and add two sources of  randomness to ensure differentiation between the base learners: randomness in which observations are sampled via the bootstrapping and randomness in which predictors are considered at each split. 
  - Concluding, Random forests average the results of several decision trees and add two sources of randomness to ensure differentiation between the base learners: randomness in which observations are sampled via the bootstrapping and randomness in which predictors are considered at each split. Here, I have presented the implementation of Random Forest using row sampling and column sampling here in the snapshot. Excited about the days ahead!

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/08_random_forest.png)


**Day9 of 66DaysOfData!**

  **💡 Adaboost:**
  - Like bagging and random forests, boosting combines multiple weak learners into one improved model. Boosting trains these weak learners sequentially, each one learning from the mistakes of the last. Rather than a single model, “boosting” refers to a class of sequential learning methods. We fit a weighted learner depends on the type of learner. AdaBoost (Adaptive Boosting) is a very popular boosting technique that aims at combining multiple weak classifiers to build one strong classifier. It follows a decision tree model with a depth equal to one. Here, I have presented the understanding and maths behind Adaboost algorithm. Excited about the days ahead!

          Implementation of the adaboost algorithm:
          • Assign equal weights to all observation, W = 1/N
          • Classify random samples using stumps
          • Calculate the total error
          • Calculate performance of the stump
          • Update the weights
          • Update weights in Interation
          • Final Prediction

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/09_adaboost_a.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/09_adaboost_b.png)

**Day10 of 66DaysOfData!**

  **💡 Gradient Boosting:**
  - Gradient Boosting is a system of machine learning boosting, representing a decision tree for large and complex data. It relies on the presumption that the next possible model will minimize the gross prediction error if combined with the previous set of models. The decision trees are used for the best possible predictions. It is a very powerful technique for building predictive models which is applicable for risk functions and optimizes prediction accuracy of those functions. 
 
          Steps of Gradient Tree Boosting Algorithm
          ![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/10_gradient_boosting_steps.png)
