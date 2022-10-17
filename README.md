# **Journey of 66DaysOfData in Machine Learning**

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/main.png)

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
  - A random forest is a slight extension to the bagging approach for decision trees that can further decrease overfitting and improve out-of-sample precision. Unlike bagging, random forest are exclusively designed for decision trees. Like bagging, a random forest combines the predictions of several base learners, each trained on a bootstrapped sample of the original training set. Random forests, however, add one additional regulatory step: at each split within each tree, we only consider splitting a randomly-chosen subset of the predictors. Hence, random forests average the results of several decision trees and add two sources of  randomness to ensure differentiation between the base learners: randomness in which observations are sampled via the bootstrapping and randomness in which predictors are considered at each split. 
  - Concluding, Random forests average the results of several decision trees and add two sources of randomness to ensure differentiation between the base learners: randomness in which observations are sampled via the bootstrapping and randomness in which predictors are considered at each split. Here, I have presented the implementation of Random Forest using row sampling and column sampling here in the snapshot. Excited about the days ahead!

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/08_random_forest.png)


**Day9 of 66DaysOfData!**

  **💡 Adaboost:**
  - Like bagging and random forests, boosting combines multiple weak learners into one improved model. Boosting trains these weak learners sequentially, each one learning from the mistakes of the last. Rather than a single model, “boosting” refers to a class of sequential learning methods. We fit a weighted learner depends on the type of learner. AdaBoost (Adaptive Boosting) is a very popular boosting technique that aims at combining multiple weak classifiers to build one strong classifier. It follows a decision tree model with a depth equal to one. Here, I have presented the understanding and math behind Adaboost algorithm. Excited about the days ahead!

          Implementation of the adaboost algorithm:
          • Assign equal weights to all observation, W = 1/N
          • Classify random samples using stumps
          • Calculate the total error
          • Calculate performance of the stump
          • Update the weights
          • Update weights in Iteration
          • Final Prediction

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/09_adaboost_a.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/09_adaboost_b.png)

**Day10 of 66DaysOfData!**

  **💡 Gradient Boosting:**
  - Gradient Boosting is a system of machine learning boosting, representing a decision tree for large and complex data. It relies on the presumption that the next possible model will minimize the gross prediction error if combined with the previous set of models. The decision trees are used for the best possible predictions. It is a very powerful technique for building predictive models which is applicable for risk functions and optimizes prediction accuracy of those functions. 
 
          Steps of Gradient Tree Boosting Algorithm
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/10_gradient_boosting_steps.png)

**Day11 of 66DaysOfData!**

  **💡 Stacking:**
  - Stacking also known as Stacked Generalization is a way to ensemble multiple classification or regression model. The architecture of the stacking model is designed in such way that it consists of two or more base/learner's models and meta-model that combines the predictions of base models. These base models are called level 0 models, and the meta model is called level 1 model. 
 - Stacking takes the outputs of base-model as input and attempts to learn how to best combine the input predictions to make a better output prediction by meta-model. It involves combining the predictions from multiple learning models on the same datasets which is designed to improve modeling performance. The meta model is trained on the predictions made by base models on out-of-sample data. Here, I have presented the implementation of Stacking by K-Fold approach in the snapshot. Excited about the days ahead!

          The steps applied in Stacking | K - Fold Approach:
          • Split the data into Training/Testing and Validation datasets
          • Decide the value of the k-fold(k)
          • Train the level 0 Model
          • Train the Level 1 Model
          • Make predictions for the Testing/Validation Data

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/11_stacking.png)

**Day12 of 66DaysOfData!**

  **💡 Principal Component Analysis:**
  - PCA is a dimensionality reduction technique that enables us to identify correlations and patterns in a datasets so that it can be transformed into a new dataset of significantly best possible lower dimension without losing any important data. It is an unsupervised algorithm that ignores the class labels and finds directions of maximal variance of data. It works on a condition that while the data in a higher-dimensional space is mapped to data in a lower dimension space, the variance or spread of the data in the lower dimensional space should be maximum. Here, I have presented the implementation of PCA using Iris dataset in the snapshot. I hope you will gain some insights and work on the same. Excited about the days ahead!
  
          The process involved during PCA:
          • Standardize the data
          • Calculate covariance matrix
          • Find EigenValues and EigenVectors
          • Compute Principal Components 
          • Reduce the dimension of the datasets

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/12_PCA.png)


**Day13 of 66DaysOfData!**

  **💡 K-Means Clustering Algorithm:**
  - K-Means Clustering is an Unsupervised Learning Algorithm which attempts to group observations into k groups, with each group having roughly equal variance. The number of groups, k, is specified by the user as a hyperparameter.
  - The Elbow Method is one of the popular way to find the optimal number of clusters. This method uses the concept of WCSS value. WCSS stands for Within Cluster Sum of Squares, which defines the total variations within a cluster. Here, I have presented the implementation of K-Means algorithm from scratch in the snapshot. I hope you will gain some insights and work on the same. Excited about the days ahead!
  
            Steps included in K-Means Algorithm:
            1. Decide the number of clusters 'k' 
            2. Initialize k random points from data as centroids 
            3. Assign all points to closest cluster centroid
            4. Recompute the centroids of newly formed clusters 
            5. Repeat the 3rd & 4th steps until newly formed clusters are not changing.
            
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/13_KMeans_a.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/13_kmeans_b.png)

**Day14 of 66DaysOfData!**

  **💡 Hierarchical Clustering Algorithm:**
  - Hierarchical Clustering is an clustering algorithm that builds a hierarchy of clusters. This algorithm starts with all the data points assigned to a cluster of their own. Then two nearest clusters are merged into same cluster. In the end, this algorithm terminates when there is only a single cluster left. 
  - The decision of the number of clusters that can best depict different groups can be chosen by observing the dendrogram. The best choice of the number of clusters is the number of  vertical lines in the dendrogram cut by a horizontal line that can transverse the maximum distance vertically without intersecting a cluster. Here, I have presented the implementation of Hierarchical Clustering, obtained dendrogram and final visualization output of clustering in the snapshot. I hope you will gain some insights and work on the same. Excited about the days ahead! 
  
              It is basically two types :
              1. Agglomerative Clustering:
              Here, each observation is initially considered as a cluster of its own. Then, the most similar clusters are successively merged until there is just one single big cluster. This hierarchy of clusters is represented as a dendrogram.

              The steps followed for Agglomerative Clustering:
              1. Initialize the proximity matrix
              2. Make each point a cluster
              3. Inside a loop:
                • Merge the 2 closest clusters
                • Update the proximity matrix.
              4. Run the loop, until we left with single cluster
              [Here, we can calculate the distance between clusters by Single Linkage, Complete Linkage, Group Average, Ward methods.]

              2. Divisive Clustering:
              It follows a top-to-down approach which is just opposite to Agglomerative clustering. Here, all the data points are assigned to a single cluster where each iteration, clusters are separated into other clusters based upon dissimilarity and the process repeats until we are left with n clusters.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/14_hierarchical_clustering_a.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/14_hierarchical_clustering_b.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/14_hierarchical_clustering_c.png)

**Day15 of 66DaysOfData!**

  **💡 DBSCAN Clustering:**
  - DBSCAN stands for Density Based Spatial Clustering of Application with Noise is based on point that are close to each other on a distance measurement(usually, Euclidean Distance) and a minimum numbers of points. It also marks as outlier the points that are in low-density regions labeled as noise. 
  - The main idea behind DBSCAN is that a point belongs to a cluster if it is close to many points from that cluster. Here, I have presented the implementation of DBSCAN algorithm from scratch and its clustered output in the snapshot. I hope you will gain some insights and work on the same. Excited about the days ahead! 
  
              The data are classified into three points based on epsilon and minimum points parameters:
              1. Core Points: Data points has at least minimum points within epsilon distance.
              2. Boundary / Border Points: Data points has at least one core point within epsilon distance.
              3. Noise Points: Data points that has no core points within epsilon distance.
              
              Steps followed in DBSCAN Algorithm:
              1. Label points as the core, border, and noise points.
              2. Eliminate all noise points.
              3. For every core point that has not yet assigned a cluster,
                a) Create a new cluster with the point.
                b) add all the points that are density-connected.
              4. Assign each border points to the cluster of the closest core point.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/15_dbscan_a.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/15_dbscan_b.png)

**Day16 of 66DaysOfData!**

  **💡 Support Vector Machine:**
  - It's a supervised machine learning algorithm which can be used for both classification or regression problems. But it's usually used for classification. Given 2 or more labeled classes of data, it acts as a discriminative classifier, formally defined by an optimal hyperplane that seperates all the classes.
  - Support vectors are the data points nearest to the hyperplane, the points of a data set that, if removed, would alter the position of the dividing hyperplane. Because of this, they can be considered the critical elements of a data set, they are what help us build our SVM.
  - The data is sometimes not linearly separable so the kernel trick is to map this space into higher dimension, where they are linearly separable. Then mapping the classification back into the original space gives non-linear classifier. This mapping function that maps lower dimensional data to higher is called as the kernel. Here, I have presented the implementation of Support Vector Machine algorithm from scratch in the snapshot. I hope you will gain some insights and work on the same. Excited about the days ahead! 
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/16_support_vector_machine.png)

**Day17 of 66DaysOfData!**

  **💡 Cross Validation:**
  - Cross Validation is a resampling procedure used to evaluate Machine Learning Models on a limited data sample which has a parameter that splits the data into number of groups. In a cross validation method, we have to fix the number of folds of data and run the analysis on each folds and average overall error estimate. 
  - K fold cross validation is one of the popular method because it is easy to understand and also it is less biased than other method. Here, in the snapshot I have presented the implementation of k fold cross validation using breast cancer dataset and got around 95% accuracy. I hope you will gain some insights and work on the same. Excited about the days ahead! 

              Steps for k-fold- Cross Validation:
              1. Shuffle the dataset randomly.
              2. Split the dataset into k groups
              3. For each unique group:
                    • Take the group as a holdout or test data set
                    • Take the remaining groups as a training data set
                    • Fit a model on the training set and evaluate it on the test set
                    • Retain the evaluation score and discard the model
              4. Summarize the skill of the model using the sample of model evaluation scores
              
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/17_cross_validation.png)

**Day18 of 66DaysOfData!**

  **💡 Bias - Variance TradeOff:**
  - Bias: Bias is the inability of a model to learn enough about the relationship between the predictors and the response that’s implied by the dataset. Such error is occured due to wrong assumptions. High bias underfit the training data and create a high training error.
  - Variance: It is the variability of model prediction for a given data point which tells us spread of data. The model with high variance has a very complex fit to the training data and thus not able to fit accurately on the data which it hasn’t seen before. 
  - Irreducible error: It is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (fixing the data sources, such as broken sensors, or detect and remove outliers).
  - The bias-variance trade-off is the tension between bias and variance in ML models. Biased models fail to capture the true trend, resulting in underfitting, whereas low-bias high-variance models likely result in overfitting.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/18_biasvaraince_tradeoff.png)

**Day19 of 66DaysOfData!**

  **💡 Gradient Descent:**
  - Gradient Descent is an first order interactive optimization algorithm for finding a local minimum of a differentiable function. It's based on a convex function and tweaks its parameters iteratively to minimize a given function to its local minimum. The main objective of using a gradient descent algorithm is to minimize the cost function using iteration.
  -  We start by filling theta value with random values, which is called random initialization and then we improve it gradually taking one small step at a time. The size of these steps is known learning rate. This is typically a small value that is evaluated and updated based on the behavior of the cost function.

**💡 Batch Gradient Descent:**
  - Batch Gradient Descent involves calculations over the full training set at each step as a result of which it is very slow on very large training data. Parameters are updated after computing the gradient of the error with respect to the entire training set. As a result, it takes a longer time to train when the size of the training set is large. 

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/19_gradient_descent_b.png)

**Day20 of 66DaysOfData!**

  **💡 Stochastic Gradient Descent:**
  - In batch gradient descent, we look at every example in the entire training set on every step. This can be quite slow if the training set is sufficiently large. To overcome this, stochastic gradient descent update values after looking at each item in the training set,so that we can start making progress right away. It has been shown that SGD almost surely converges to the global cost minimum if the cost function is convex.
  - Stochastic gradient descent attempts to find the global minimum by adjusting the configuration of the network after each training point. Instead of decreasing the error, or finding the gradient, for the entire data set, this method merely decreases the error by approximating the gradient for a randomly selected batch.
  - SGD algorithm derivative is computed taking one point at a time & memory requirement is less compared to the gradient descent algorithm. A crucial parameter for SGD is the learning rate, it is necessary to decrease the learning rate over the time. 

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/20_sgd_a.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/20_sgd_b.png)

**Day21 of 66DaysOfData!**

  **💡 Perceptron:**
  - Perceptron are linear, binary classifier which is used to classify instances into one of two classes. Perceptrons fit a linear decision boundary in order to seperate the classes where classes are linearly seperable. The structure of perceptron consists of a single layer of Threshold Logic Units (TLU's) where it compute's a weighted sum of it's inputs and applies step function to sum and output results. A drawback to the perceptron model is that it is unable to fit non-linear decision boundaries.

  **💡 Multilayer Perceptron:**
  - Multilayer Perceptron also known as vanilla neural network are able to overcome the drawbacks of perceptrons. They are able to fit complex non-linear decision boundaries. A MLP is composed of multiple perceptrons which has one input layer, one or more layers of hidden layers, and one output layer. Every layer except the output layer includes a bias neuron and is fully connected to the next layer. 
- The inputs are combined with the initial weights in a weighted sum and subjected to the activation function where each layer is linearly combined to the propagated to the next layer. All the computation of internal representation of the data goes through the hidden layer to the output layer. 
- The inputs are pushed forward through the MLP by taking the dot product of the input with the weights that exist between the input layer and the hidden layer. This dot product results a value at the hidden layer where it uses activation function in each hidden layer. Once the calculated output at the hidden layer been pushed through the activation function, it pushes to the next layer in the MLP by taking the dot product with the corresponding weights and the steps is done until it reaches to the output layer.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/21_MLP.png)


**Day22 of 66DaysOfData!**

  **💡 Backpropagation:**
  - Back-propagation is the way of propagating the total loss back into the neural network to know how much of the loss every node is responsible for subsequently updating the weights in such a way that it minimizes the loss by giving the nodes with higher error rates lower weights and vice versa.
  - We can only change the weights and biases, but activations are direct calculation of those weights and biases, which means we can indirectly adjust every part of the neural network, to get the desired output except for the input layer, since that is the dataset that we input.
  
              The steps for the back-propagation:
              • At first, we start with the random initialization
              • Predict the input by the forward propagation.
              • Calcualte the loss using the loss function.
              • Calculate the derivative of the error (to update weight and bias using gradient descent).
              • Calculate the average loss and the update weight, continue the process until we get minimum error.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/22_backpropagation.png)

**Day23 of 66DaysOfData!**

  **💡 Vanishing Gradient Problem:**
  - Vanishing Gradient Problem is a difficulty found in training certain ANN with gradient based methods (e.g Back Propagation). In particular, this problem makes it really hard to learn and tune the parameters of the earlier layers in the network. This problem becomes worse as the number of layers in the architecture increases.
  - It arises when more layers using sigmoid or tanh activation functions are added to neural networks, the gradients of the loss function approaches zero, making the network hard to train.
  - Gradient based methods learn a parameter's value by understanding how a small change in the parameter's value will affect the network's output. If a change in the parameter's value causes very small change in the network's output - the network just can't learn the parameter effectively, which is a problem.
              
              Some methods for handling vanishing gradient problems:
              • Reduce model complexity (This method is not much applicable)
              • Using activation functions like ReLU
              • Proper weight initialization
              • Use Batch initialization
              • Residual Network (ResNets)
  - We can avoid this problem by using activation functions which don't have this property of 'squashing' the input space into a small region. A popular choice is Rectified Linear Unit(ReLU)

**Day24 of 66DaysOfData!**

  **💡 Improving the performance of Neural Networks:**
  - Neural networks are ML algorithms that provides state of the accuracy on many use cases. But, many times the performance of network we are building might not be satisfactory. So, in order to improve the performance of our models there are several ways through which we can improve the performance. 

              There are several problems ocurred due to which the performance may become slow:
              - Vanishing or Exploding gradient problems
              - Not having enough data
              - Slow training process
              - Overfitting

              Here are some of the ways to increase the performance(accuracy) of neural models.
              1. Experimenting with number of hidden layers
              2. Number of neurons per layer
              3. Learning rate
              4. Optimizer
              5. Batch size
              6. Activation functions
              7. Epochs

**Day25 of 66DaysOfData!**

  **💡 Data Scaling:**
  - Data scaling can improve deep learning model stability and performance and is a recommended pre-processing step when working with deep learning neural networks. Data scaling can be achieved by normalizing or standardizing real-valued input and output variables.
  - Normalization is a rescaling of the data from the original range so that all values are within the range of 0 and 1.
  - Standardizing is the process of rescaling the distribution of values so that the mean of observed values is 0 and the standard deviation is 1.
  - The first plot shows the graph without scaling and here the validation accuracy oscillate from 40 - 60 and it can't converge even after completing the epochs. But, in the second graph plot after applying data scaling the validation accuracy gradually increases and reach to it's maximum point, for this here we have used standardization.
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/25_scaling_a.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/25_scaling_b.png)

**Day26 of 66DaysOfData!**

  **💡 Dropout:**
  - Dropout is an approach  to regularization in neural networks which helps to reduce interdependent learning amongst the neurons. It is used as regularization technique in order to reduce overfitting. A fully connected layer occupies most of the parameters, and hence, neurons develop co-dependency amongst each other during training which curbs the individual power of each neuron leading to over-fitting of training data.
  - At each training stage, individual nodes are either dropped out of the net with probability (1-p) or kept with probability p, so that a reduced network is left,  incoming and outgoing edges to a dropped-out node are also removed.

              General tricks for dropout technique:
                1. Dropout layer is usually applied in last layer & if the result did not came good then only applied to other layers.
                2. If there is overfitting increase value of p and if underfitting then decrease the value of p
                
  - One of the drawbacks of dropout is that it increases training time. A dropout network typically takes 2-3 times longer to train than a standard neural network of the same architecture.              
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/26_dropout.png)

**Day27 of 66DaysOfData!**

  **💡 Activation Function:**
  - An activation function in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network. A neural network without an activation function is essentially just a linear regression model. 
  - Activation function also helps to normalize the output of any input in the range between 1 to -1 or 0 to 1. The choice of activation function in the hidden layer will control how well the network model learns the training dataset. The choice of activation function in the output layer will define the type of predictions the model can make.
   
              Types of activation function:
              1. Binary Step Function
              2. Linear Activation Function
              3. Non - linear Activation Function
              
  - **Binary Step Function:** A binary step function is generally used in the Perceptron linear classifier. It thresholds the input values to 1 and 0, if they are greater or less than zero, respectively.
  - **Linear Activation Function:** It is a simple straight line activation function where our function is directly proportional to the weighted sum of neurons or input.
  - **Non - linear Activation Function:** Modern neural network models use non-linear activation functions. They allow the model to create complex mappings between the network’s inputs and outputs, such as images, video, audio, and data sets that are non-linear or have high dimensionality.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/27_act_function.gif)

**Day28 of 66DaysOfData!**

  **💡 Sigmoid Activation Function:**
  - Sigmoid Activation function is very simple which takes a real value as input and gives probability that ‘s always between 0 or 1. The sigmoid is a non-linear function, continuously differentiable, monotonic, and has a fixed output range. Big disadvantage of the function is that it gives rise to a problem of 'vanishing gradients' because it's output isn’t zero centered. Also it takes very high computational time in hidden layer of neural network.
  
  **💡 Tanh or Hyperbolic tangent::** 
  - Tanh help to solve non zero centered problem of sigmoid function. Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear too. It give us almost same as sigmoid's derivative function where it solve sigmoid's drawback but it still can't remove the vanishing gradient problem completely.
  
  **💡ReLU Activation Function:**
  - This is most popular activation function which is used in hidden layer of NN. The formula is deceptively simple: 𝑚𝑎𝑥(0,𝑧)max(0,z). Despite its name and appearance, it's not linear and provides the same benefits as sigmoid but with better performance. It’s main advantage is that it avoids and rectifies vanishing gradient problem and less computationally expensive than tanh and sigmoid.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/28_activation_function.gif)


**Day29 of 66DaysOfData!**

  **💡 Leaky ReLU Function:**
  - For activations in the region (x<0) of ReLU, gradient will be 0 because of which the weights will not get adjusted during descent. That means, those neurons which go into that state will stop responding to variations in error/ input (simply because gradient is 0, nothing changes). This is called the dying ReLu problem.
  - Leaky ReLU function is an improved version of the ReLU activation function. It has a small slope for negative values instead of a flat slope, y = max(0.01x, x). By making this small modification, the gradient comes out to be a non zero value. Hence, there would no longer encounter dead neurons in that region.
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/29_Leaky%20ReLU.png)

**Day30 of 66DaysOfData!**

  **💡 Weight Initialization:**
  - Weight initialization plays an important role in improving the training process of deep learning methods. The goal of weight initialization is to prevent layer activation outputs from exploding or vanishing during the training of the DL technique. Training the network without a useful weight initialization can lead to a very slow convergence or an inability to converge.
  
              Some of the weights initializing techniques which should be avoided:
              1. Zero Initialization 
              2. Initialization with non zero constant value
              3. Random Initialization
              
  - The problem with using this method is that it cause exploding and vanishing gradient problem. Therefore, it is rarely used as a neural network weight initializer. 
  - **Here are some heuristics for weight initialization:**
    - **1. Xavier / Glorot Initialization:** Xavier proposed a more straightforward method, where the weights such as the variance of the activations are the same across every layer. This will prevent the gradient from exploding or vanishing problem.
  
    - **2. He Initialization:** This initialization preserves the non-linearity of activation functions such as ReLU activations. Using the He method, we can reduce or magnify the magnitudes of inputs exponentially and it also solves dying neuron problems.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/30_weight_init.gif)

**Day31 of 66DaysOfData!**

  **💡 Batch Normalization:**
  - It consists of adding an operation in the model just before or after the activation function of each hidden layer, simply zero-centering and normalizing each input, then scaling and shifting the result using two new parameter vectors per layer:one for scaling, the other for shifting.
  - This operation lets the model learn the optimal scale and mean of each of the layer’s inputs. In many cases, if you add a BN layer as the very first layer of your neural network, you do not need to standardize your training set:the BN layer will do it for you.
  - In order to zero-center and normalize the inputs, the algorithm needs to estimate each input’s mean and standard deviation. It does so by evaluating the mean and standard deviation of each input over the current mini-batch.
  
              Some pros of using Batch Normalization:
              1. Networks train faster, converge more quickly & give better results overall.
              2. It allows higher learning rates. Gradient descent usually requires small learning rates for the network to converge.
              3. Makes weights easier to initialize
              4. It solves the problem of internal covariate shift. Through this, we ensure that the input for every layer is distributed around the same mean and standard deviation. 

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/31_batch_normalization.png)

**Day32 of 66DaysOfData!**

  **💡 Momentum Optimization:**
  - Momentum Optimizer in Deep Learning is a technique that reduces the time taken to train a model. 
  - The path of learning in mini-batch gradient descent is zig-zag, and not straight. Thus, some time gets wasted in moving in a zig-zag direction. Momentum Optimizer in Deep Learning smooths out the zig-zag path and makes it much straighter, thus reducing the time taken to train the model. 
  - Momentum Optimizer uses Exponentially Weighted Moving Average, which averages out the vertical movement and the net movement is mostly in the horizontal direction. Thus zig-zag path becomes straighter.
  - Momentum optimization cares a great deal about what previous gradients were: at each iteration, it subtracts the local gradient from the momentum vector m (multiplied by the learning rate η), and it updates the weights by simply adding this momentum vector.
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/32-momentum-optimization.gif)

**Day33 of 66DaysOfData!**

  **💡 Nesterov Accelerated Gradient:**
  - The idea of Nesterov Momentum optimization, or Nesterov Accelerated Gradient (NAG), is to measure the gradient of the cost function not at the local position but slightly ahead in the direction of the momentum.
  - The acceleration of momentum can overshoot the minima at the bottom of basins or valleys. Nesterov momentum is an extension of momentum that involves calculating the decaying moving average of the gradients of projected positions in the search space rather than the actual positions themselves.
  - This has the effect of harnessing the accelerating benefits of momentum whilst allowing the search to slow down when approaching the optima and reduce the likelihood of missing or overshooting it.
- NAG will almost always speed up training compared to regular Momentum optimization. To use it, simply set nesterov=True when creating the SGD optimizer.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/33_NAG.png)

**Day34 of 66DaysOfData!**

  **💡AdaGrad Optimizer:**
  - Adagrad is an algorithm for gradient-based optimization that adapts the learning rate to the parameters, performing smaller updates (i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features.
  - This algorithm performs best for sparse data because it decreases the learning rate faster for frequent parameters, and slower
for parameters infrequent parameter. 
- The weakness of AdaGrad is an aggressive monotonic growth of the denominator as squared gradients get accumulated. After a certain number of iterations the learning rate becomes infinitesimally small, at which point the algorithm essentially stops making steps in the direction of the minimum.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/34_adagrad.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/34(b)%20adagrad.png)

**Day35 of 66DaysOfData!**

  **💡RMS Prop:**
  - AdaGrad got stuck when it was close to convergence, it was no longer able to move in the vertical (b) direction because of the decayed learning rate. RMSProp overcomes this problem by being less aggressive on the decay.
  - Root Mean Squared Propagation, or RMSProp, is an extension of gradient descent and the AdaGrad version of gradient descent that uses a decaying average of partial gradients in the adaptation of the step size for each parameter.
  - The hyperparameter β is known as the decay rate which is used to control the focus of the adaptive learning rate on more recent gradients. In almost all cases RMSProp will outperform AdaGrad & as a result this RMSProp was preferred optimization algorithm until the Adam optimization algorithm was introduced.
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/35_rmsprop.png)

**Day36 of 66DaysOfData!**

  **💡Adam Optimization:**
  - Adam which stands for Adaptive Moment Estimation combines the ideas of Momentum Optimization and RMSProp where Momentum Optimization keeps track of an exponentially decaying average of past gradients and RMSProp keeps track of an exponentially decaying average of past squared gradients. Instead of adapting the parameter learning rates based on the average first moment (the mean) as in RMSProp, Adam also makes use of the average of the second moments of the gradients (the uncentered variance). Here, I have presented the Implementation of Adam Optimizer using Python in the Snapshot. I hope you will also spend time leaerning the Topics. Excited about the days ahead!
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/36_adamoptimizer.png)


**Day37 of 66DaysOfData!**

  **💡Convolution Neural Network:**
  - Our brain processes things in a pictorial fashion. It tries to look for features and identify or classify objects in our surroundings. Since, our aim with neural networks is to mimic the human brain, a convolutional neural network (CNN) is mechanised such that it looks for features in an object.
  - CNN is an neural network which extracts or identifies a feature in a particular image. This forms one of the
most fundamental operations in Machine Learning and is widely used as a base model in majority of Neural Networks like GoogleNet, VGG19 and
others for various tasks such as Object Detection, Image Classification and others.

              CNN has the following five basic components:
              1.Convolution : to detect features in an image
              2. ReLU : to make the image smooth and make boundaries distinct
              3. Pooling : to help fix distored images
              4. Flattening : to turn the image into a suitable representation
              5. Full connection : to process the data in a neural network
              
  - A CNN works in pretty much the same way an ANN works but since we are dealing with images, a CNN has more layers to it than an ANN. In an ANN, the input is a vector, however in a CNN, the input is a multi-channelled image.

**Day38 of 66DaysOfData!**

  **💡Convolution Layer:**
  - The most important building block of CNN is the Convolutional Layer. Neurons in the first Convolutional Layer are not connected to every single pixel in the Input Image but only to pixels in their respective fields. Similarly, each Neurons in second CL is connected only to neurons located within a small rectangle in the first layer. 
  - This architecture allows the network to concentrate on small low-level features in the first hidden layer, then assemble them into larger higher-level features in the next hidden layer, and so on. 

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/38_CNN%20layer.png)

**Day39 of 66DaysOfData!**
  -  Convolutional architecture allows the network to concentrate on small low-level features in the first hidden layer, then assemble them into larger higher-level features in the next hidden layer, and so on. Today I read and implemented about Convolutional Neural Networks, Convolutional Layer, Zero Padding, Filters, Stacking Multiple Feature Maps, Padding, Pooling Layer, Invariance, Convolutional Neural Network Architectures and revised the previous concepts from the **Book Hands On Machine Learning with Scikit Learn, Keras and TensorFlow.** Here, I have presented the implementation of Convolutional Neural Network Architecture using Fashion MNSIT dataset in the Snapshot. I hope you will gain some insights and hope you will also spend some time reading the Topics. Excited for the days ahead !

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/39_CNN%20layer%20for%20Fashion%20MNIST%20dataset.png)

**Day40 of 66DaysOfData!**

  **💡Residual Network (ResNet):**
  - Residual Networks was proposed by He et al. in 2015 to solve the image classification problem. In ResNets, the information from the initial layers is passed to deeper layers by matrix addition. This operation doesn’t have any additional parameters as the output from the previous layer is added to the layer ahead.
  - Deep residual nets make use of residual blocks to improve the accuracy of the models. The concept of “skip connections,” which lies at the core of the residual blocks, is the strength of this type of neural network. Here, I have presented the implementation of ResNet 34 CNN using Keras in the Snapshot. I hope you will gain some insights and spend some time learning the topics. Excited about the days ahead !
 
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/40_resnet.png)

**Day41 of 66DaysOfData!**

  **💡Xception:**
  - Xception stands for Extreme Inception is a deep convolutional neural network architecture that involves Depthwise Separable Convolutions introduced Francois Chollet who works at Google(also creator of Keras).
  - It merges the ideas of GoogLeNet and ResNet Architecture but it replaces the Inception modules with a special layer called Depthwise Separable Convolution.
  - Depthwise Separable Convolutions are alternatives to classical convolutions that are supposed to be much more efficient in terms of computation time.
  - The data first goes through the entry flow, then after than it. goes through the middle flow (repeating itself 8 times in this middle flow), and finally through the exit flow.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/41_xception%20model.png)


**Day42 of 66DaysOfData!**

  **💡Semantic Segmentation:**
  - Semantic segmentation, or image segmentation, is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category. 
  - The task of Semantic Segmentation can be referred to as classifying a certain class of image and separating it from the rest of the image classes by overlaying it with a segmentation mask.
  - Semantic Segmentation finds applications in fields like autonomous driving, medical image analysis, aerial image processing, and more. Today I read about Semantic Segmentation, Classification and Localization, Fully  Mean Average Precision(mAP), Transpose Convolutions, Convolutional Networks or FCNs, Data Augmentation, CNN, and some other topics. Here, I have presented the implementation of Classification and Localization, Transpose Convolutions in the Snapshots. I hope you will gain some insights and work on the same. Excited about the days ahead !
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/42_semantic%20segmentation_a.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/42_semantic%20segmentation_b.png)


**Day43 of 66DaysOfData!**

  **💡Recurrent Neural Network:**
  - A recurrent neural network is a type of artificial neural network which uses sequential data or time-series data. 
  - It is commonly used for ordinal or temporal problems, such as language translation, natural language processing (NLP), speech recognition, and image captioning.
  - The output of RNN depends on the prior elements within the sequence unlike traditional deep neural networks assume that inputs and outputs are independent of each other.

               The different types of RNN are:
              • One to one
              • One to many
              • Many to one
              • Many to many
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/43_RNN.png)

**Day44 of 66DaysOfData!**
  - Deep learning is a subset of machine learning, where artificial neural networks—algorithms modeled to work like the human brain learn from large amounts of data. It attempts to draw similar conclusions as humans would by continually analyzing data with a given logical structure. To achieve this, deep learning uses a multi-layered structure of algorithms called neural networks. Today, I have started reading the Book **Deep Learning with PyTorch**. I learned about PyTorch, Deep Learning Introduction and Revolution, Tensors and Arrays, Deep Learning Competitive Landscape, pretrained neural network that recognizes the subject of an Image, ImageNet, AlexNet and ResNet.. Here, I have presented the implementation of obtaining Pretrained Neural Networks for Image Recognition using PyTorch in the Snapshot. I hope you will also spend some time learning the topics.  Excited about the days ahead !
  
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/44_pretrained_model.png)


**Day45 of 66DaysOfData!**

  **💡The GAN Game:**
  - GAN stands for generative adversarial network, where generative means something is being created, adversarial means the two networks are competing to outsmart the other, and well, network is pretty obvious.
  - The generator network takes the role of the painter in our scenario, tasked with producing realistic looking images, starting from an arbitrary input. The discriminator network is the amoral art inspector, needing to tell whether a given image was fabricated by the generator or belongs in a set of real images. This two network design is atypical for most deep learning architectures but, when used to implement a GAN game, can lead to incredible results.
  - A CycleGAN can turn images of one domain into images of another domain (and back), without the need for us to explicitly provide matching pairs in the training set.
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/45_cycleGAN_a.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/45_cycleGAN_b.png)


**Day46 of 66DaysOfData!**

  **💡Tensors and Multi Dimensional Arrays:**
  - A tensor is an array that is a data structure which stores a collection of numbers that are accessible individually using a index and that can be indexed with
  multiple indices.
  - It is a generalization of vectors and matrices and is easily understood as a multidimensional array. 
  - It is a term and set of techniques known in machine learning in the training and operation of deep learning models that can be described in terms of tensors. Most machines cannot learn without having any data, And modern data is often multi-dimensional. Tensors can play an important role in ML by encoding multi-dimensional data.

**Day47 of 66DaysOfData!**

  **💡Encoding color channels:**
  - An image is represented as a collection of scalars arranged in a regular grid with a height and a width (in pixels). We might have a single scalar per grid point (the pixel), which would be represented as a grayscale image; or multiple scalars per grid point, which would typically represent different colors.
  - There are several ways to encode colors into numbers. The most common is RGB, where a color is defined by three numbers representing the intensity of red, green, and blue. Today, I learned about Serializing tensors, Tensors, Data Representation using Tensors, Working with Images, Adding Color Channels, Changing the Layout, Normalizing the Data. Here, I have presented the implementation of Working with Images like, adding color channels, loading image file, changing images layout, etc using PyTorch in the Snapshot. I hope you will gain some insights and spend the time learning the topics. Excited about the days ahead !
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/47_working%20with%20images.png)

**Day48 of 66DaysOfData!**

  **💡Continuous, Ordinal, and Categorical Values:**
  - Continuous values are strictly ordered, and a difference between various values has a strict meaning. Ordinal values has a strict ordering like continuous values but it has no fixed relationship between values. Categorical values have enumerations of possibilites assigned arbitrary numbers, they simply have distinct values to differentiate them.
  - Today, I learned to represent tabular data, continuous, ordinal, and categorical values, representing scores, one-hot encoding, when to categorize the data, finding thresholds. Here, I have presented the implementation of Working with categorical, ordinal, and categorical data, One-hot encoding, to find the thresholds using PyTorch in the Snapshot. I hope you will gain some insights and spend the time learning the topics. Excited about the days ahead !
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/48_working_with_image.png)

**Day49 of 66DaysOfData!**

  **💡Encoding:**
  - Every written character is represented by a code, i.e. sequence of bits of appropriate length so that each character can be uniquely identified. Encoding is the process of applying a specific code, such as letters, symbols and numbers, to data for conversion into an equivalent cipher. The simplest such encoding is ASCII (American Standard Code for Information Interchange). ASCII encodes 128 characters using 128 integers.
  - Today, I learned about about time series, adding time dimension in data, to shape data by time period, encoding the text into numbers, one-hot encoding. Here, I have presented the implementation of working with time series data, one-hot encoding, converting text into numbers, etc using PyTorch in the Snapshot. I hope you will gain some insights and spend the time learning the topics. Excited about the days ahead !

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/49_a_encoding.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/49_b_encoding.png)

**Day50 of 66DaysOfData!**

  **💡Hyperparameter Tuning:**
  - Hyperparameter Tuning refers to the training of model's parameters and hyperparameters control how the training goes. They are generally set manually. Model parameters are learned from data and hyper-parameters are tuned to get the best fit. Searching for the best hyper-parameter can be tedious.
  - Today, I learned about mechanics of learning, loss function, broadcasting, optimizing the training Loop, overtraining, learning rate, hyperparameter tuning, Normalizing the Inputs, visualization or plotting the data, , Linear Model, gradient descent, backpropagation, PyTorch's Autograd. Here, I have presented the implementation of simple linear model, mechanics of learning, gradient function, training loop, gradient descent , visualization using PyTorch in the snapshot. I hope you will gain some insights and spend the time learning the topics. Excited about the days ahead!

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/50_a_hyperparameter_tuning.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/50_b_hyperparameter_tuning.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/50_c_hyperparameter_tuning.png)


**Day51 of 66DaysOfData!**

  **💡Optimizers:**
  - It is very important to tweak the weights of the model during the training process, to make our predictions as correct and optimized as possible so we use optimizers. Optimizers are methods used to minimize a loss function or to maximize the efficiency of production. They are just a mathematical functions which are dependent on model’s learnable parameters i.e Weights & Biases. There are several types of optimizers like, gradient descent, SGD, RMS-Prop, Ada-delta, Adam, etc.
  - Today, I learned about pytorch autograd,  to compute the gradient,  accumulating the grad function, optimizers, stochastic gradient descent, etc. Here, I have presented the implementation of working with simple linear model, gradient function, training loop, loss function, implementation of SGD in the snapshot. I hope you will gain some insights and spend the time learning the topics. Excited about the days ahead!

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/51_a_optimizers.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/51_b_optimizers.png)


**Day52 of 66DaysOfData!**

  **💡Optimizers:**
  - It is very important to tweak the weights of the model during the training process, to make our predictions as correct and optimized as possible so we use optimizers. Optimizers are methods used to minimize a loss function or to maximize the efficiency of production. They are just a mathematical functions which are dependent on model’s learnable parameters i.e Weights & Biases. There are several types of optimizers like, gradient descent, SGD, RMS-Prop, Ada-delta, Adam, etc.
  - Today, I learned about optimizers, SGD, splitting the sample, training, validation, overfitting, evaluating training loss, autograd nits, etc. Here, I have presented the implementation of SGD & Adam optimizer along with the training loop in the snapshot. I hope you will gain some insights and spend the time learning the topics. Excited about the days ahead!
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/52_a_optimizers.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/52_b_optimizers.png)

**Day53 of 66DaysOfData!**

  **💡Activation Function:**
  - In ANN, each neuron forms a weighted sum of its inputs and passes the resulting scalar value through a function referred to as an activation function. It performs a nonlinear transformation on the input to get better results on a complex neural network. The addition of activation function to neural network executes the non-linear transformation to input and make it capable to solve the several complex problems.
  - Today, I learned about artificial neural network, weights, biases, activation function, error function, etc. Here, I have presented the implementation simple linear model, neural network, along with the training loop using PyTorch in the snapshot. I hope you will gain some insights and spend time learning the topics. Excited about the days ahead!
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/53_a_activation_function.png)
![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/53_b_activation_function.png)
