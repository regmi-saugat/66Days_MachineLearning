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

**Day19 of 66DaysOfData!**

  **💡 Bias - Variance TradeOff:**
  - Bias: Bias is the inability of a model to learn enough about the relationship between the predictors and the response that’s implied by the dataset. Such error is occured due to wrong assumptions. High bias underfit the training data and create a high training error.
  - Variance: It is the variability of model prediction for a given data point which tells us spread of data. The model with high variance has a very complex fit to the training data and thus not able to fit accurately on the data which it hasn’t seen before. 
  - Irreducible error: It is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (fixing the data sources, such as broken sensors, or detect and remove outliers).
  - The bias-variance trade-off is the tension between bias and variance in ML models. Biased models fail to capture the true trend, resulting in underfitting, whereas low-bias high-variance models likely result in overfitting.

![Images](https://github.com/regmi-saugat/66Days_MachineLearning/blob/main/Images/18_biasvaraince_tradeoff.png)
