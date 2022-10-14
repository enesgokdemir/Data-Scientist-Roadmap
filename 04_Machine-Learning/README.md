# 4_ Machine learning

## 1_ What is ML ?

### Definition

Machine Learning is part of the Artificial Intelligences study. It concerns the conception, devloppement and implementation of sophisticated methods, allowing a machine to achieve really hard tasks, nearly impossible to solve with classic algorithms.

Machine learning mostly consists of three algorithms:
![](images/Mk.png)

### Utilisation examples

* Computer vision
* Search engines
* Financial analysis
* Documents classification
* Music generation
* Robotics ...

## 2_ Numerical var

Variables which can take continous integer or real values. They can take infinite values.

These types of variables are mostly used for features which involves measurements. For example, hieghts of all students in a class.

## 3_ Categorical var

Variables that take finite discrete values. They take a fixed set of values, in order to classify a data item.

They act like assigned labels. For example: Labelling the students of a class according to gender: 'Male' and 'Female'

## 4_ Supervised learning

Supervised learning is the machine learning task of inferring a function from labeled training data. 

The training data consist of a set of training examples. 

In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). 

A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples. 

In other words:

Supervised Learning learns from a set of labeled examples. From the instances and the labels, supervised learning models try to find the correlation among the features, used to describe an instance, and learn how each feature contributes to the label corresponding to an instance. On receiving an unseen instance, the goal of supervised learning is to label the instance based on its feature correctly.

An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instance.

## 5_ Unsupervised learning

Unsupervised machine learning is the machine learning task of inferring a function to describe hidden structure from "unlabeled" data(a classification or categorization is not included in the observations). 

Since the examples given to the learner are unlabeled, there is no evaluation of the accuracy of the structure that is output by the relevant algorithm—which is one way of distinguishing unsupervised learning from supervised learning and reinforcement learning.

Unsupervised learning deals with data instances only. This approach tries to group data and form clusters based on the similarity of features. If two instances have similar features and placed in close proximity in feature space, there are high chances the two instances will belong to the same cluster. On getting an unseen instance, the algorithm will try to find, to which cluster the instance should belong based on its feature.

Resource:

[Guide to unsupervised learning](https://towardsdatascience.com/a-dive-into-unsupervised-learning-bf1d6b5f02a7)

## 6_ Concepts, inputs and attributes

A machine learning problem takes in the features of a dataset as input.

For supervised learning, the model trains on the data and then it is ready to perform. So, for supervised learning, apart from the features we also need to input  the corresponding labels of the data points to let the model train on them.

For unsupervised learning, the models simply perform by just citing complex relations among data items and grouping them accordingly. So, unsupervised learning do not need a labelled dataset. The input is only the feature section of the dataset.

## 7_ Training and test data

If we train a supervised machine learning model using a dataset, the model captures the dependencies of that particular data set very deeply. So, the model will always perform well on the data and it won't be proper measure of how well the model performs. 

To know how well the model performs, we must train and test the model on different datasets. The dataset we train the model on is called Training set, and the dataset we test the model on is called the test set.

We normally split the provided dataset to create the training and test set. The ratio of splitting is majorly: 3:7 or 2:8 depending on the data, larger being the trining data.

#### sklearn.model_selection.train_test_split is used for splitting the data.

Syntax:

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  
[Sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

## 8_ Classifiers

Classification is the most important and most common machine learning problem. Classification problems can be both suprvised and unsupervised problems.

The classification problems involve labelling data points to belong to a particular class based on the feature set corresponding to the particluar data point.

Classification tasks can be performed using both machine learning and deep learning techniques.

Machine learning classification techniques involve: Logistic Regressions, SVMs, and Classification trees. The models used to perform the classification are called classifiers.

## 9_ Prediction

The output generated by a machine learning models for a particuolar problem is called its prediction. 

There are majorly two kinds of predictions corresponding to two types of problen: 

1. Classification

2. Regression

In classiication, the prediction is mostly a class or label, to which a data points belong

In regression, the prediction is a number, a continous a numeric value, because regression problems deal with predicting the value. For example, predicting the price of a house.

## 10_ Lift

## 11_ Overfitting

Often we train our model so much or make our model so complex that our model fits too tghtly with the training data.

The training data often contains outliers or represents misleading patterns in the data. Fitting the training data with such irregularities to deeply cause the model to lose its generalization. The model performs very well on the training set but not so good on the test set. 
![](images/over.png)

As we can see on training further a point the training error decreases and testing error increases.

A hypothesis h1 is said to overfit iff there exists another hypothesis h where h gives more error than h1 on training data and less error than h1 on the test data

## 12_ Bias & variance

Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.

Variance is the variability of model prediction for a given data point or a value which tells us spread of our data. Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data.


Basically High variance causes overfitting and high bias causes underfitting. We want our model to have low bias and low variance to perform perfectly. We need to avoid a model with higher variance and high bias
![](images/biasVVariance.png)

We can see that for Low bias and Low Variance our model predicts all the data points correctly. Again in the last image having high bias and high variance the model predicts no data point correctly.
![](images/Bias-Variance-tradeoff-in-Machine-Learning.png)

We can see from the graph that rge Error increases when the complex is either too complex or the model is too simple. The bias increases with simpler model and Variance increases with complex models.

This is one of the most important tradeoffs in machine learning



## 13_ Tree and classification

We have previously talked about classificaion. We have seen the most used methods are Logistic Regression, SVMs and decision trees. Now, if the decision boundary is linear the methods like logistic regression and SVM serves best, but its a complete scenerio when the decision boundary is non linear, this is where decision tree is used.

![](images/Dimensionality-reduction-using-linear-matrix-factorization-projecting-the-data-on-a.png)

The first image shows linear decision boundary and second image shows non linear decision boundary.

Ih the cases, for non linear boundaries, the decision trees condition based approach work very well for classification problems. The algorithm creates conditions on features to drive and reach a decision, so is independent of functions.

![](images/decision-tree-example.png)

Decision tree approach for classification

## 14_ Classification rate

## 15_ Decision tree

Decision Trees are some of the most used machine learning algorithms. They are used for both classification and Regression. They can be used for both linear and non-linear data, but they are mostly used for non-linear data. Decision Trees as the name suggests works on a set of decisions derived from the data and its behavior. It does not use a linear classifier or regressor, so its performance is independent of the linear nature of the data. 

One of the other most important reasons to use tree models is that they are very easy to interpret.

Decision Trees can be used for both classification and regression. The methodologies are a bit different, though principles are the same. The decision trees use the CART algorithm (Classification and Regression Trees)

Resource:

[Guide to Decision Tree](https://towardsdatascience.com/a-dive-into-decision-trees-a128923c9298)


## 16_ Boosting

#### Ensemble Learning

It is the method used to enhance the performance of the Machine learning models by combining several number of models or weak learners. They provide improved efficiency.

There are two types of ensemble learning:

1. Parallel ensemble learning or bagging method

2. Sequential ensemble learning or boosting method

In parallel method or bagging technique, several weak classifiers are created in parallel. The training datasets are created randomly on a bootstrapping basis from the original dataset. The datasets used for the training and creation phases are weak classifiers. Later during predictions, the reults from all the classifiers are bagged together to provide the final results.

![](images/bagging.png)

Ex: Random Forests

In sequential learning or boosting weak learners are created one after another and the data sample set are weighted in such a manner that during creation, the next learner focuses on the samples that were wrongly predicted by the previous classifier. So, at each step, the classifier improves and learns from its previous mistakes or misclassifications.
![](images/ensemble-learning.jpg)

There are mostly three types of boosting algorithm:

1. Adaboost

2. Gradient Boosting

3. XGBoost

Adaboost algorithm works in the exact way describe. It creates a weak learner, also known as stumps, they are not full grown trees, but contain a single node based on which the classification is done. The misclassifications are observed and they are weighted more than the correctly classified ones while training the next weak learner. 

sklearn.ensemble.AdaBoostClassifier is used for the application of the classifier on real data in python.
![](images/adaboost.jpg)

Reources:

[Understanding](https://blog.paperspace.com/adaboost-optimizer/#:~:text=AdaBoost%20is%20an%20ensemble%20learning,turn%20them%20into%20strong%20ones.)


Gradient Boosting algorithm starts with a node giving 0.5 as output for both classification and regression. It serves as the first stump or weak learner. We then observe the Errors in predictions. Now, we create other learners or decision trees to actually predict the errors based on the conditions. The errors are called Residuals. Our final output is:

0.5 (Provided by the first learner) + The error provided by the second tree or learner.

Now, if we use this method, it learns the predictions too tightly, and loses generalization. In order to avoid that gradient boosting uses a learning parameter alpha. 

So, the final results after two learners is obtained as:

0.5 (Provided by the first learner) + alpha X (The error provided by the second tree or learner.)

We can see that using the added portion we take a small leap towards the correct results. We continue adding learners until the point we are very close to the actual value given by the training set.

Overall the equation becomes:


0.5 (Provided by the first learner) + alpha X (The error provided by the second tree or learner.)+ alpha X (The error provided by the third tree or learner.)


sklearn.ensemble.GradientBoostingClassifier used to apply gradient boosting in python

![GBM](https://www.elasticfeed.com/wp-content/uploads/09cc1168a39db0c0d6ea1c66d27ecfd3.jpg)

Resource:

[Guide](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d) 

## 17_ Naïves Bayes classifiers

The Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem.

Bayes theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event. It is given by:

![bayes](https://wikimedia.org/api/rest_v1/media/math/render/svg/87c061fe1c7430a5201eef3fa50f9d00eac78810)

Where P(A|B) is the probabaility of occurrence of A knowing B already occurred and P(B|A) is the probability of occurrence of B knowing A occurred.

[Scikit-learn Guide](https://github.com/abr-98/data-scientist-roadmap/edit/master/04_Machine-Learning/README.md)

There are mostly two types of Naive Bayes:

1. Gaussian Naive Bayes

2. Multinomial Naive Bayes.

#### Multinomial Naive Bayes

The method is used mostly for document classification. For example, classifying an article as sports article or say film magazine. It is also used for differentiating actual mails from spam mails. It uses the frequency of words used in different magazine to make a decision.

For example, the word "Dear" and "friends" are used a lot in actual mails and "offer" and "money" are used a lot in "Spam" mails. It calculates the prorbability of the occurrence of the words in case of actual mails and spam mails using the training examples. So, the probability of occurrence of "money" is much higher in case of spam mails and so on. 

Now, we calculate the probability of a mail being a spam mail using the occurrence of words in it. 

#### Gaussian Naive Bayes

When the predictors take up a continuous value and are not discrete, we assume that these values are sampled from a gaussian distribution.
![](images/1_AYsUOvPkgxe3j1tEj2lQbg.gif)

It links guassian distribution and Bayes theorem. 

Resources:

[GUIDE](https://youtu.be/H3EjCKtlVog)

## 18_ K-Nearest neighbor

K-nearest neighbour algorithm is the most basic and still essential algorithm. It is a memory based approach and not a model based one. 

KNN is used in both supervised and unsupervised learning. It simply locates the data points across the feature space and used distance as a similarity metrics.

Lesser the distance between two data points, more similar the points are. 

In K-NN classification algorithm, the point to classify is plotted on the feature space and classified as the class of its nearest K-neighbours. K is the user parameter. It gives the measure of how many points we should consider while deciding the label of the point concerned. If K is more than 1 we consider the label that is in majority.

If the dataset is very large, we can use a large k. The large k is less effected by noise and generates smooth boundaries. For small dataset, a small k must be used. A small k helps to notice the variation in boundaries better.

![knn](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/46117/versions/4/screenshot.jpg)

Resource:

[GUIDE](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)

## 19_ Logistic regression

Regression is one of the most important concepts used in machine learning.

[Guide to regression](https://towardsdatascience.com/a-deep-dive-into-the-concept-of-regression-fb912d427a2e)

Logistic Regression is the most used classification algorithm for linearly seperable datapoints. Logistic Regression is used when the dependent variable is categorical. 

It uses the linear regression equation:

Y= w1x1+w2x2+w3x3……..wkxk

in a modified format:

Y= 1/ 1+e^-(w1x1+w2x2+w3x3……..wkxk)

This modification ensures the value always stays between 0 and 1. Thus, making it feasible to be used for classification.

The above equation is called Sigmoid function. The function looks like:

![Logreg](https://miro.medium.com/max/700/1*HXCBO-Wx5XhuY_OwMl0Phw.png)

The loss fucnction used is called logloss or binary cross-entropy.

Loss= —Y_actual. log(h(x)) —(1 — Y_actual.log(1 — h(x)))

If Y_actual=1, the first part gives the error, else the second part.

![loss](https://miro.medium.com/max/700/1*GZiV3ph20z0N9QSwQTHKqg.png)

Logistic Regression is used for multiclass classification also. It uses softmax regresssion or One-vs-all logistic regression.

[Guide to logistic Regression](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)


sklearn.linear_model.LogisticRegression is used to apply logistic Regression in python.

## 20_ Ranking

## 21_ Linear regression

Regression tasks deal with predicting the value of a dependent variable from a set of independent variables i.e, the provided features. Say, we want to predict the price of a car. So, it becomes a dependent variable say Y, and the features like engine capacity, top speed, class, and company become the independent variables, which helps to frame the equation to obtain the price.


Now, if there is one feature say x. If the dependent variable y is linearly dependent on x, then it can be given by y=mx+c, where the m is the coefficient of the feature in the equation, c is the intercept or bias. Both M and C are the model parameters.

We use a loss function or cost function called Mean Square error of (MSE). It is given by the square of the difference between the actual and the predicted value of the dependent variable.

MSE=1/2m * (Y_actual — Y_pred)²

If we observe the function we will see its a parabola, i.e, the function is convex in nature. This convex function is the principle used in Gradient Descent to obtain the value of the model parameters

![loss](https://miro.medium.com/max/2238/1*Xgk6XI4kEcSmDaEAxqB1CA.png)

The image shows the loss function.

To get the correct estimate of the model parameters we use the method of Gradient Descent

[Guide to Gradient Descent](https://towardsdatascience.com/an-introduction-to-gradient-descent-and-backpropagation-81648bdb19b2)

[Guide to linear Regression](https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86)

sklearn.linear_model.LinearRegression is used to apply linear regression in python

## 22_ Perceptron

The perceptron has been the first model described in the 50ies.

This is a binary classifier, ie it can't separate more than 2 groups, and thoses groups have to be linearly separable.

The perceptron works like a biological neuron. It calculate an activation value, and if this value if positive, it returns 1, 0 otherwise.

## 23_ Hierarchical clustering

The hierarchical algorithms are so-called because they create tree-like structures to create clusters. These algorithms also use a distance-based approach for cluster creation.

The most popular algorithms are:

Agglomerative Hierarchical clustering

Divisive Hierarchical clustering

Agglomerative Hierarchical clustering: In this type of hierarchical clustering, each point initially starts as a cluster, and slowly the nearest or similar most clusters merge to create one cluster.

Divisive Hierarchical Clustering: The type of hierarchical clustering is just the opposite of Agglomerative clustering. In this type, all the points start as one large cluster and slowly the clusters get divided into smaller clusters based on how large the distance or less similarity is between the two clusters. We keep on dividing the clusters until all the points become individual clusters.

For agglomerative clustering, we keep on merging the clusters which are nearest or have a high similarity score to one cluster. So, if we define a cut-off or threshold score for the merging we will get multiple clusters instead of a single one. For instance, if we say the threshold similarity metrics score is 0.5, it means the algorithm will stop merging the clusters if no two clusters are found with a similarity score less than 0.5, and the number of clusters present at that step will give the final number of clusters that need to be created to the clusters.

Similarly, for divisive clustering, we divide the clusters based on the least similarity scores. So, if we define a score of 0.5, it will stop dividing or splitting if the similarity score between two clusters is less than or equal to 0.5. We will be left with a number of clusters and it won’t reduce to every point of the distribution.

The process is as shown below:

![HC](https://miro.medium.com/max/1000/1*4GRJvFaRdapnF3K4yH97DA.png)

One of the most used methods for the measuring distance and applying cutoff is the dendrogram method.

The dendogram for above clustering is:

![Dend](https://miro.medium.com/max/700/1*3TV7NtpSSFoqeX-p9wr1xw.png)

[Guide](https://towardsdatascience.com/understanding-the-concept-of-hierarchical-clustering-technique-c6e8243758ec)

## 24_ K-means clustering

The algorithm initially creates K clusters randomly using N data points and finds the mean of all the point values in a cluster for each cluster. So, for each cluster we find a central point or centroid calculating the mean of the values of the cluster. Then the algorithm calculates the sum of squared error (SSE) for each cluster. SSE is used to measure the quality of clusters. If a cluster has large distances between the points and the center, then the SSE will be high and if we check the interpretation it allows only points in the close vicinity to create clusters.

The algorithm works on the principle that the points lying close to a center of a cluster should be in that cluster. So, if a point x is closer to the center of cluster A than cluster B, then x will belong to cluster A. Thus a point enters a cluster and as even a single point moves from one cluster to another, the centroid changes and so does the SSE. We keep doing this until the SSE decreases and the centroid does not change anymore. After a certain number of shifts, the optimal clusters are found and the shifting stops as the centroids don’t change any more.

The initial number of clusters ‘K’ is a user parameter.

The image shows the method

![Kmeans](https://miro.medium.com/max/1000/1*lZdpqQxhcGyqztp_mvXi4w.png)

We have seen that for this type of clustering technique we need a user-defined parameter ‘K’ which defines the number of clusters that need to be created. Now, this is a very important parameter. To, find this parameter a number of methods are used. The most important and used method is the elbow method.
For smaller datasets, k=(N/2)^(1/2) or the square root of half of the number of points in the distribution.

[Guide](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)

## 25_ Neural networks

Neural Networks are a set of interconnected layers of artificial neurons or nodes. They are frameworks that are modeled keeping in mind, the structure and working of the human brain. They are meant for predictive modeling and applications where they can be trained via a dataset. They are based on self-learning algorithms and predict based on conclusions and complex relations derived from their training sets of information.

A typical Neural Network has a number of layers. The First Layer is called the Input Layer and The Last layer is called the Output Layer. The layers between the Input and Output layers are called Hidden Layers. It basically functions like a Black Box for prediction and classification. All the layers are interconnected and consist of numerous artificial neurons called Nodes.

[Guide to nueral Networks](https://medium.com/ai-in-plain-english/neural-networks-overview-e6ea484a474e)

Neural networks are too complex to work on Gradient Descent algorithms, so it works on the principles of Backproapagations and Optimizers.

[Guide to Backpropagation](https://towardsdatascience.com/an-introduction-to-gradient-descent-and-backpropagation-81648bdb19b2)

[Guide to optimizers](https://towardsdatascience.com/introduction-to-gradient-descent-weight-initiation-and-optimizers-ee9ae212723f)

## 26_ Sentiment analysis

Text Classification and sentiment analysis is a very common machine learning problem and is used in a lot of activities like product predictions, movie recommendations, and several others.

Text classification problems like sentimental analysis can be achieved in a number of ways using a number of algorithms. These are majorly divided into two main categories:

A bag of Word model: In this case, all the sentences in our dataset are tokenized to form a bag of words that denotes our vocabulary. Now each individual sentence or sample in our dataset is represented by that bag of words vector. This vector is called the feature vector. For example, ‘It is a sunny day’, and ‘The Sun rises in east’ are two sentences. The bag of words would be all the words in both the sentences uniquely.

The second method is based on a time series approach: Here each word is represented by an Individual vector. So, a sentence is represented as a vector of vectors.

[Guide to sentimental analysis](https://towardsdatascience.com/a-guide-to-text-classification-and-sentiment-analysis-2ab021796317)

## 27_ Collaborative filtering

We all have used services like Netflix, Amazon, and Youtube. These services use very sophisticated systems to recommend the best items to their users to make their experiences great. 

Recommenders mostly have 3 components mainly, out of which, one of the main component is Candidate generation. This method is responsible for generating smaller subsets of candidates to recommend to a user, given a huge pool of thousands of items.

Types of Candidate Generation Systems:

Content-based filtering System

Collaborative filtering System

Content-based filtering system: Content-Based recommender system tries to guess the features or behavior of a user given the item’s features, he/she reacts positively to.

Collaborative filtering System: Collaborative does not need the features of the items to be given. Every user and item is described by a feature vector or embedding.

It creates embedding for both users and items on its own. It embeds both users and items in the same embedding space.

It considers other users’ reactions while recommending a particular user. It notes which items a particular user likes and also the items that the users with behavior and likings like him/her likes, to recommend items to that user.

It collects user feedbacks on different items and uses them for recommendations.

[Guide to collaborative filtering](https://towardsdatascience.com/introduction-to-recommender-systems-1-971bd274f421)

## 28_ Tagging

## 29_ Support Vector Machine

Support vector machines are used for both Classification and Regressions. 

SVM uses a margin around its classifier or regressor. The margin provides an extra robustness and accuracy to the model and its performance.  

![SVM](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/300px-SVM_margin.png)

The above image describes a SVM classifier. The Red line is the actual classifier and the dotted lines show the boundary. The points that lie on the boundary actually decide the Margins. They support the classifier margins, so they are called Support Vectors.

The distance between the classifier and the nearest points is called Marginal Distance.

There can be several classifiers possible but we choose the one with the maximum marginal distance. So, the marginal distance and the support vectors help to choose the best classifier.

[Official Documentation from Sklearn](https://scikit-learn.org/stable/modules/svm.html)

[Guide to SVM](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)

## 30_Reinforcement Learning

“Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward.”

To play a game, we need to make multiple choices and predictions during the course of the game to achieve success, so they can be called a multiple decision processes. This is where we need a type of algorithm called reinforcement learning algorithms. The class of algorithm is based on decision-making chains which let such algorithms to support multiple decision processes.

The reinforcement algorithm can be used to reach a goal state from a starting state making decisions accordingly. 

The reinforcement learning involves an agent which learns on its own. If it makes a correct or good move that takes it towards the goal, it is positively rewarded, else not. This way the agent learns.

![reinforced](https://miro.medium.com/max/539/0*4d9KHTzW6xrWTBld)

The above image shows reinforcement learning setup.

[WIKI](https://en.wikipedia.org/wiki/Reinforcement_learning#:~:text=Reinforcement%20learning%20(RL)%20is%20an,supervised%20learning%20and%20unsupervised%20learning.)

