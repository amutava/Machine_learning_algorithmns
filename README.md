# Machine_learning_algorithmns.

#### Linear Regression:

Used to estimate real values based on continous variables. A relationship between a dependent and independent variable is established based on the best line of fit. The regression line is represented by a linear equation :<br/>
  ** Y = aX + b ** <br/>

In this equation:<br/>

* Y – Dependent Variable

* a – Slope

* X – Independent variable

* b – Intercept

These coefficients a and b are derived based on minimizing the sum of squared difference of distance between data points and regression line. In maths we used to call it slope. <br/>

There are two types of linear regression. Simple and multiple linear regression. The naming is based on the number of independent variables. <br/>

Mostly used in predictive modelling.The representation of linear regression is an equation that describes a line that best fits the relationship between the input variables (x) and the output variables (y), by finding specific weightings for the input variables called coefficients (a, b). <br/>
Some good rules of thumb when using this technique are to remove variables that are very similar (correlated) and to remove noise from your data, if possible. It is a fast and simple technique and good first algorithm to try.
Squared loss: a popular loss function
The linear regression models we'll examine here use a loss function called squared loss (also known as L2 loss). The squared loss for a single example is as follows:

```
  = the square of the difference between the label and the prediction
  = (observation - prediction(x))2
  = (y - y')2
```
Mean square error (MSE) is the average squared loss per example over the whole dataset. To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples:


#### Logistic Regression:

This classification is used to estimate discrete values(binary values...) based on the set of independent variables. <br/>
It predicts the probability of occurence of an event by fitting data to a logit function. This classification is at times called logit regression. It predicts probability so it's values lie between 0 and 1.<br/>
The log odds of the outcome is modeled as a linear combination of the predictor variables. <br/>

Logistic regression is like linear regression in that the goal is to find the values for the coefficients that weight each input variable. Unlike linear regression, the prediction for the output is transformed using a non-linear function called the logistic function. <br/>
The logistic function looks like a big S and will transform any value into the range 0 to 1. This is useful because we can apply a rule to the output of the logistic function to snap values to 0 and 1. <br/>
Because of the way that the model is learned, the predictions made by logistic regression can also be used as the probability of a given data instance belonging to class 0 or class 1. This can be useful for problems where you need to give more rationale for a prediction.

Like linear regression, logistic regression does work better when you remove attributes that are unrelated to the output variable as well as attributes that are very similar (correlated) to each other. It’s a fast model to learn and effective on binary classification problems.

#### Decision Tree:

This is a supervised learning algorithmn that is mostly used for classification problems. It works for both categorical and continous dependent variables. In this algorithm, we split the population into two or more homogeneous sets. This is done based on most significant attributes/ independent variables to make as distinct groups as possible.<br/>
Decision trees work in the  fashion by dividing a population in as different groups as possible.


###3 SVM (Support Vector Machines):

It is a classification method. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate.<br/>
For example, if we only had two features like Height and Hair length of an individual, we’d first plot these two variables in two dimensional space where each point has two co-ordinates (these co-ordinates are known as Support Vectors).<br/>
Now, we will find some line that splits the data between the two differently classified groups of data. This will be the line such that the distances from the closest point in each of the two groups will be farthest away.<br/>
A hyperplane is a line that splits the input variable space. In SVM, a hyperplane is selected to best separate the points in the input variable space by their class, either class 0 or class 1. In two-dimensions, you can visualize this as a line and let’s assume that all of our input points can be completely separated by this line. The SVM learning algorithm finds the coefficients that results in the best separation of the classes by the hyperplane.<br/>

#### Naive Bayes:

It is a classification technique based on Bayes’ theorem with an assumption of independence between predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. <br/>
Naive Bayesian model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.<br/>

Bayes theorem provides a way of calculating posterior probability P(c|x) from P(c), P(x) and P(x|c). Look at the equation below:<br/>

#### kNN(k- Nearest Neighbors):

It can be used for both classification and regression problems. K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases by a majority vote of its k neighbors.<br/>
The case being assigned to the class is most common amongst its K nearest neighbors measured by a distance function.<br/>
KNN can easily be mapped to our real lives. If you want to learn about a person, of whom you have no information, you might like to find out about his close friends and the circles he moves in and gain access to his/her information!<br/>

###### Things to consider before selecting kNN:

* KNN is computationally expensive. 

* Variables should be normalized else higher range variables can bias it. 

* Works on pre-processing stage more before going for kNN like outlier, noise removal. 

#### K-Means:

It is a type of unsupervised algorithm which  solves the clustering problem. Its procedure follows a simple and easy  way to classify a given data set through a certain number of  clusters. (assume k clusters) Data points inside a cluster are homogeneous and heterogeneous to peer groups.<br/>

###### How K-means forms cluster:

* K-means picks k number of points for each cluster known as centroids. 

* Each data point forms a cluster with the closest centroids i.e. k clusters. 

* Finds the centroid of each cluster based on existing cluster members. Here we have new centroids.

* As we have new centroids, repeat step 2 and 3. Find the closest distance for each data point from new centroids and get associated with new k-clusters. Repeat this process until convergence occurs i.e. centroids does not change. <br/>

###### How to determine value of K:

In K-means, we have clusters and each cluster has its own centroid. Sum of square of difference between centroid and the data points within a cluster constitutes within sum of square value for that cluster. Also, when the sum of square values for all the clusters are added, it becomes total within sum of square value for the cluster solution.<br/>

We know that as the number of cluster increases, this value keeps on decreasing but if you plot the result you may see that the sum of squared distance decreases sharply up to some value of k, and then much more slowly after that. Here, we can find the optimum number of cluster. <br/>

#### Random Forest:
Random Forest is one of the most popular and most powerful machine learning algorithms. It is a type of ensemble machine learning algorithm called Bootstrap Aggregation or bagging. <br/>
The bootstrap is a powerful statistical method for estimating a quantity from a data sample. Such as a mean. You take lots of samples of your data, calculate the mean, then average all of your mean values to give you a better estimation of the true mean value. </br>
In bagging, the same approach is used, but instead for estimating entire statistical models, most commonly decision trees. Multiple samples of your training data are taken then models are constructed for each data sample. When you need to make a prediction for new data, each model makes a prediction and the predictions are averaged to give a better estimate of the true output value. <br/>

#### Dimensionality Reduction Algorithms:

In the last 4-5 years, there has been an exponential increase in data capturing at every possible stages. Corporates/ Government Agencies/ Research organisations are not only coming with new sources but also they are capturing data in great detail. <br/>
For example: E-commerce companies are capturing more details about customer like their demographics, web crawling history, what they like or dislike, purchase history, feedback and many others to give them personalized attention more than your nearest grocery shopkeeper. <br/>
As a data scientist, the data we are offered also consist of many features, this sounds good for building good robust model but there is a challenge. How’d you identify highly significant variable(s) out 1000 or 2000? In such cases, dimensionality reduction algorithm helps us along with various other algorithms like Decision Tree, Random Forest, PCA, Factor Analysis, Identify based on correlation matrix, missing value ratio and others. <br/>

#### Gradient Boosting Algorithms:

* GBM

GBM is a boosting algorithm used when we deal with plenty of data to make a prediction with high prediction power. Boosting is actually an ensemble of learning algorithms which combines the prediction of several base estimators in order to improve robustness over a single estimator. It combines multiple weak or average predictors to a build strong predictor. <br/>

* XGBoost

Another classic gradient boosting algorithm that’s known to be the decisive choice between winning and losing in some Kaggle competitions. <br/>
The XGBoost has an immensely high predictive power which makes it the best choice for accuracy in events as it possesses both linear model and the tree learning algorithm, making the algorithm almost 10x faster than existing gradient booster techniques. <br/>

The support includes various objective functions, including regression, classification and ranking. One of the most interesting things about the XGBoost is that it is also called a regularized boosting technique. <br/>

* LightGBM

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:<br/>

* Faster training speed and higher efficiency.  

* Lower memory usage

* Better accuracy

* Parallel and GPU learning supported

* Capable of handling large-scale data

The framework is a fast and high-performance gradient boosting one based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.<br/>

Since the LightGBM is based on decision tree algorithms, it splits the tree leaf wise with the best fit whereas other boosting algorithms split the tree depth wise or level wise rather than leaf-wise. So when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms.

* Catboost:

CatBoost is a recently open-sourced machine learning algorithm from Yandex. It can easily integrate with deep learning frameworks like Google’s TensorFlow and Apple’s Core ML. <br/>
The best part about CatBoost is that it does not require extensive data training like other ML models, and can work on a variety of data formats; not undermining how robust it can be.<br/>
Make sure you handle missing data well before you proceed with the implementation.<br/>
Catboost can automatically deal with categorical variables without showing the type conversion error, which helps you to focus on tuning your model better rather than sorting out trivial errors.



#### Reducing the loss of machine learning algorithmns:
###### An Iterative Approach.
 Iterative strategies are prevalent in machine learning, primarily because they scale so well to large data sets.
 The "model" takes one or more features as input and returns one prediction (y') as output. To simplify, consider a model that takes one feature and returns one prediction:
 	```
 		y = b + wx

	```
The "Compute Loss" part of the diagram is the loss function that the model will use. Suppose we use the squared loss function. The loss function takes in two input values:<br/>

	*  y': The model's prediction for features x
	*  y: The correct label corresponding to features x.
At last, we've reached the "Compute parameter updates" part of the diagram. It is here that the machine learning system examines the value of the loss function and generates new values for  and . For now, just assume that this mysterious box devises new values and then the machine learning system re-evaluates all those features against all those labels, yielding a new value for the loss function, which yields new parameter values. And the learning continues iterating until the algorithm discovers the model parameters with the lowest possible loss. Usually, you iterate until overall loss stops changing or at least changes extremely slowly. When that happens, we say that the model has converged.

###### Gradient Descent.
Suppose we had the time and the computing resources to calculate the loss for all possible values of . For the kind of regression problems we've been examining, the resulting plot of loss vs.  will always be convex. In other words, the plot will always be bowl-shaped, kind of like this: <br/>

Convex problems have only one minimum; that is, only one place where the slope is exactly 0. That minimum is where the loss function converges.<br/>
Calculating the loss function for every conceivable value of  over the entire data set would be an inefficient way of finding the convergence point. Let's examine a better mechanism—very popular in machine learning—called gradient descent.<br/>
The first stage in gradient descent is to pick a starting value (a starting point) for . The starting point doesn't matter much; therefore, many algorithms simply set  to 0 or pick a random value.<br/>
The gradient descent algorithm then calculates the gradient of the loss curve at the starting point. Here in Figure 3, the gradient of loss is equal to the derivative (slope) of the curve, and tells you which way is "warmer" or "colder." When there are multiple weights, the gradient is a vector of partial derivatives with respect to the weights.<br/>

Note that a gradient is a vector, so it has both of the following characteristics:

	* a direction
	* a magnitude
The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible <br/>

###### Learning Parameters.
The gradient vector has both a direction and a magnitude. Gradient descent algorithms multiply the gradient by a scalar known as the learning rate (also sometimes called step size) to determine the next point. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.<br/>
Hyperparameters are the knobs that programmers tweak in machine learning algorithms. Most machine learning programmers spend a fair amount of time tuning the learning rate. If you pick a learning rate that is too small, learning will take too long:<br/>

Conversely, if you specify a learning rate that is too large, the next point will perpetually bounce haphazardly across the bottom of the well like a quantum mechanics experiment gone horribly wrong:<br/>

There's a Goldilocks learning rate for every regression problem. The Goldilocks value is related to how flat the loss function is. If you know the gradient of the loss function is small then you can safely try a larger learning rate, which compensates for the small gradient and results in a larger step size.<br/>