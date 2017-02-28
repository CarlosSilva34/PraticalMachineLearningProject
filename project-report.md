# Coursera Practical Machine Learning Project
Carlos Silva  
`r format(Sys.Date(), '%B %d, %Y')`  

# Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

## Project Goal

Predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.


# Data Processing

## Loadind the data

Load the R packages needed for analysis and then download the training and testing data sets


```r
library(caret); library(rattle); library(rpart);library(randomForest)
```


```r
set.seed(1235)
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```

## Reading the Data


```r
training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
dim(training)
```

```
## [1] 19622   160
```


```r
dim(testing)
```

```
## [1]  20 160
```

The training dataset has 19622 observations and 160 variables and the testing data set contains 20 observations and the same 160 variables.

## Data cleaning

All predictors of the training set that contain any missing values, will be deleted.


```r
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
```

Remove some columns with little predicting power for the outcome.


```r
trainClean <- training[, -c(1:7)]
testClean <- testing[, -c(1:7)]
dim(trainClean)
```

```
## [1] 19622    53
```

```r
dim(testClean)
```

```
## [1] 20 53
```

The cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables.

## Data spliting

Split the cleaned training set trainClean into a training set (train, 70%) for prediction and a validation set (test 30%).


```r
set.seed(5235) 
inTrain <- createDataPartition(trainClean$classe, p = 0.7, list = FALSE)
train <- trainClean[inTrain, ]
test <- trainClean[-inTrain, ]
```

# Data Modeling

Fit a predictive model using Random Forest algorith, because it is appropriate for a classification problem and tends to be more accurate than some other classification models. I will use 5-fold cross validation when applying the algorithm.



```r
controlRf <- trainControl(method="cv", 5)
modRf <- train(classe ~ ., data = train, method = "rf", 
                   trControl = controlRf)
modRf
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10989, 10990, 10989, 10991 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9910457  0.9886721
##   27    0.9914097  0.9891330
##   52    0.9866054  0.9830541
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

Estimate the performance of the model on the validation set.
 

```r
predictRf <- predict(modRf, test)
cMatrix<- confusionMatrix(test$classe, predictRf) 
```


```r
accuracyRf <- cMatrix$overall[1]
accuracyRf
```

```
##  Accuracy 
## 0.9908241
```


The estimated accuracy of the model is 99.1% and the estimated out-of-sample error is 0.9%

# Prediction on Testing Data Set

Finally, use random forests to predict the outcome variable classe for the testing set.


```r
predict(modRf, testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

