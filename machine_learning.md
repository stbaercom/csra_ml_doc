Machine Learning Assignment - Classifing Training Activities
========================================================

This is my writeup for the Coursera Machine Learning Assignment.

## Libraries

In addition to the caret package, I also install doMC to make use of all 4 cores in my PC. 
Given the time time the the random forst algorithm takes, this should save some time


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(doMC)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoMC(cores = 4)
```



## Loading Data

Loading the data is straight-forward. At this point, I define emtpy strings and "NA" as missing data.


```r
raw_train <- read.csv("pml-training.csv", na.strings = c("NA", ""))
raw_test <- read.csv("pml-testing.csv", na.strings = c("NA", ""))
```


## Data Preparation

I adapt some of the data preparation steps presented in the Coursera Disscussion Thread 
https://class.coursera.org/predmachlearn-002/forum/thread?thread_id=119
The steps replicate my own data preparation strategy, but are shorter.
The approach first finds all columns in the training dataset that only contain NAs and saves the 
indices of these columns. The coloums are then removed from both the training and the testing dataset.

In a second step, some further columns are removed, that are not important to the training task:

- The Timestamp
- The Username
- Information on the Windows


```r
NAs <- apply(raw_train, 2, function(x) {
    sum(is.na(x))
})
raw_train <- raw_train[, which(NAs == 0)]
raw_test <- raw_test[, which(NAs == 0)]
removeIndex <- grep("timestamp|X|user_name|new_window", names(raw_train))
train_all <- raw_train[, -removeIndex]
test_all <- raw_test[, -removeIndex]
```



## Data Preparation 2 - Sampling

Here, I take a sample of 20% of the data from the testset. I do this for two reason.

1) It makes for faster training. Random forest, especially with the cross-validation approach used
here, would otherwise take too long to train.
2) It allows me to test the out-of-sample accuracy of my model on the remaining datasets in the 
training set. Since random forest have a tendency towards overtraining, this should help.


```r
train_sample <- train_all[createDataPartition(y = train_all$classe, p = 0.2, 
    list = FALSE), ]
```



## Training the Model with Cross Validation

I use the trainControl function from the caret package to set up cross-validation. This will help to 
ensure that my model is not to overfitted. As the prediction model, I use random forest. I make this 
choice based on the discussion in the orginal paper "Qualitative Activity Recognition of 
Weight Lifting Exercises" by Velloso et al.



```r
ctrl <- trainControl(method = "cv", savePred = T)
mod <- train(classe ~ ., data = train_sample, method = "rf", trControl = ctrl)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```


##Applying the Model

I apply the model, both to the full training set and to the test set.


```r
p1 <- predict(mod, newdata = train_all)
p2 <- predict(mod, newdata = test_all)
```


## Thoughts on the out-of-sample Error

Looking at the predictions for the full training set against the real data, I see that the accuracy
of the predict for the full training set if above 98%. Given that the model was only trained on 20% 
of the training set, the random forest models works well for out of sample cases and I expect good 
performance on the test set


```r
xtab <- table(p1, train_all$classe)
xtab
```

```
##    
## p1     A    B    C    D    E
##   A 5579   61    0   10    5
##   B    0 3700   37   11   11
##   C    0   33 3371   50    1
##   D    0    3   14 3145   16
##   E    1    0    0    0 3574
```

```r
confusionMatrix(xtab)
```

```
## Confusion Matrix and Statistics
## 
##    
## p1     A    B    C    D    E
##   A 5579   61    0   10    5
##   B    0 3700   37   11   11
##   C    0   33 3371   50    1
##   D    0    3   14 3145   16
##   E    1    0    0    0 3574
## 
## Overall Statistics
##                                         
##                Accuracy : 0.987         
##                  95% CI : (0.985, 0.989)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.984         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.974    0.985    0.978    0.991
## Specificity             0.995    0.996    0.995    0.998    1.000
## Pos Pred Value          0.987    0.984    0.976    0.990    1.000
## Neg Pred Value          1.000    0.994    0.997    0.996    0.998
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.189    0.172    0.160    0.182
## Detection Prevalence    0.288    0.192    0.176    0.162    0.182
## Balanced Accuracy       0.997    0.985    0.990    0.988    0.995
```


## Writing Results to File

The following R code just writes the predict to disk.


```r
result_str <- paste0(p2, collapse = "")
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(p2)
```

