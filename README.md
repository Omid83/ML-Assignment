# ML-Assignment
Machine learning assignment Coursera

*** FOR ALL PLOTS -> download word document or rmd file***

Prediction Assignment description

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable 
in the training set. You may use any of the other variables to predict with. You should create a report describing 
how you built your model, how you used cross validation, what you think the expected out of sample error is, and 
why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Selection of Models:

I've decided to make my analysis based on two models. Random Forest Model and Gradient Boosting Model. I will start by 
analysing the data quality in order to base the analysis on correct and clean data. Then I'll proceed to build the actual models on training-data and later on try and validate the models och test-data and validation-data. I will make final predictions on the model with best accuracy. Below you'll find the R-code:


---
title: "ML Assignment"
author: "Omid Sedighi"
date: "6 mars 2017"
output: html_document
---

```{r}
#LOAD NEEDED PACKAGES
library(caret)
library(randomForest)
library(rpart)
library(RColorBrewer)
library(gbm)




###LOAD DATA 
#Remove errors and blanks with NA
trainData <- read.csv("pml-training.csv", head=TRUE, sep=",", na.strings=c("NA","#DIV/0!","")) 
testData <- read.csv("pml-testing.csv", head=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))         


#Take a look at the data before pre-processing
dim(trainData)
dim(testData)
levels(trainData$classe)
summary(trainData$classe)

###Data pre-processing

#Dectecting missing vlaues
NApercentTrain <- sapply(trainData, function(df) {sum(is.na(df)==TRUE)/length(df)})
NApercentTest <- sapply(testData, function(df) {sum(is.na(df)==TRUE)/length(df)})

#The results from above show that a lot of variables have more than 97% missing values
#The below table shows this
table(NApercentTrain > .97)
table(NApercentTest > .97)


#Remove exceeding variables
removetrain <- names(which(NApercentTrain < 0.97))
trainData <- trainData[, removetrain]
removetest <- names(which(NApercentTest < 0.97))
testData <- testData[, removetest]

sum(is.na(trainData) == TRUE)
sum(is.na(testData) == TRUE)


#And if we look at the dimensions now
dim(trainData)
dim(testData)

#Now we can see that the first 7 variables are metadata and we which we can exklude

trainData <- trainData[,-c(1:7)]
testData <- testData[,-c(1:7)]
dim(trainData)
dim(testData)


###Data splitting for resampling

set.seed(123)
inTrain <- createDataPartition (trainData$classe, p=0.7, list=FALSE)
training <- trainData [inTrain ,]
testing <- trainData [- inTrain,]


###Model fitting

####Random Forest####
RandomForestModel <- randomForest(classe ~ ., data=training, ntree=453, mtry = 15)
```
```{r}
#Check error ratez§
plot(RandomForestModel)
#Find optimal number of trees and variables with smallest error rate
#which.min(RandomForestModel$err.rate[,1])
#tuneRF(training[,-53], training[,53], stepFactor=1.5)
```

```{r}
#Print and plot importance plot
print(RandomForestModel)
importance(RandomForestModel)

plot.new()
varImpPlot(RandomForestModel, pch=19, col=1, cex=1.0, main="")
abline(v=10000, col="blue")

#Predictions
RFpred <- predict(RandomForestModel, testing, type="response", decision.values = TRUE, probability = TRUE)

#Look at results from RF-model
confusionMatrix(data = RFpred, testing$classe)
```

```{r}
#####Gradient boosting classification#####

control = trainControl(method = "CV", number = 10)

#gbmtrain <- train(classe ~ ., data = training, method = "gbm", trControl=control)

#Set and train model
gbm <- gbm(classe ~ ., data = training, n.trees = 150, interaction.depth = 3, shrinkage = 0.1)
gbm

gbmtest = predict(gbm, newdata = testing, type = "response", n.trees = 150)

#Classify the final results from given probabilities of every outcome
gbmres <- apply(gbmtest, 1, which.max)

#Classify above to letters according to classe variable
gbmres2 <- ifelse(gbmres ==1, "A", ifelse(gbmres ==2, "B", ifelse(gbmres ==3, "C", ifelse(gbmres ==4, "D", "E"))))


#Print results
confusionMatrix(data = gbmres2, testing$classe)

summary(gbm)

#Predictions on real data - Final Model
testData$RFfinal <- predict(RandomForestModel, testData, type="response", decision.values = TRUE, probability = TRUE)
testData$RFfinal

###REPEAT SETTING MODELS WITH DIFFERENT SEEDS TO SEE IF THE OUTCOME IS THE SAME!!!
```






