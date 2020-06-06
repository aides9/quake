library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(party)
library(randomForest)
library(MLmetrics)
library(tidyverse)

set.seed(1234)

setwd('~/desktop/data')

x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")


x_train$damage_grade <- as.factor(y_train$damage_grade)
x_train = x_train[sample(nrow(x_train), 5000), ]
y_train = as.factor(x_train$damage_grade)
x_train=select(x_train, -69)

#Search parameter
y_test$damage_grade = as.factor(y_test$damage_grade)
rpart.grid <- expand.grid(mtry=c(1:5), ntree=c(200, 500)) 
trctrl <- trainControl(method = "oob", number = 10, search="grid")
system.time(tree_with_params <- train(x=x_train, y=y_train, method="rf",
                                      trControl=trctrl,metric = "Accuracy", tuneLength=10,
                                      parms=list(split='information')))
summary(tree_with_params)
plot(tree_with_params)

#Fit parameter to model
x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")
set.seed(1234)
x_train$damage_grade = as.factor(y_train$damage_grade)
y_test$damage_grade = as.factor(y_test$damage_grade)
system.time(rf <- randomForest(y=x_train$damage_grade,x=x_train[0:68],ntree=500,sampsize=5000,
                mtry =tree_with_params$bestTune$mtry,
		do.trace=100,importance = TRUE))
ypred = predict(rf, x_test)
y_test$damage_grade <- as.factor(y_test$damage_grade)
ypred = as.data.frame(ypred)
Confusion_matrix = confusionMatrix(ypred$ypred, reference=y_test$damage_grade)
Confusion_matrix[["byClass"]][,"F1"]
Confusion_matrix
F1_Score(y_test$damage_grade, ypred$ypred)
getTree(rf, 1)
importance(rf)
varImpPlot(rf)
plot(rf)