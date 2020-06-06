library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(party)
library(MLmetrics)
library(caret)
library(tidyverse)
library(randomForest)

set.seed(1234)

setwd('~/desktop/data')

x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

x_train$damage_grade = y_train$damage_grade
x_train = x_train[sample(nrow(x_train), 2000), ]
y_train = x_train$damage_grade
x_train=select(x_train, -69)

#RFE
ctrl_param <- rfeControl(functions = rfFuncs,
                         method = "cv",
                         repeats = 1,
                         verbose = TRUE)

rfe_rf_profile <- rfe(x_train, as.matrix(y_train),
                      sizes = c(1:68),
                      rfeControl = ctrl_param)

varImp(rfe_lm_profile)


#Fit feature list to model
x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

x_train$damage_grade = as.factor(y_train$damage_grade)
y_test$damage_grade = as.factor(y_test$damage_grade)

feature_list <- rfe_rf_profile$optVariables
system.time(rf <- randomForest(y=x_train$damage_grade,x=x_train[feature_list],ntree=500,
                               sampsize=5000, mtry =38, do.trace=100,importance = TRUE))

ypred = predict(rf, x_test[feature_list])
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

