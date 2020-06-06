library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(party)
library(MLmetrics)
library(xgboost)
library(tidyverse)

set.seed(1234)

setwd('~/desktop/data')

x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

y_train$damage_grade <- as.numeric(y_train$damage_grade)-1
y_test$damage_grade <- as.numeric(y_test$damage_grade)-1
x_train$damage_grade <- as.factor(y_train$damage_grade)
x_train = x_train[sample(nrow(x_train), 1000), ]
y_train = as.factor(x_train$damage_grade)
x_train=select(x_train, -69)

#Parameter search
xgb.grid <- expand.grid(nrounds = 2, 
                          max_depth = c(5, 10, 15), 
                          eta = c(0.01, 0.001, 0.0001), 
                          gamma = c(1, 2, 3), 
                          colsample_bytree = c(0.4, 0.7, 1.0), 
                          min_child_weight = c(0.5, 1, 1.5),
                          subsample = 1)
trctrl <- trainControl(method = "cv", number = 5, search="grid",allowParallel = TRUE)
system.time(xgb <- train(x =as.matrix(x_train), y=y_train,method="xgbTree",
                                      trControl=trctrl,metric = "Accuracy", tuneLength=10,
                                      parms=list(split='information'), verbose=TRUE))

#Manual Fit parameter to model
x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

y_train$damage_grade <- as.numeric(y_train$damage_grade)-1
y_test$damage_grade <- as.numeric(y_test$damage_grade)-1

xgb.train = xgb.DMatrix(data=as.matrix(x_train),label=y_train$damage_grade)
xgb.test = xgb.DMatrix(data=as.matrix(x_test),label=y_test$damage_grade)

system.time(xgb.final <- xgboost(data =  xgb.train, eta = 0.3,max_depth = 2, nround=250,gamma=0,
                           colsample_bytree = 0.6, seed = 1, eval_metric = "merror",  min_child_weight=1, subsample=0.8888889,
                           objective = "multi:softprob", num_class = 3,nthread = 3))
Predict <- predict(xgb.final,  xgb.test, reshape=T)
predicted_labels= factor(max.col(Predict),levels=1:3)

model <- xgb.dump(xgb.final, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model

importance_matrix <- xgb.importance(model = xgb.final)
# Nice graph
xgb.plot.importance(importance_matrix[1:20,])

predicted_labels = as.data.frame(predicted_labels)
Confusion_matrix = confusionMatrix(predicted_labels$predicted_labels, reference=as.factor(y_test$damage_grade+1))
Confusion_matrix[["byClass"]][,"F1"]
Confusion_matrix

#Manual Fit parameter to model (with adjusted parameters)
x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

y_train$damage_grade <- as.numeric(y_train$damage_grade)-1
y_test$damage_grade <- as.numeric(y_test$damage_grade)-1

xgb.train = xgb.DMatrix(data=as.matrix(x_train),label=y_train$damage_grade)
xgb.test = xgb.DMatrix(data=as.matrix(x_test),label=y_test$damage_grade)

system.time(xgb.final <- xgboost(data =  xgb.train, eta = 0.1,max_depth = 15, nround=250,gamma=0,
                                 colsample_bytree = 0.6, seed = 1, eval_metric = "merror",  min_child_weight=1, subsample=0.8888889,
                                 objective = "multi:softprob", num_class = 3,nthread = 3))
Predict <- predict(xgb.final,  xgb.test, reshape=T)
predicted_labels= factor(max.col(Predict),levels=1:3)

model <- xgb.dump(xgb.final, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model

importance_matrix <- xgb.importance(model = xgb.final)
# Nice graph
xgb.plot.importance(importance_matrix[1:20,])

predicted_labels = as.data.frame(predicted_labels)
Confusion_matrix = confusionMatrix(predicted_labels$predicted_labels, reference=as.factor(y_test$damage_grade+1))
Confusion_matrix[["byClass"]][,"F1"]
Confusion_matrix



