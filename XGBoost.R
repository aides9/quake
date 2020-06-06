library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(party)
library(MLmetrics)
library(xgboost)
library(data.table)

set.seed(1234)

setwd('~/desktop/data')

x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

y_train$damage_grade <- as.numeric(y_train$damage_grade)-1
y_test$damage_grade <- as.numeric(y_test$damage_grade)-1

xgb.train = xgb.DMatrix(data=as.matrix(x_train),label=y_train$damage_grade)
xgb.test = xgb.DMatrix(data=as.matrix(x_test),label=y_test$damage_grade)

system.time(xgb <- xgboost(data =  xgb.train, eta = 0.1,max_depth = 15, nround=25, subsample = 0.5,
                           colsample_bytree = 0.5, seed = 1, eval_metric = "merror", 
                           objective = "multi:softprob", num_class = 3,nthread = 3))

Predict <- predict(xgb,  xgb.test, reshape=T)
predicted_labels= factor(max.col(Predict),levels=1:3)
predicted_labels = as.data.frame(predicted_labels)
Confusion_matrix = confusionMatrix(predicted_labels$predicted_labels, reference=as.factor(y_test$damage_grade+1))
Confusion_matrix[["byClass"]][,"F1"]
Confusion_matrix

F1_Score_micro(y_test$damage_grade+1, predicted_labels$predicted_labels, labels = c(1,2,3))


model <- xgb.dump(xgb, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model

importance_matrix <- xgb.importance(model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:20,])
