library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(party)
library(randomForest)
library(MLmetrics)

set.seed(1234)

setwd('~/desktop/data')

x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

x_train$damage_grade = as.factor(y_train$damage_grade)
y_test$damage_grade = as.factor(y_test$damage_grade)

system.time(rf <- randomForest(y=x_train$damage_grade,x=x_train[0:68],ntree=100,sampsize=5000, 
                               mtry = 1, do.trace=100,importance = TRUE))

pred = predict(rf, x_test)
ypred = as.data.frame(ypred)
Confusion_matrix = confusionMatrix(ypred$ypred, reference=y_test$damage_grade)
Confusion_matrix[["byClass"]][,"F1"]
Confusion_matrix

getTree(rf, 1)
importance(rf)
varImpPlot(rf)
plot(rf)
# add legend to know which is which
legend("top", colnames(rf$err.rate), fill=1:ncol(rf$err.rate))
ranger::treeInfo(rf)