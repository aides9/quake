library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(party)
library(MLmetrics)

set.seed(1234)

setwd('~/desktop/data')

x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

y_train$damage_grade <- as.factor(y_train$damage_grade)

#Search parameter
rpart.grid <- expand.grid(cp=seq(-1,0,0.5)) 
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
system.time(tree_with_params <- train(x=x_train, y=y_train$damage_grade, method="rpart",
                                trControl=trctrl,metric = "Accuracy", tuneLength=50,
                                parms=list(split='information')))

#Fit parmeter to model
x_train$damage_grade = y_train$damage_grade
x_train$damage_grade <- as.factor(x_train$damage_grade)
system.time(tree_with_params <- rpart(damage_grade ~ ., data=x_train, method="class", 
                                      minsplit = 20, minbucket = 7, cp = -1))
ypred = predict(tree_with_params, x_test, type = "class")

y_test$damage_grade <- as.factor(y_test$damage_grade)
ypred = as.data.frame(ypred)
Confusion_matrix = confusionMatrix(ypred$ypred, reference=y_test$damage_grade)
Confusion_matrix[["byClass"]][,"F1"]
Confusion_matrix

