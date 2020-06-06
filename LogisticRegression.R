
library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(glmnet)

set.seed(1234)

setwd('~/desktop/data')
x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

#y_train$damage_grade <- as.factor(y_train$damage_grade)
classifier = glmnet(data.matrix(x_train[1:68]),factor(y_train[,1]),
                    type.measure = "class",
                    type.multinomial ="grouped",
                    alpha=0,
                    nfolds = 10,
                    family = "multinomial")
summary(classifier)
plot(classifier)

# Predicting the Test set results
prob_pred = predict(classifier,data.matrix(x_test[1:68]), type = "class", s=0.05 )
test_sparse<-as.data.frame(prob_pred[,-1]) #removing intercept
cm = table(y_test$damage_grade, prob_pred)
library(MLmetrics)

F1_Score(y_test$damage_grade, prob_pred)
