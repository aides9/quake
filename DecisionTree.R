library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(party)
library(MLmetrics)
library(tidyverse)

set.seed(1234)

setwd('~/desktop/data')

x_train <- read.csv("x_train.csv")
y_train <- read.csv("y_train.csv")
x_test <- read.csv("x_test.csv")
y_test <- read.csv("y_test.csv")

x_train$damage_grade = y_train$damage_grade
x_train$damage_grade <- as.factor(x_train$damage_grade)
system.time(tree_with_params <- rpart(damage_grade ~ ., data=x_train, method="class", 
                                      minsplit = 1, minbucket = 10, cp = -1))

plotcp(tree_with_params)

Predict = predict(tree_with_params, x_test,type = "class")

Confusion_matrix = confusionMatrix(Predict, reference=as.factor(y_test$damage_grade))
Confusion_matrix[["byClass"]][,"F1"]
Confusion_matrix

F1_Score_micro(y_test$damage_grade, predicted_labels$predicted_labels, labels = c(1,2,3))
#prp (tree_with_params)
#print(tree_with_params)
#summary(tree_with_params)
#plot(tree_with_params)
#text(tree_with_params)

text(tree_with_params, splits = FALSE, all = FALSE,
     pretty = NULL, digits = getOption("digits") - 3, )

fit <- rpart(Kyphosis ~ Age + Number + Start, data = kyphosis)
df <- data.frame(imp = tree_with_params$variable.importance)
df2 <- df %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))
ggplot2::ggplot(df2) +
  geom_col(aes(x = variable, y = imp),
           col = "black", show.legend = F) +
  coord_flip() +
  scale_fill_grey() +
  theme_bw()


