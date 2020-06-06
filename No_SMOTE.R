library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)

set.seed(1234)

setwd('~/desktop/data')

train <- read.csv("train_values.csv")
label <- read.csv("train_labels.csv")

train$damage_grade = label$damage_grade

trainIndex <- createDataPartition(train$damage_grade, p = .75, list = FALSE, times = 1)
train <- train[ trainIndex,]
test  <- train[-trainIndex,]
as.data.frame(table(train$damage_grade))
as.data.frame(table(test$damage_grade))


train$damage_grade = as.factor(train$damage_grade)
test$damage_grade = as.factor(test$damage_grade)

# library(intervals)
# library(gstat)
# library(UBL)
# trainSplit <- SmoteClassif(damage_grade ~ ., train, C.perc = "balance", dis="HEOM")
# as.data.frame(table(trainSplit$damage_grade))

y_train = as.data.frame(train$damage_grade)
x_train = subset(train, select = -c(damage_grade))
y_test = as.data.frame(test$damage_grade)
x_test = subset(test, select = -c(damage_grade))

colnames(y_train) <- c("damage_grade")
colnames(y_test) <- c("damage_grade")

num_data=c("geo_level_1_id", "geo_level_2_id", "geo_level_3_id", "area_percentage", "height_percentage",
           "count_floors_pre_eq", "age","count_families")
cat_data=c("land_surface_condition", "foundation_type", "roof_type",
           "ground_floor_type", "other_floor_type", "position","legal_ownership_status",
           "plan_configuration") 
bin_data=c("has_superstructure_adobe_mud",
           "has_superstructure_mud_mortar_stone", "has_superstructure_stone_flag",
           "has_superstructure_cement_mortar_stone",
           "has_superstructure_mud_mortar_brick",
           "has_superstructure_cement_mortar_brick", "has_superstructure_timber",
           "has_superstructure_bamboo", "has_superstructure_rc_non_engineered",
           "has_superstructure_rc_engineered", "has_superstructure_other",
           "has_secondary_use",
           "has_secondary_use_agriculture", "has_secondary_use_hotel",
           "has_secondary_use_rental", "has_secondary_use_institution",
           "has_secondary_use_school", "has_secondary_use_industry",
           "has_secondary_use_health_post", "has_secondary_use_gov_office",
           "has_secondary_use_use_police", "has_secondary_use_other")

library(CatEncoders)
library(data.table)
library(mltools)
preprocess_category <- function(train,test) {
  z <- one_hot(as.data.table(train))
  y <- one_hot(as.data.table(test))
  return (list(z,y))
}
preprocess_numeric <- function(train,test) {
  preProc <- preProcess(train, method=c("center", "scale"))
  z <- predict(preProc, train)
  y <- predict(preProc, test)
  return (list(z,y))
}

cat = preprocess_category(x_train[cat_data], x_test[cat_data])
num = preprocess_numeric(x_train[num_data], x_test[num_data])
final_x_train = cbind(cat[[1]], num[[1]], x_train[bin_data])
final_x_test = cbind(cat[[2]], num[[2]], x_test[bin_data])

write.csv(final_x_train,"No_SMOTE/x_train.csv", row.names=FALSE)
write.csv(final_x_test,"No_SMOTE/x_test.csv", row.names=FALSE)
write.csv(y_train,"No_SMOTE/y_train.csv", row.names=FALSE)
write.csv(y_test,"No_SMOTE/y_test.csv", row.names=FALSE)

#Decision Tree
library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(party)
library(MLmetrics)
library(tidyverse)

x_train <- read.csv("No_SMOTE/x_train.csv")
y_train <- read.csv("No_SMOTE/y_train.csv")
x_test <- read.csv("No_SMOTE/x_test.csv")
y_test <- read.csv("No_SMOTE/y_test.csv")

set.seed(1234)
x_train$damage_grade = y_train$damage_grade
x_train$damage_grade <- as.factor(x_train$damage_grade)

system.time(tree_with_params <- rpart(damage_grade ~ ., data=x_train, method="class", 
                                      minsplit = 20, minbucket = 7, cp = -1))
plotcp(tree_with_params)

Predict = predict(tree_with_params, x_test,type = "class")

Confusion_matrix = confusionMatrix(Predict, reference=as.factor(y_test$damage_grade))
Confusion_matrix[["byClass"]][,"F1"]
Confusion_matrix

#XGBoost
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

x_train <- read.csv("No_SMOTE/x_train.csv")
y_train <- read.csv("No_SMOTE/y_train.csv")
x_test <- read.csv("No_SMOTE/x_test.csv")
y_test <- read.csv("No_SMOTE/y_test.csv")

y_train$damage_grade <- as.numeric(y_train$damage_grade)-1
y_test$damage_grade <- as.numeric(y_test$damage_grade)-1

xgb.train = xgb.DMatrix(data=as.matrix(x_train),label=y_train$damage_grade)
xgb.test = xgb.DMatrix(data=as.matrix(x_test),label=y_test$damage_grade)

system.time(xgb.final <- xgboost(data =  xgb.train, eta = 0.1,max_depth = 15, nround=250,gamma=0,
                                 colsample_bytree = 0.6, seed = 1, eval_metric = "merror",  
                                 min_child_weight=1, subsample=0.8888889,
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


