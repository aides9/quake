library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)

set.seed(1234)

setwd('~/desktop/data')

train <- read.csv("train_values.csv")
label <- read.csv("train_labels.csv")

train$damage_grade = label$damage_grade

#Train test split
trainIndex <- createDataPartition(train$damage_grade, p = .75, list = FALSE, times = 1)
train <- train[ trainIndex,]
test  <- train[-trainIndex,]
as.data.frame(table(train$damage_grade))
as.data.frame(table(test$damage_grade))


train$damage_grade = as.factor(train$damage_grade)
test$damage_grade = as.factor(test$damage_grade)

#Smoteclassif
library(intervals)
library(gstat)
library(UBL)
trainSplit <- SmoteClassif(damage_grade ~ ., train, C.perc = "balance", dis="HEOM")

#feature label split
y_train = as.data.frame(trainSplit$damage_grade)
x_train = subset(trainSplit, select = -c(damage_grade))
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

#Encoding and scaling
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

#Export
write.csv(final_x_train,"x_train.csv", row.names=FALSE)
write.csv(final_x_test,"x_test.csv", row.names=FALSE)
write.csv(y_train,"y_train.csv", row.names=FALSE)
write.csv(y_test,"y_test.csv", row.names=FALSE)

