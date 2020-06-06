library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)

set.seed(1234)

setwd('~/desktop/data')
train <- read.csv("train_values.csv")
label <- read.csv("train_labels.csv")

train$damage_grade = label$damage_grade

library(DataExplorer)
config <- configure_report(
  global_ggtheme = quote(theme_minimal(base_size = 14))
)

create_report(train, y = "damage_grade", config = config)

View(train)
names(train)
dim(train)
str(train)
head(train)

#QQ Plot
qq_data <- train[, c('geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'area_percentage', 'height_percentage',
                     'count_floors_pre_eq', 'age','count_families')]

plot_qq(qq_data, by = "name_origin", sampled_rows = 1000L)


#Correlation Plot
plot_correlation(na.omit(train), type = "c", maxcat = 5L)

#PCA Plot
plot_prcomp(qq_data, variance_cap = 0.9, nrow = 2L, ncol = 2L)

library(inspectdf)
library(tidyverse)
library(readr)
library(ggfortify)
train$damage_grade

prcomp(train$damage_grade)
train.pca <- prcomp(train[,c('geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'area_percentage', 'height_percentage',
                             'count_floors_pre_eq', 'age','count_families')], center = TRUE,scale. = TRUE)
str(train.pca)
ggbiplot(train[,c('geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'area_percentage', 'height_percentage',
                  'count_floors_pre_eq', 'age','count_families')])
model <- prcomp(train[,c('geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'area_percentage', 'height_percentage',
                'count_floors_pre_eq', 'age','count_families')], center = TRUE,scale. = TRUE)

model$sdev
model$rotation
model$center
model$scale

library(psych)
library(corrplot)
library(dplyr)
library(ggplot2)
library(nFactors)
corr <- cor(train[, c('geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'area_percentage', 'height_percentage',
                          'count_floors_pre_eq', 'age','count_families')])
cortest.bartlett(corr)
corrplot(corr, method = "shade")

pca = principal(corr, nfactors = 6, rotate = "varimax")
print(pca$loadings, cutoff = 0.6)


#BoxPlot
plot_boxplot(train[, c('geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'area_percentage', 'height_percentage',
                       'count_floors_pre_eq', 'age','count_families','damage_grade')], by = "damage_grade")


train$damage_grade