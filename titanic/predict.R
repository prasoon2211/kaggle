library("mice")
library("caret")
library("xgboost")

setwd("~/work/kaggle/titanic")
set.seed(87)
train_df <- read.csv("train.csv", na.strings = c(""))
test_df <- read.csv("test.csv", na.strings = c(""))
test_df$Survived <- 0
tune <- T

preproc <- function(df) {
  df$Title <- sapply(df$Name, function(x) trimws(strsplit(strsplit(as.character(x), ",")[[1]][2], ".", fixed = T)[[1]][1]))
  df$Title[df$Title %in% c('Mme', 'Mlle', 'Miss', 'Ms')] <- 'Miss'
  df$Title[df$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Rev', 'Col')] <- 'Sir'
  df$Title[df$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
  df$Title <- as.factor(df$Title)
  
  imp.data <- mice(subset(df, select = -c(Cabin, Ticket, PassengerId, Survived, Name)), m = 1, maxit = 10)
  comp.df <- complete(imp.data, 1)
  comp.df$Family <- comp.df$SibSp + comp.df$Parch + 1
  comp.df <- subset(comp.df, select = -c(SibSp, Parch))
  dummy.model <- dummyVars(~ ., data = comp.df)
  comp.df <- as.data.frame(predict(dummy.model, comp.df))
  comp.df[, c("PassengerId", "Survived")] <- df[, c("PassengerId", "Survived")]
  comp.df$Survived <- comp.df$Survived
  return(comp.df)
}

train <- preproc(train_df)
test <- preproc(test_df)
x_train <- as.matrix(subset(train, select = -c(PassengerId, Survived)))
y_train <- train$Survived
x_test <- as.matrix(subset(test, select = -c(PassengerId, Survived)))

if(tune){
  dtrain <- xgb.DMatrix(x_train, label = y_train)
  grid <- expand.grid(nrounds = c(10, 20, 30, 40, 100),
                          eta = c(0.05,0.1, 0.3),
                          max_depth = c(2,4,6,8,10,14, 20),
                          subsample = c(0.5, 0.7, 0.9))
  grid[, c("err", "err_ind")] <- NA
  for(i in 1:nrow(grid)) {
    row <- grid[i, ]
    cv <- xgb.cv(data = dtrain, eta = row$eta, nthread = 4, subsample = row$subsample,
                   max_depth = row$max_depth, nrounds = row$nrounds, objective = "binary:logistic",
                   verbose = 0, nfold = 5, metrics = c("logloss"))
    ll <- min(cv$evaluation_log[, "test_logloss_mean"])
    ind_ll <- which.min(cv$evaluation_log$test_logloss_mean)
    grid[i, c("err", "err_ind")] <- c(ll, ind_ll)
  }
}

clf <- xgboost(data = x_train, label = y_train, eta = 0.3, nthread = 4, subsample = 0.5,
               max_depth = 6, nrounds = 13, objective = "binary:logistic",
               verbose = 1)
pred_df <- predict(clf, x_test)
test$Survived <- ifelse(pred_df > 0.5, 1, 0)

pred <- test[, c("PassengerId", "Survived")]
write.csv(pred, "predict-R.csv", row.names = F)
