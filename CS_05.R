# Case-Study Title: Sports Analytics (Regularization and DecisionTree based approaches for Regression problems)
# Data Analysis methodology: CRISP-DM
# Dataset: Hitters dataset (Major League Baseball Data from the 1986 and 1987 seasons in US)
# Case Goal: Annual Salary prediction of each Player in 1987 base on his performance in 1986


### Required Libraries ----
install.packages('glmnet')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('randomForest')
library('glmnet')
library('rpart')
library('rpart.plot')
library('randomForest')


### Read Data from File ----
load('dataset_v1.RData')  # pre-processed Data
dim(data2)  # 263 records, 19 variables


### Step 1: Business Understanding ----
 # know business process and issues
 # know the context of the problem
 # know the order of numbers in the business


### Step 2: Data Understanding ----
### Step 2.1: Data Inspection (Data Understanding from Free Perspective) ----
## Dataset variables definition
colnames(data2)  # KPI (Key Performance Indicator) variables

#indexes that measure player performance in league 1986
#Hits:       Number of hits in 1986
#HmRun:      Number of home runs in 1986
#Runs:       Number of runs in 1986
#RBI:        Number of runs batted in in 1986
#Walks:      Number of walks in 1986
#PutOuts:    Number of put outs in 1986
#Assists:    Number of assists in 1986
#Errors:     Number of errors in 1986

#indexes that measure player performance in whole his career
#Years:      Number of years in the major leagues
#CAtBat:     Number of times at bat during his career
#CHits:      Number of hits during his career
#CHmRun:     Number of home runs during his career
#CRuns:      Number of runs during his career
#CRBI:       Number of runs batted in during his career
#CWalks:     Number of walks during his career

#categorical variables
#League:     A factor with levels A and N indicating player's league at the end of 1986
#Division:   A factor with levels E and W indicating player's division at the end of 1986
#NewLeague:  A factor with levels A and N indicating player's league at the beginning of 1987

#target variable
#Salary:     1987 annual salary on opening day in thousands of dollars


### Step 2.2: Data Exploring (Data Understanding from Statistical Perspective) ----
## Overview of Dataframe
class(data2)
head(data2)
tail(data2)
str(data2)
summary(data2)


### Step 3: Data PreProcessing ----
# Divide Dataset into Train and Test randomly
head(train)
dim(train)  # 18 predictor variables
str(train)

head(test)
dim(test)
str(test)


### Step 4: Modeling ----
#results of 4 created models:
#1. Linear Regression with all 18 variables
#2. BestSubSelection base on statistical indexes (BIC, CP, Adjusted-R2)
#3. BestSubSelection base on k-fold cross validation
#4. BestSubSelection with trimmed data

View(models_comp)  # models comparison

# Model 5: Ridge Regression
x <- model.matrix(Log_Salary ~ . - Salary, data = train)[,-1]  # remove intercept
y <- train$Log_Salary

lambda_ridge_grid <- 10 ^ seq(10, -2, length.out = 100)
lambda_ridge_grid

ridgereg_1 <- glmnet::glmnet(x, y, alpha = 0, lambda = lambda_ridge_grid)
dim(coef(ridgereg_1))  # 19 coefficients (18 predictors + 1 intercept) in 100 iteration (different lambdas)

plot(ridgereg_1, xvar = 'lambda')  # plot Regression Coefficients vs. Log lambda

ridgereg_1$lambda[50]  # lambda value with index 50
coef(ridgereg_1)[,50]  # retrieve one of Regressions Coefficients (Regression Coefficients for lambda 11497.57)

#k-fold cross validation approach to choose the best model (select the best lambda)
set.seed(1234)
ridge_cv <- cv.glmnet(x, y, alpha = 0, nfolds = 10, lambda = lambda_ridge_grid)
ridge_cv

ridge_cv$cvm  # the mean cross-validated error (per every lambda)
ridge_cv$cvsd  # estimate of standard deviation error of cvm
ridge_cv$lambda.min  # value of lambda that gives the minimum cvm

ridgereg_2 <- glmnet(x, y, alpha = 0, lambda = ridge_cv$lambda.min)
coef(ridgereg_2)  # coefficients of regression with best lambda value

# Model 6: LASSO Regression
lassoreg_1 <- glmnet(x, y, alpha = 1, lambda = lambda_ridge_grid)
dim(coef(lassoreg_1))

plot(lassoreg_1, xvar = 'lambda')  # plot Regression Coefficients vs. Log lambda

lassoreg_1$lambda[90]  # lambda value with index 90
coef(lassoreg_1)[,90]  # retrieve one of Regression Coefficients (Regression Coefficients for lambda 0.1629751

k-fold cross validation approach to choose the best model (select the best lambda)
set.seed(1234)
lasso_cv <- cv.glmnet(x, y, alpha = 1, nfolds = 10, lambda = lambda_ridge_grid)

lasso_cv$cvm  # the mean cross-validated error (per every lambda)
lasso_cv$cvsd  # estimate of standard deviation error of cvm
lasso_cv$lambda.min  # value of lambda that gives the minimum cvm

lassoreg_2 <- glmnet(x, y, alpha = 1, lambda = lasso_cv$lambda.min)
coef(lassoreg_2)  # coefficients of regression with best lambda value

# Model 7: Decision Tree
tree_1 <- rpart(Log_Salary ~ Years + Hits + League, data = train, cp = 0.1, maxdepth = 3)
plot(tree_1)
rpart.plot::prp(tree_1)  # plot the tree
tree_1

tree_2 <- rpart(Log_Salary ~ Years + Hits + League, data = train, cp = 0.001, maxdepth = 3)
plot(tree_2)
rpart.plot::prp(tree_2)
tree_2

#which cp gives minimum Errors in Train dataset?
plotcp(tree_2)
tree_2$cptable
tree_2$cptable[which.min(tree_2$cptable[,'xerror'])]

#Prune the complex Tree
tree_3 <- prune.rpart(tree_2, cp = tree_2$cptable[which.min(tree_2$cptable[,'xerror'])])  # prune tree_2 base on cp being equal to 0.0173088
prp(tree_3)  # plot the pruned tree

tree_4 <- rpart(formula = Log_Salary ~ . - Salary, data = train, cp = 0.0001, maxdepth = 20)  # Decision Tree model using all variables
plot(tree_4)
prp(tree_4)
tree_4

#which cp gives minimum Errors in Train dataset?
plotcp(tree_4)
tree_4$cptable
tree_4$cptable[which.min(tree_4$cptable[,'xerror'])]

#Prune the complex Tree
tree_5 <- prune.rpart(tree_4, cp = tree_4$cptable[which.min(tree_4$cptable[,'xerror'])])  # prune tree_4 base on optimum cp
plot(tree_5)
prp(tree_5)
tree_5

# Model 8: Bagging
set.seed(1234)
bagging_1 <- randomForest(Log_Salary ~ . - Salary, mtry = ncol(train) - 2, ntree = 500, data = train)

bagging_1
#each Tree uses 18 variables
#makes 500 different Trees
#MSE: 0.193834
#R^2: 0.7657 (76.57% of variance has been explained by this model)

# Model 9: Random Forest
set.seed(1234)
rf_1 <- randomForest(Log_Salary ~ . - Salary, data = train, mtry = 6, ntree = 500, importance = T)

rf_1
#each Tree uses 6 variables
#500 different Trees have been made
#R^2: 0.7777 (77.77% of variance has been explained by this model)

importance(rf_1)
varImpPlot(rf_1)
#CAtBat has the most impact on increasing MSE if we remove it from the model -> the most important variable

#k-fold cross-validation for feature selection
set.seed(12345)
rf_cv <- rfcv(train[, -c(18, 20)], train$Log_Salary, cv.fold = 10, step = 0.75, mtry = function(p) max(1, floor(sqrt(p))), recursive = F)
class(rf_cv)
str(rf_cv)

rf_cv$n.var  # vector of number of variables of 'Variables Pool' at each step
rf_cv$error.cv  # vector of MSEs at each step by each 'Variables Pool'
which.min(rf_cv$error.cv)  # best 'Variables Pool' has 10 variables

#remove 8 less important variables base on 'relative importance-variables table'
sort(importance(rf_1)[,1], decreasing = T)
varImpPlot(rf_1)

reg_formula <- as.formula(Log_Salary ~ CAtBat + CHits + CRuns + Hits + Years + CRBI + Runs + CWalks + CHmRun + Walks)  # pool of 10 main important variables

floor(sqrt(10))  # use maximum 3 variable for every tree in RandomForest

#make the optimum model base on k-fold result
set.seed(1234)
rf_2 <- randomForest(reg_formula, data = train, mtry = floor(sqrt(10)), ntree = 500)
rf_2
# R^2: 0.7876 (78.76% of variance has been explained by this model)


### Step 5: Model Evaluation ----
# Test the Model 5 performance
#Prediction
#create model matrix for Test dataset
test$Log_Salary <- log(test$Salary)

x_test <- model.matrix(Log_Salary ~ . - Salary, data = test)[,-1]  # remove intercept

pred_ridgereg <- predict(ridgereg_2, s = ridge_cv$lambda.min, newx = x_test)  # prediction on test dataset
pred_ridgereg  # predictions of Log_Salary
pred_ridgereg <- exp(pred_ridgereg)
pred_ridgereg  # predictions of Salary

#Evaluate model performance in Test dataset:
#Actual vs. Prediction
plot(test$Salary, pred_ridgereg, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)  # compare with 45' line

#Absolute Error mean, median, sd, max, min
abs_err_ridgereg <- abs(pred_ridgereg - test$Salary) #absolute value (AEV)

hist(abs_err_ridgereg, breaks = 25)  # residuals distribution
mean(abs_err_ridgereg)
median(abs_err_ridgereg)
sd(abs_err_ridgereg)
max(abs_err_ridgereg)
min(abs_err_ridgereg)

#boxplot (which observations are outliers?)
boxplot(abs_err_ridgereg, main = 'Error distribution')

models_comp <- rbind(models_comp, "RidgeReg" = c(mean(abs_err_ridgereg),
                                                 median(abs_err_ridgereg),
                                                 sd(abs_err_ridgereg),
                                                 IQR(abs_err_ridgereg),
                                                 range(abs_err_ridgereg))) 

View(model_comp)

# Test the Model 6 performance
#Prediction
pred_lassoreg <- predict(lassoreg_2, s = lasso_cv$lambda.min, newx = x_test)  # prediction on test dataset
pred_lassoreg  # predictions of Log_Salary
pred_lassoreg <- exp(pred_lassoreg)
pred_lassoreg  # predictions of Salary

#Evaluate model performance in Test dataset:
#Actual vs. Prediction
plot(test$Salary, pred_lassoreg, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)  # compare with 45' line

#Absolute Error mean, median, sd, max, min
abs_err_lassoreg <- abs(pred_lassoreg - test$Salary) #absolute value (AEV)

hist(abs_err_lassoreg, breaks = 25)  # residuals distribution
mean(abs_err_lassoreg)
median(abs_err_lassoreg)
sd(abs_err_lassoreg)
max(abs_err_lassoreg)
min(abs_err_lassoreg)

#boxplot (which observations are outliers?)
boxplot(abs_err_lassoreg, main = 'Error distribution')

models_comp <- rbind(models_comp, "LassoReg" = c(mean(abs_err_lassoreg),
                                                 median(abs_err_lassoreg),
                                                 sd(abs_err_lassoreg),
                                                 IQR(abs_err_lassoreg),
                                                 range(abs_err_lassoreg))) 

View(model_comp)

# Test the Model 7 performance
#Prediction
pred_tree <- predict(tree_5, test)  # prediction on test dataset
pred_tree  # predictions of Log_Salary
pred_tree <- exp(pred_tree)
pred_tree  # predictions of Salary

#Evaluate model performance in Test dataset:
#Actual vs. Prediction
plot(test$Salary, pred_tree, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)  # compare with 45' line

#Absolute Error mean, median, sd, max, min
abs_err_tree <- abs(pred_tree - test$Salary) #absolute value (AEV)

hist(abs_err_tree, breaks = 25)  # residuals distribution
mean(abs_err_tree)
median(abs_err_tree)
sd(abs_err_tree)
max(abs_err_tree)
min(abs_err_tree)

#boxplot (which observations are outliers?)
boxplot(abs_err_tree, main = 'Error distribution')

models_comp <- rbind(models_comp, "Tree" = c(mean(abs_err_tree),
                                                 median(abs_err_tree),
                                                 sd(abs_err_tree),
                                                 IQR(abs_err_tree),
                                                 range(abs_err_tree))) 

View(model_comp)

# Test the Model 8 performance
#Prediction
pred_bagging <- predict(tree_5, test)  # prediction on test dataset
pred_bagging  # predictions of Log_Salary
pred_bagging <- exp(pred_bagging)
pred_bagging  # predictions of Salary

#Evaluate model performance in Test dataset:
#Actual vs. Prediction
plot(test$Salary, pred_bagging, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)  # compare with 45' line

#Absolute Error mean, median, sd, max, min
abs_err_bagging <- abs(pred_bagging - test$Salary) #absolute value (AEV)

hist(abs_err_bagging, breaks = 25)  # residuals distribution
mean(abs_err_bagging)
median(abs_err_bagging)
sd(abs_err_bagging)
max(abs_err_bagging)
min(abs_err_bagging)

#boxplot (which observations are outliers?)
boxplot(abs_err_bagging, main = 'Error distribution')

models_comp <- rbind(models_comp, "Bagging" = c(mean(abs_err_bagging),
                                                 median(abs_err_bagging),
                                                 sd(abs_err_bagging),
                                                 IQR(abs_err_bagging),
                                                 range(abs_err_bagging))) 

View(model_comp)

# Test the Model 9 performance
#Prediction
pred_rf <- predict(tree_5, test)  # prediction on test dataset
pred_rf  # predictions of Log_Salary
pred_rf <- exp(pred_rf)
pred_rf  # predictions of Salary

#Evaluate model performance in Test dataset:
#Actual vs. Prediction
plot(test$Salary, pred_rf, xlab = "Actual", ylab = "Prediction")
abline(a = 0, b = 1, col = "red", lwd = 2)  # compare with 45' line

#Absolute Error mean, median, sd, max, min
abs_err_rf <- abs(abs_err_rf - test$Salary) #absolute value (AEV)

hist(abs_err_rf, breaks = 25)  # residuals distribution
mean(abs_err_rf)
median(abs_err_rf)
sd(abs_err_rf)
max(abs_err_rf)
min(abs_err_rf)

#boxplot (which observations are outliers?)
boxplot(abs_err_rf, main = 'Error distribution')

models_comp <- rbind(models_comp, "RandomForest" = c(mean(abs_err_rf),
                                                 median(abs_err_rf),
                                                 sd(abs_err_rf),
                                                 IQR(abs_err_rf),
                                                 range(abs_err_rf))) 

View(model_comp)
