library(ISLR2)

# Load the Weekly dataset
data("Weekly")

# (a) Numerical and graphical summaries
summary(Weekly)
par(mfrow=c(2,2))
plot(Weekly$Year, Weekly$Lag1, xlab="Year", ylab="Lag1")
plot(Weekly$Year, Weekly$Lag2, xlab="Year", ylab="Lag2")
plot(Weekly$Year, Weekly$Lag3, xlab="Year", ylab="Lag3")
plot(Weekly$Year, Weekly$Lag4, xlab="Year", ylab="Lag4")
plot(Weekly$Year, Weekly$Lag5, xlab="Year", ylab="Lag5")
plot(Weekly$Year, Weekly$Volume, xlab="Year", ylab="Volume")

# (b) Logistic regression with full data
fit_full <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, data=Weekly, family=binomial)
summary(fit_full)

# (c) Confusion matrix and overall fraction of correct predictions
predict_full <- ifelse(predict(fit_full, type="response") > 0.5, "Up", "Down")
conf_mat_full <- table(predict_full, Weekly$Direction)
conf_mat_full
accuracy_full <- sum(diag(conf_mat_full)) / sum(conf_mat_full)
accuracy_full

# (d) Logistic regression with training data (1990-2008) and Lag2 as the predictor
train_data <- Weekly[Weekly$Year <= 2008,]
test_data <- Weekly[Weekly$Year > 2008,]
fit_train <- glm(Direction ~ Lag2, data=train_data, family=binomial)
summary(fit_train)
predict_train <- ifelse(predict(fit_train, newdata=test_data, type="response") > 0.5, "Up", "Down")
conf_mat_train <- table(predict_train, test_data$Direction)
conf_mat_train
accuracy_train <- sum(diag(conf_mat_train)) / sum(conf_mat_train)
accuracy_train

# (e) KNN with K=1
library(class)
knn_pred <- knn(train = as.matrix(train_data["Lag2"]), test = as.matrix(test_data["Lag2"]), cl = train_data$Direction, k = 1)
conf_mat_knn <- table(knn_pred, test_data$Direction)
accuracy_knn <- sum(diag(conf_mat_knn)) / sum(conf_mat_knn)
conf_mat_knn
accuracy_knn

# (f) Naive Bayes
library(e1071)
nb_fit <- naiveBayes(Direction ~ Lag2, data = train_data)
nb_pred <- predict(nb_fit, newdata = test_data)
conf_mat_nb <- table(nb_pred, test_data$Direction)
accuracy_nb <- sum(diag(conf_mat_nb)) / sum(conf_mat_nb)
conf_mat_nb
accuracy_nb

# Create a results dataframe
results <- data.frame(Method=c("Logistic Regression (Full Data)", "Logistic Regression (1990-2008)", "KNN (K=1)", "Naive Bayes"),
                      Accuracy=c(accuracy_full, accuracy_train, accuracy_knn, accuracy_nb))
results
