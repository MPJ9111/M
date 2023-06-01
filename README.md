# MPJ
# _Term Project_

## 260364 Applied Stat Model

### Team: Cookies and Friends

# Import Libraries and Declare Function

library(olsrr)
library(caret)
library(pROC)

hosmerlem <-function (y, yhat, g = 10)
{
     cutyhat <- cut(yhat, breaks = quantile(yhat,
     probs = seq(0,1, 1/g)), include.lowest = T)
     obs <- xtabs(cbind(1 - y, y) ~ cutyhat)
     expect <- xtabs(cbind(1 - yhat, yhat) ~ cutyhat)
     chisq <- sum((obs - expect)^2/expect)
     P <- 1 - pchisq(chisq, g - 2)
     c("X^2" = chisq, Df = g - 2, "P(>Chi)" = P)
}

# Data Preparation

### Import data

df <- read.csv("clean_data.csv", sep=",")
attach(df)

head(df)
dim(df)

names(df)

### Variable Abbreviations

col_dict <- list(
    PD = "PD",
    previous_credit_problem = "PCP",
    gender = "GEN",
    n_credits = "CRE",
    occupation = "OCC",
    n_dependents = "DEP",
    telephone_registered = "TEL",
    foreign_worker = "FRW",
    age = "AGE",
    current_account_with_money = "CAM",
    credit_amount = "CRA",
    duration_of_credit = "DUR",
    business.consumption = "BCO",
    saving_availability = "SAV",
    employed = "EMP",
    instalment_percent = "INS",
    guarantor_availability = "GUA",
    duration_in_current_address = "DCA",
    valuable_asset_availability = "VAA",
    concurrent_credit_existence = "CCE",
    free_housing = "FRE"
)

for (col in names(col_dict)) {
    names(df)[names(df) == col] <- col_dict[col]
}

names(df)

head(df)

print(paste0("Good Credit: ", sum(df$PD == 0)))
print(paste0("Bad Credit: ", sum(df$PD == 1)))

# Experimental Setting

### Influence Diagnostics

full_model <- glm(PD ~ ., data = df, family=binomial(link="logit"))
summary(full_model)

#### Outlying X Detection (leverage)

h <- hatvalues(full_model)
cutoff <- 2 * full_model$rank / nrow(df)
par(bg = "white")
plot(h, ylab = "Leverage", ylim=c(0,0.3), type="h")
abline(h=cutoff, col="blue", lty=2)
print(paste("the leverage cutoff is ", cutoff))

print("Outlier for X")
outlier_x_idx <- which(h > cutoff)
df[outlier_x_idx,]

#### Outlying Y Detection (External Deteled Residuals)

n <- nrow(df)
alpha <- 0.05
ext_res <- rstudent(full_model)
par(bg = "white")
plot(abs(ext_res), ylab="Absolute Studentized Deleted Residuals", ylim=c(0,4), type="h")
cutoff<-qt(1-alpha/(2*n), full_model$df.residual-1)
abline(h=cutoff, col="blue", lty=2)
print(paste("the cutoff is ", cutoff))

print("Outlier for Y")
outlier_y_idx <- which(abs(ext_res) > cutoff)
df[outlier_y_idx,]

# outlier indices
outlier_idx <- c(outlier_x_idx, outlier_y_idx)

#### Cook's Distance

influ_table <- data.frame(Cooks.D=cooks.distance(full_model)[outlier_idx])
influ_table$`Cooks.D > 4/n` <- influ_table$Cooks.D > 4/nrow(df)
influ_table

print(paste0("No of influential points: ", sum(influ_table$`Cooks.D > 4/n` == TRUE)))
ip_idx <- which(influ_table$`Cooks.D > 4/n` == TRUE)

#### Remove Influential Data

new_df = df[-c(ip_idx),]
dim(new_df)
print(paste0("Good Credit: ", sum(new_df$PD == 0)))
print(paste0("Bad Credit: ", sum(new_df$PD == 1)))

### Data Splitting (Stratified Train/Test Splitting)

set.seed(1)
train_index <- createDataPartition(new_df$PD, p = .8, list = FALSE)
train_df <- new_df[ train_index,]
test_df  <- new_df[-train_index,]
dim(train_df)
print(paste0("Good Credit: ", sum(train_df$PD == 0)))
print(paste0("Bad Credit: ", sum(train_df$PD == 1)))
dim(test_df)
print(paste0("Good Credit: ", sum(test_df$PD == 0)))
print(paste0("Bad Credit: ", sum(test_df$PD == 1)))

# Experiment

## 0. Fit full model by training set

full_model <- glm(PD ~ ., data = train_df, family=binomial(link="logit"))
summary(full_model)

hosmerlem(train_df$PD, full_model$fitted.values)

pred_model = glm(
    PD ~ .,
    family = binomial(link = "logit"),
    data = train_df
)

pred_prob <- predict(pred_model, test_df, type = "response")
pred_class <- rep(0, nrow(test_df))
# Use 0.5 as cutoff point
pred_class[pred_prob>=0.5]<-1
truth_class <- test_df$PD
xtab <- table(pred_class, truth_class)
xtab

print(paste0("Sensitivity: ", sensitivity(xtab)))

print(paste0("Specificity: ", specificity(xtab)))

par(bg = "white")
roc(truth_class, pred_prob, plot = TRUE)

roc_obj <- roc(truth_class, pred_prob)
cc <- coords(roc_obj, "best", best.method="youden")
youden_index = cc$threshold
youden_index

pred_class <- rep(0, nrow(test_df))
pred_class[pred_prob>=youden_index]<-1
truth <- test_df$PD
xtab <- table(pred_class, truth)
xtab

confusionMatrix(xtab)[4]$byClass[['F1']]

sensitivity(xtab)
specificity(xtab)
confusionMatrix(xtab)[3]$overall[1][['Accuracy']]

full_result = data.frame(
    Approach = c("Full"),
    # Accuracy = confusionMatrix(xtab)[3]$overall[1][['Accuracy']],
    Bal_Accuracy = confusionMatrix(xtab)[4]$byClass[['Balanced Accuracy']],
    Sensitivity = c(sensitivity(xtab)),
    Specificity = c(specificity(xtab)),
    F1 = confusionMatrix(xtab)[4]$byClass[['F1']],
    AUC = c(auc(truth_class, pred_prob))
)
full_result

## 1. Backward Stepwise Selection (AIC)

### Model Selection

best_model <- step(full_model)
summary(best_model)

### Model Diagnostics

#### Goodness of Fit (Hosmer-Lemeshow Test)

hosmerlem(train_df$PD, full_model$fitted.values)

hosmerlem(train_df$PD, best_model$fitted.values)

#### Drop in deviance

pchisq(deviance(best_model)-deviance(full_model), df = 13, lower=F)

#### Overdispersion (Quasi-likelihood)

# quasi-likelihood
est.phi <- function(glmobj) {
    sum(residuals(glmobj, type="pearson")^2)/df.residual(glmobj)
}
est.phi(best_model)

### Prediction and Evaluation

pred_model = glm(
    PD ~ PCP + TEL + FRW + CAM + CRA + DUR + SAV + 
    INS + VAA + CCE + FRE,
    family = binomial(link = "logit"),
    data = train_df
)
summary(pred_model)

pred_prob <- predict(pred_model, test_df, type = "response")
pred_class <- rep(0, nrow(test_df))
# Use 0.5 as cutoff point
pred_class[pred_prob>=0.5]<-1
truth_class <- test_df$PD
xtab <- table(pred_class, truth_class)
xtab

print(paste0("Sensitivity: ", sensitivity(xtab)))

print(paste0("Specificity: ", specificity(xtab)))

par(bg = "white")
roc(truth_class, pred_prob, plot = TRUE)

roc_obj <- roc(truth_class, pred_prob)
cc <- coords(roc_obj, "best", best.method="youden")
youden_index = cc$threshold
youden_index

pred_class <- rep(0, nrow(test_df))
pred_class[pred_prob>=youden_index]<-1
truth <- test_df$PD
xtab <- table(pred_class, truth)
xtab

sensitivity(xtab)
specificity(xtab)
confusionMatrix(xtab)[3]$overall[1][['Accuracy']]

aic_result = data.frame(
    Approach = c("AIC"),
    # Accuracy = confusionMatrix(xtab)[3]$overall[1][['Accuracy']],
    Bal_Accuracy = confusionMatrix(xtab)[4]$byClass[['Balanced Accuracy']],
    Sensitivity = c(sensitivity(xtab)),
    Specificity = c(specificity(xtab)),
    F1 = confusionMatrix(xtab)[4]$byClass[['F1']],
    AUC = c(auc(truth_class, pred_prob))
)
aic_result

## 2. Backward Elimination (Wald's Test with p-value cutoff = 0.05)

### Model Selection

reduced_model <- update(full_model, PD ~ . - GEN, data = train_df)
summary(reduced_model)

reduced_model <- update(reduced_model, PD ~ . - BCO, data = train_df)
summary(reduced_model)

reduced_model <- update(reduced_model, PD ~ . - CRE, data = train_df)
summary(reduced_model)

reduced_model <- update(reduced_model, PD ~ . - OCC, data = train_df)
summary(reduced_model)

reduced_model <- update(reduced_model, PD ~ . - GUA, data = train_df)
summary(reduced_model)

reduced_model <- update(reduced_model, PD ~ . - EMP, data = train_df)
summary(reduced_model)

reduced_model <- update(reduced_model, PD ~ . - DCA, data = train_df)
summary(reduced_model)

reduced_model <- update(reduced_model, PD ~ . - DEP, data = train_df)
summary(reduced_model)

reduced_model <- update(reduced_model, PD ~ . - AGE, data = train_df)
summary(reduced_model)

reduced_model <- update(reduced_model, PD ~ . - VAA, data = train_df)
summary(reduced_model)

best_model <- update(reduced_model, PD ~ . - FRW, data = train_df)
summary(best_model)

### Model Diagnostics

#### Goodness of Fit (Hosmer-Lemeshow Test)

hosmerlem <-function (y, yhat, g = 10)
{
     cutyhat <- cut(yhat, breaks = quantile(yhat,
     probs = seq(0,1, 1/g)), include.lowest = T)
     obs <- xtabs(cbind(1 - y, y) ~ cutyhat)
     expect <- xtabs(cbind(1 - yhat, yhat) ~ cutyhat)
     chisq <- sum((obs - expect)^2/expect)
     P <- 1 - pchisq(chisq, g - 2)
     c("X^2" = chisq, Df = g - 2, "P(>Chi)" = P)
}

hosmerlem(train_df$PD, full_model$fitted.values)

hosmerlem(train_df$PD, best_model$fitted.values)

#### Drop in deviance

pchisq(deviance(best_model)-deviance(full_model), df = 15, lower=F)

#### Overdispersion (Quasi-likelihood)

# quasi-likelihood
est.phi <- function(glmobj) {
    sum(residuals(glmobj, type="pearson")^2)/df.residual(glmobj)
}
est.phi(best_model)

### Prediction and Evaluation

pred_model = glm(
    PD ~ PCP + TEL + CAM + CRA + DUR + SAV + INS + CCE + FRE,
    family = binomial(link = "logit"),
    data = train_df
)
summary(pred_model)

pred_prob <- predict(pred_model, test_df, type = "response")
pred_class <- rep(0, nrow(test_df))
# Use 0.5 as cutoff point
pred_class[pred_prob>=0.5]<-1
truth_class <- test_df$PD
xtab <- table(pred_class, truth_class)
xtab

print(paste0("Sensitivity: ", sensitivity(xtab)))

print(paste0("Specificity: ", specificity(xtab)))

par(bg = "white")
roc(truth_class, pred_prob, plot = TRUE)

roc_obj <- roc(truth_class, pred_prob)
cc <- coords(roc_obj, "best", best.method="youden")
youden_index = cc$threshold
youden_index

pred_class <- rep(0, nrow(test_df))
pred_class[pred_prob>=youden_index]<-1
truth <- test_df$PD
xtab <- table(pred_class, truth)
xtab

sensitivity(xtab)
specificity(xtab)
confusionMatrix(xtab)[3]$overall[1][['Accuracy']]

wal_result = data.frame(
    Approach = c("WAL"),
    # Accuracy = confusionMatrix(xtab)[3]$overall[1][['Accuracy']],
    Bal_Accuracy = confusionMatrix(xtab)[4]$byClass[['Balanced Accuracy']],
    Sensitivity = c(sensitivity(xtab)),
    Specificity = c(specificity(xtab)),
    F1 = confusionMatrix(xtab)[4]$byClass[['F1']],
    AUC = c(auc(truth_class, pred_prob))
)
wal_result

# Experimental Result

exp_result = rbind(full_result, aic_result, wal_result)
exp_result

Wald's test wins AIC.

# Final Model (Best Model by All data)

full_model = glm(
    PD ~ .,
    family = binomial(link = "logit"),
    data = new_df
)

final_model = glm(
    PD ~ PCP + TEL + CAM + CRA + DUR + SAV + INS + CCE + FRE,
    family = binomial(link = "logit"),
    data = new_df
)
summary(final_model)

### Diagnostic

hosmerlem(new_df$PD, final_model$fitted.values)

pchisq(deviance(final_model)-deviance(full_model), df = 15, lower=F)

# quasi-likelihood
est.phi <- function(glmobj) {
    sum(residuals(glmobj, type="pearson")^2)/df.residual(glmobj)
}
est.phi(final_model)

par(bg="white")
plot(full_model, 1)
plot(full_model, 2)
