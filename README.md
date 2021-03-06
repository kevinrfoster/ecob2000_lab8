Lab 8
================

### Econ B2000, MA Econometrics

### Kevin R Foster, the Colin Powell School at the City College of New York, CUNY

### Fall 2021

This is the last lab (wipe away tear) and we’ll continue from previous
setup.

Last time we estimated OLS and logit on question of whether people got
vaxx. Now we use that complicated setup (creating a particular object of
the training data) in order that we can use it consistently through all
the other estimations. We want to compare all of them on a level field.

A word of caution: just because a particular technique works better or
worse on this particular dataset does *not* mean that it’s always better
or worse. Just in this particular case for these particular data. Since
some of these were developed for larger datasets, it might not be
surprising that they don’t work as well on smaller versions.

One thing that we keep coming back to, in this class, is that there is
both art and science to data analytics. The methods look so mathy and
technical and have imposing names but there is quite a lot of personal
artistry in how to use them. There is good and bad to the artistry (and
also generally accepted vs unusual) but I want you to remember that
fancy estimations don’t guarantee correct results.

In previous lab, you’d set a subsample (perhaps everybody over 12,
perhaps particular groups) and figured a set of X variables that are
plausibly causal. You made choices about how to deal with NA values. You
created this thing, `sobj <- standardize(vaxx ~ X1 + X2 ...)` where you
had choices for X variables and filled in the `...` part. You estimated
OLS and logit and created confusion matrix for each, and checked
predicted values overall and for subgroups.

Now let’s estimate some fancy models.

Here is code for a Random Forest, which takes a bit of computing,

``` r
require('randomForest')
set.seed(54321)
model_randFor <- randomForest(as.factor(vaxx) ~ ., data = sobj$data, importance=TRUE, proximity=TRUE)
print(model_randFor)
round(importance(model_randFor),2)
varImpPlot(model_randFor)
# look at confusion matrix for this too
pred_model1 <- predict(model_randFor,  s_dat_test)
table(pred = pred_model1, true = dat_test$vaxx)
```

Note that the estimation prints out a Confusion Matrix first but that’s
within the training data; the later one calculates how well it does on
the test data.

Next is Support Vector Machines. First it tries to find optimal tuning
parameter, next uses those optimal values to train. (Tuning takes a long
time so skip for now\!)

``` r
require(e1071)
# tuned_parameters <- tune.svm(as.factor(vaxx) ~ ., data = sobj$data, gamma = 10^(-3:0), cost = 10^(-2:2)) 
# summary(tuned_parameters)
# figure best parameters and input into next
svm.model <- svm(as.factor(vaxx) ~ ., data = sobj$data, cost = 1, gamma = 0.1)
svm.pred <- predict(svm.model, s_dat_test)
table(pred = svm.pred, true = dat_test$vaxx)
```

Here is Elastic Net. It combines LASSO with Ridge and the alpha
parameter (from 0 to 1) determines the relative weight. Begin with alpha
= 1 so just LASSO.

``` r
# Elastic Net
require(glmnet)
model1_elasticnet <-  glmnet(as.matrix(sobj$data[,-1]),sobj$data$vaxx) 
# default is alpha = 1, lasso

print(model1_elasticnet)

cvmodel1_elasticnet = cv.glmnet(data.matrix(sobj$data[,-1]),data.matrix(sobj$data$vaxx)) 
cvmodel1_elasticnet$lambda.min
log(cvmodel1_elasticnet$lambda.min)
coef(cvmodel1_elasticnet, s = "lambda.min")

pred1_elasnet <- predict(model1_elasticnet, newx = data.matrix(s_dat_test), s = cvmodel1_elasticnet$lambda.min)
pred_model1_elasnet <- (pred1_elasnet < mean(pred1_elasnet)) 
table(pred = pred_model1_elasnet, true = dat_test$vaxx)

model2_elasticnet <-  glmnet(as.matrix(sobj$data[,-1]),sobj$data$vaxx, alpha = 0) 
# or try different alpha values to see if you can improve
print(model2_elasticnet)
```

Here is Spike and Slab.

``` r
require(spikeslab)
set.seed(54321)
model1_spikeslab <- spikeslab(sobj$formula, data = sobj$data)
summary(model1_spikeslab)
print(model1_spikeslab)
plot(model1_spikeslab)
```

When you summarize, you should be able to explain which models predict
best (noting if there is a tradeoff of false positive vs false negative)
and if there are certain explanatory variables that are consistently
more or less useful. Also try other lists of explanatory variables.

Explain carefully about what is the marginal product of each of these
methods. Old-fashioned OLS and logit give some predictions – are these
other methods better overall or in particular cases? Do they tend to
make the same sort of errors or different ones? (If different then
perhaps you can create an ensemble of models?)
