Lab 9
================

### Econ B2000, MA Econometrics

### Kevin R Foster, the Colin Powell School at the City College of New York, CUNY

### Fall 2023

This is part 3 B (the fifth part) of a 3-part segment.

In the last 3 weeks we estimated OLS and logit models, created a
standardized dataset and split it into training and test sets, then
random forest and support vector machines. This week we play with some
more estimation techniques and see how they perform.

Here is Elastic Net. It combines LASSO with Ridge and the alpha
parameter (from 0 to 1) determines the relative weight. Begin with alpha
= 1 so just LASSO.

``` r
# Elastic Net
require(glmnet)
model1_elasticnet <-  glmnet(as.matrix(sobj$data[,-1]),sobj$data$pub_work) 
# default is alpha = 1, lasso

print(model1_elasticnet)

cvmodel1_elasticnet = cv.glmnet(data.matrix(sobj$data[,-1]),data.matrix(sobj$data$pub_work)) 
cvmodel1_elasticnet$lambda.min
log(cvmodel1_elasticnet$lambda.min)
coef(cvmodel1_elasticnet, s = "lambda.min")

pred1_elasnet <- predict(model1_elasticnet, newx = data.matrix(s_dat_test), s = cvmodel1_elasticnet$lambda.min)
pred_model1_elasnet <- (pred1_elasnet < mean(pred1_elasnet)) 
table(pred = pred_model1_elasnet, true = dat_test$pub_work)

model2_elasticnet <-  glmnet(as.matrix(sobj$data[,-1]),sobj$data$pub_work, alpha = 0) 
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
