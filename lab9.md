Lab 9
================

### Econ B2000, MA Econometrics

### Kevin R Foster, the Colin Powell School at the City College of New York, CUNY

### Fall 2024

In the last 3 weeks we estimated OLS and logit models, created a
standardized dataset and split it into training and test sets. Now I
will show you a variety of machine learning techniques, which you can
choose from.

These are: random forest support vector machines elastic net (LASSO +
Ridge) spike and slab

But first! First we have to do some thinking – which is usually the
hardest part of any task.

Some of these machine learning techniques are very computationally
intensive so we might want to split up the problems. You should convince
yourselves that these estimations provide identical coefficient
estimates (once properly interpreted):

``` r
# for now, really simplify the education dummy
trad_data$BA_plus <- trad_data$educ_college + trad_data$educ_advdeg
# (and check your understanding, will this ever equal 2?)

# whole dataset
model_lpm_v1 <- lm(he_more_than_5yrs_than_her ~ BA_plus + AGE + + I(AGE*BA_plus), data = trad_data)
summary(model_lpm_v1)
# ignoring h_educ for now -- you can add

dat_use_BAplus <- trad_data %>% filter(as.logical(BA_plus)) 
dat_use_ltBA <- trad_data %>% filter(!as.logical(BA_plus)) 

# now split into 2 parts
model_lpm_v1_BAplus <- lm(he_more_than_5yrs_than_her ~ AGE, data = dat_use_BAplus)
summary(model_lpm_v1_BAplus)
model_lpm_v1_ltBA <- lm(he_more_than_5yrs_than_her ~ AGE, data = dat_use_ltBA)
summary(model_lpm_v1_ltBA)
```

(Actually that’s kinda nice exam question!) Please convince yourself
(and explain) that the model predictions for different people are
identical in the big equation with full interactions or the subsets. The
big model with full interactions is tougher to interpret although it
does provide easy access to hypothesis tests about the split (are the
coefficients on age statistically different, for education levels?).
Obviously with so few variables this was not necessary but with more
complicated models it might become necessary.

The point is that sometimes you might want to split the data into even
smaller subsets, in order to run the model without crashing or taking
hours. Your own computer has its limitations and you need to learn how
to work around those. It’s useful for future work – many problems have
vast amounts of data that need to be trimmed down. (Although there’s a
whole segment of data scientists who don’t worry about their data size
since they just splash it all onto AWS and then complain about how much
they’re paying AWS.)

A word of caution: just because a particular technique works better or
worse on this particular dataset does *not* mean that it’s always better
or worse. Just in this particular case for these particular data.

One thing that we keep coming back to, in this class, is that there is
both art and science to data analytics. The methods look so mathy and
technical and have imposing names but there is quite a lot of personal
artistry in how to use them. There is good and bad to the artistry (and
also generally accepted vs unusual) but I want you to remember that
fancy estimations don’t guarantee correct results.

Sometimes it can be useful to try these techniques in sequence. Some are
good at wringing out every drop of juice for prediction; others are good
at selecting which variables don’t give enough juice to be worth the
squeeze. You might find some of your X-variables that are consistently
not selected as useful for prediction.

In previous lab, you’d set a subsample and figured a set of X variables
that are plausibly causal. You made choices about how to deal with NA
values. You created this thing, `sobj <- standardize(y ~ X1 + X2 ...)`
where you had choices for X variables and filled in the `...` part. You
estimated OLS and logit and created confusion matrix for each, and
checked predicted values overall and for subgroups.

### Random Forest

Here is code for a Random Forest, which takes a lot of compute
(especially if you include state dummies!),

``` r
require('randomForest')
set.seed(54321)
model_randFor <- randomForest(as.factor(he_more_than_5yrs_than_her) ~ ., data = sobj$data, importance=TRUE, proximity=TRUE)
print(model_randFor)
round(importance(model_randFor),2)
varImpPlot(model_randFor)
# look at confusion matrix for this too
pred_model1 <- predict(model_randFor,  s_dat_test)
table(pred = pred_model1, true = dat_test$he_more_than_5yrs_than_her)
```

Note that the estimation prints out a Confusion Matrix first but that’s
within the training data; the later one calculates how well it does on
the test data.

### Support Vector Machines

Next is Support Vector Machines. First it tries to find optimal tuning
parameter, next uses those optimal values to train. (Tuning takes a long
time so skip for now!)

``` r
require(e1071)
# tuned_parameters <- tune.svm(as.factor(pub_work) ~ ., data = sobj$data, gamma = 10^(-3:0), cost = 10^(-2:2)) 
# summary(tuned_parameters)
# figure best parameters and input into next
svm.model <- svm(as.factor(he_more_than_5yrs_than_her) ~ ., data = sobj$data, cost = 1, gamma = 0.1)
svm.pred <- predict(svm.model, s_dat_test)
table(pred = svm.pred, true = dat_test$he_more_than_5yrs_than_her)
```

### Elastic Net - LASSO + Ridge

Here is Elastic Net. It combines LASSO with Ridge and the alpha
parameter (from 0 to 1) determines the relative weight. Begin with alpha
= 1 so just LASSO.

``` r
# Elastic Net
require(glmnet)
model1_elasticnet <-  glmnet(as.matrix(sobj$data[,-1]),sobj$data$he_more_than_5yrs_than_her) 
# default is alpha = 1, lasso

print(model1_elasticnet)
plot(model1_elasticnet, label = TRUE)

cvmodel1_elasticnet = cv.glmnet(data.matrix(sobj$data[,-1]),data.matrix(sobj$data$he_more_than_5yrs_than_her)) 
cvmodel1_elasticnet$lambda.min
log(cvmodel1_elasticnet$lambda.min)
coef(cvmodel1_elasticnet, s = "lambda.min")

pred1_elasnet <- predict(model1_elasticnet, newx = data.matrix(s_dat_test), s = cvmodel1_elasticnet$lambda.min)
pred_model1_elasnet <- (pred1_elasnet < mean(pred1_elasnet)) 
table(pred = pred_model1_elasnet, true = dat_test$he_more_than_5yrs_than_her)

model2_elasticnet <-  glmnet(as.matrix(sobj$data[,-1]),sobj$data$pub_work, alpha = 0) 
# or try different alpha values to see if you can improve
print(model2_elasticnet)
```

### Spike and Slab

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
