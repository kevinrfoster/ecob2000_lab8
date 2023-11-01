Lab 8
================

### Econ B2000, MA Econometrics

### Kevin R Foster, the Colin Powell School at the City College of New York, CUNY

### Fall 2023

This is part 3 A of a 3-part segment. Part 3 B will be the fifth part of
a 3-part segment, because econometricians count good.

In the last 2 weeks we estimated OLS and logit models. Last week’s
addendum included basics on how to create a standardized dataset split
into training and test. We’ll move to estimate other models in a variety
of specifications.

Start with a review. Some of these machine learning techniques are very
computationally intensive so we might want to split up the problems. You
should convince yourselves that these estimations provide identical
coefficient estimates (once properly interpreted):

``` r
# for now, really simplify the education dummy
dat_use$BA_plus <- dat_use$educ_college + dat_use$educ_advdeg

# whole dataset
model_lpm_v1 <- lm(public_work_num ~ female + BA_plus + AGE + I(female*BA_plus) + I(AGE * female), data = dat_use)
summary(model_lpm_v1)

dat_use_female <- subset(dat_use,as.logical(dat_use$female))
dat_use_male <- subset(dat_use,!(dat_use$female))

# now split into 2 parts
model_lpm_v1f <- lm(public_work_num ~ BA_plus + AGE, data = dat_use_female)
summary(model_lpm_v1f)
model_lpm_v1m <- lm(public_work_num ~ BA_plus + AGE, data = dat_use_male)
summary(model_lpm_v1m)
```

(Actually that’s kinda nice exam question!) Please convince yourself
(and explain) that the model predictions for different people are
identical in the big equation with full interactions or the subsets. The
big model with full interactions is tougher to interpret although it
does provide easy access to hypothesis tests about the split (are the
coefficients on age statistically different, for males or females?).

The point is that sometimes you might want to split the data into even
smaller subsets, in order to run the model without crashing or taking
hours. Your own computer has its limitations and you need to learn how
to work around those. It’s useful for future work – many problems have
vast amounts of data that need to be trimmed down. (There’s a whole
segment of data scientists who don’t worry about their data size since
they just splash it all onto AWS and then complain about how much
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

Now let’s estimate some fancier models.

Here is code for a Random Forest, which takes a bit of computing
(especially if you include 144 PUMA dummies!),

``` r
require('randomForest')
set.seed(54321)
model_randFor <- randomForest(as.factor(pub_work) ~ ., data = sobj$data, importance=TRUE, proximity=TRUE)
print(model_randFor)
round(importance(model_randFor),2)
varImpPlot(model_randFor)
# look at confusion matrix for this too
pred_model1 <- predict(model_randFor,  s_dat_test)
table(pred = pred_model1, true = dat_test$pub_work)
```

Note that the estimation prints out a Confusion Matrix first but that’s
within the training data; the later one calculates how well it does on
the test data.

Next is Support Vector Machines. First it tries to find optimal tuning
parameter, next uses those optimal values to train. (Tuning takes a long
time so skip for now!)

``` r
require(e1071)
# tuned_parameters <- tune.svm(as.factor(pub_work) ~ ., data = sobj$data, gamma = 10^(-3:0), cost = 10^(-2:2)) 
# summary(tuned_parameters)
# figure best parameters and input into next
svm.model <- svm(as.factor(pub_work) ~ ., data = sobj$data, cost = 1, gamma = 0.1)
svm.pred <- predict(svm.model, s_dat_test)
table(pred = svm.pred, true = dat_test$pub_work)
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
