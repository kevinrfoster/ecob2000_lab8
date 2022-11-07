Lab 8
================

### Econ B2000, MA Econometrics

### Kevin R Foster, the Colin Powell School at the City College of New York, CUNY

### Fall 2022

This is the last lab (wipe away tear) and we’ll continue from previous
setup.

Last time we estimated OLS and logit on question of whether people spent
time on a particular activity (choose your own! I used sports time). Now
we use that complicated setup (creating a particular object of the
training data) in order that we can use it consistently through all the
other estimations. We want to compare all of them on a level field.

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

In previous lab, you’d set a subsample and figured a set of X variables
that are plausibly causal. You made choices about how to deal with NA
values. You created this thing, `sobj <- standardize(y ~ X1 + X2 ...)`
where you had choices for X variables and filled in the `...` part. You
estimated OLS and logit and created confusion matrix for each, and
checked predicted values overall and for subgroups.

Now let’s estimate some fancy models.

Here is code for a Random Forest, which takes a bit of computing,

``` r
require('randomForest')
set.seed(54321)
model_randFor <- randomForest(as.factor(any_time_sports) ~ ., data = sobj$data, importance=TRUE, proximity=TRUE)
print(model_randFor)
round(importance(model_randFor),2)
varImpPlot(model_randFor)
# look at confusion matrix for this too
pred_model1 <- predict(model_randFor,  s_dat_test)
table(pred = pred_model1, true = dat_test$any_time_sports)
```

Note that the estimation prints out a Confusion Matrix first but that’s
within the training data; the later one calculates how well it does on
the test data.

Next is Support Vector Machines. First it tries to find optimal tuning
parameter, next uses those optimal values to train. (Tuning takes a long
time so skip for now!)

``` r
require(e1071)
# tuned_parameters <- tune.svm(as.factor(any_time_sports) ~ ., data = sobj$data, gamma = 10^(-3:0), cost = 10^(-2:2)) 
# summary(tuned_parameters)
# figure best parameters and input into next
svm.model <- svm(as.factor(any_time_sports) ~ ., data = sobj$data, cost = 1, gamma = 0.1)
svm.pred <- predict(svm.model, s_dat_test)
table(pred = svm.pred, true = dat_test$any_time_sports)
```

Here is Elastic Net. It combines LASSO with Ridge and the alpha
parameter (from 0 to 1) determines the relative weight. Begin with alpha
= 1 so just LASSO.

``` r
# Elastic Net
require(glmnet)
model1_elasticnet <-  glmnet(as.matrix(sobj$data[,-1]),sobj$data$any_time_sports) 
# default is alpha = 1, lasso

print(model1_elasticnet)

cvmodel1_elasticnet = cv.glmnet(data.matrix(sobj$data[,-1]),data.matrix(sobj$data$any_time_sports)) 
cvmodel1_elasticnet$lambda.min
log(cvmodel1_elasticnet$lambda.min)
coef(cvmodel1_elasticnet, s = "lambda.min")

pred1_elasnet <- predict(model1_elasticnet, newx = data.matrix(s_dat_test), s = cvmodel1_elasticnet$lambda.min)
pred_model1_elasnet <- (pred1_elasnet < mean(pred1_elasnet)) 
table(pred = pred_model1_elasnet, true = dat_test$any_time_sports)

model2_elasticnet <-  glmnet(as.matrix(sobj$data[,-1]),sobj$data$any_time_sports, alpha = 0) 
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

## Details

In last lab, I encouraged you to make some of your own choices about how
to put together a y-variable and some x-variables, including some
recoding of factors. Here are my own choices – not necessarily the best
but they work.

Part of the reason for recoding is that you want to check that your
training data (which, here, is only 10% of your overall data) has
sufficient contrasts in all of the x-variables. That means that if some
dummy has min = max = zero (or any other value) then you’ve got no
contrast. You cannot answer a question of “how are group A different
from others?” if either all or none of training data are people in group
A. There’s a math reason for that (determinant of matrix is zero) but
also a logical reason. Researchers sometimes forget that – if you want
to know about, say, what causes bankruptcy then you cannot just look at
companies that went bankrupt. You have to also look at ones that didn’t,
otherwise there’s no contrast.

``` r
require(tidyverse)

# load ATUS_18192021.RData data within your system

any_time_sports <- (dat_ATUS$ACT_SPORTS > 0)

dat_ATUS$EDUC_r <- recode_factor(dat_ATUS$EDUC, "\"Less than 1st grade\"" = "ltHS", "\"1st, 2nd, 3rd, or 4th grade\"" = "ltHS", "\"5th or 6th grade\""  = "ltHS",
                                 "\"7th or 8th grade\"" = "ltHS", "\"9th grade\"" = "ltHS", "\"10th grade\"" = "ltHS", "\"11th grade\"" = "ltHS", 
                                 "\"12th grade - no diploma\"" = "ltHS",
                                 "\"High school graduate - GED\"" = "HS", "\"High school graduate - diploma\"" = "HS", 
                                 "\"Some college but no degree\"" = "some_college",
                                 "\"Associate degree - occupational vocational\"" = "associate", "\"Associate degree - academic program\"" = "associate",
                                 "\"Bachelor's degree (BA, AB, BS, etc.)\"" = "bachelor", "\"Master's degree (MA, MS, MEng, MEd, MSW, etc.)\"" = "master",
                                 "\"Professional school degree (MD, DDS, DVM, etc.)\"" = "prof_or_PhD", "\"Doctoral degree (PhD, EdD, etc.)\"" = "prof_or_PhD",
                                 .default = "D")

dat_ATUS$RACE_r <- recode_factor(dat_ATUS$RACE, "\"White only\"" = "white", "\"Black only\"" = "Black", "\"American Indian, Alaskan Native\"" = "Native",
                                 "\"Asian only\"" = "Asian", 
                                 "\"Hawaiian Pacific Islander only\"" = "Native",
                                 .default = "other")

dat_ATUS$HISPAN_r <- recode_factor(dat_ATUS$HISPAN, "\"Not Hispanic\"" = "not", .default = "Hispanic")

d_educ_r <- data.frame(model.matrix(~ dat_ATUS$EDUC_r))
d_race_r <- data.frame(model.matrix(~ dat_ATUS$RACE_r))
d_marstat <- data.frame(model.matrix(~ dat_ATUS$MARST))
d_hispanic_r <- data.frame(model.matrix(~ dat_ATUS$HISPAN_r))
d_sex <- data.frame(model.matrix(~ dat_ATUS$SEX))
d_region <- data.frame(model.matrix(~ dat_ATUS$REGION))

d_any_time_sports <- data.frame(model.matrix(~ any_time_sports)) # or whatever time use you choose 


dat_for_analysis_sub <- data.frame(
  d_any_time_sports[ !is.na(dat_ATUS$HISPAN) ,2],
  dat_ATUS$AGE[!is.na(dat_ATUS$HISPAN)],
  d_educ_r[!is.na(dat_ATUS$HISPAN),2:7],
  d_marstat[!is.na(dat_ATUS$HISPAN),2:6],
  d_race_r[!is.na(dat_ATUS$HISPAN),2:5],
  d_hispanic_r[,2],
  d_sex[!is.na(dat_ATUS$HISPAN),2],
  d_region[!is.na(dat_ATUS$HISPAN),2:4]) # need [] since model.matrix includes intercept term


# this is just about me being anal-retentive, see difference in names(dat_for_analysis_sub) before and after running this bit
names(dat_for_analysis_sub)
names(dat_for_analysis_sub) <- sub("dat_ATUS.","",names(dat_for_analysis_sub))
names(dat_for_analysis_sub) <- sub("_r",".",names(dat_for_analysis_sub))
names(dat_for_analysis_sub)[1] <- "any_time_sports"
names(dat_for_analysis_sub)[2] <- "AGE"
names(dat_for_analysis_sub)[18] <- "Hispanic"
names(dat_for_analysis_sub)[19] <- "SEX"
names(dat_for_analysis_sub)

require("standardize")
set.seed(654321)
NN <- length(dat_for_analysis_sub$any_time_sports)
restrict_1 <- (runif(NN) < 0.1) # use 10% as training data
summary(restrict_1)
dat_train <- subset(dat_for_analysis_sub, restrict_1)
dat_test <- subset(dat_for_analysis_sub, !restrict_1)

# check that none of the means are zero or entirely missing
summary(dat_train)

sobj <- standardize(any_time_sports ~ AGE + EDUC.HS + EDUC.some_college + EDUC.associate + EDUC.bachelor + EDUC.master + EDUC.prof_or_PhD + 
                      MARST.Married...spouse.absent.+ MARST.Widowed. + MARST.Divorced. + MARST.Separated. + MARST.Never.married. +
                      RACE.Black + RACE.Native + RACE.Asian + RACE.other +
                      Hispanic + SEX + REGION.Midwest. + REGION.South. + REGION.West.
                    , dat_train, family = binomial)

s_dat_test <- predict(sobj, dat_test)
```
