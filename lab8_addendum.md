Lab 8 addendum
================

### Econ B2000, MA Econometrics

### Kevin R Foster, the Colin Powell School at the City College of New York, CUNY

### Fall 2025

This is some coding background – not as much to ponder, just preparing
for Lab 9.

We do some more complicated coding now, in order to make later coding
less complicated. It’s an investment (so profitability depends on your
personal discount rate).

We want just 2 simple things: to standardize our data (all X-variables
to have values just in \[0,1\] interval) and to split the data into a
training set (that we use to estimate the model) and a test set (that we
use to evaluate how well the model performs on new data that it hasn’t
trained on).

But depending on your model, doing just those 2 simple things can take a
bit of work. Best to do that in the privacy of your own home.

I’ll show this for a simple set of X-variables. Your version will be
more complicated. The ones that are already nice dummy variables are
easy and this coding might seem overly elaborate for them. But bigger
factors such as the state (`STATEFIP`) get ugly fast.

I’m not necessarily saying you should use `STATEFIP` in your regression,
only that if you want to use a big factor with many levels, this is a
way to do it. I’ll show some code to create a new `State_factor` that
we’ll use.

``` r
# fix each variable you want in your regression
# this example is for small version, 
# he_more_than_5yrs_than_her ~ educ_hs + educ_somecoll + educ_college + educ_advdeg + AGE + State_factor

# you can change this part, depending on your analysis
dat_use <- trad_data

dat_use$State_factor <- as.factor(dat_use$STATEFIP)

d_y_varb <- data.frame(model.matrix(~ dat_use$he_more_than_5yrs_than_her)) 

d_educ_hs <- data.frame(model.matrix(~ dat_use$educ_hs))
d_educ_somecoll <- data.frame(model.matrix(~ dat_use$educ_somecoll))
d_educ_college <- data.frame(model.matrix(~ dat_use$educ_college))
d_educ_advdeg <- data.frame(model.matrix(~ dat_use$educ_advdeg))
d_age <- data.frame(model.matrix(~ dat_use$AGE))
d_State <- data.frame(model.matrix(~ dat_use$State_factor)) # which is big!
```

In this step (and later) I worry that I don’t want to accidentally
create factors that are empty. Depending on your subgroup that you
choose, this might happen. That will cause problems for later estimation
(the math tries to answer the question, how are the zero observations in
some group different from the other groups?). So we want to catch the
problem early. Run colSums() to verify.

``` r
sum( colSums(d_State) == 0) # should be zero
```

Then this puts them all together,

``` r
# there are better ways to code this, but this should be more robust to your other choices

dat_for_analysis_sub <- data.frame(
  d_y_varb[,2], # need [] since model.matrix includes intercept term
  d_educ_hs[,2],
  d_educ_somecoll[,2],
  d_educ_college[,2],
  d_educ_advdeg[,2],
  d_age[,2],
  d_State[,2:length(d_State)] ) # this last term is why model.matrix 


# this is just about me being anal-retentive, see difference in names(dat_for_analysis_sub) before and after running this bit
names(dat_for_analysis_sub)
names(dat_for_analysis_sub) <- sub("dat_use.","",names(dat_for_analysis_sub)) # drops each repetition of dat_use with some regexp foo 
names(dat_for_analysis_sub) <- sub("factor","",names(dat_for_analysis_sub)) # drops each repetition of factor

names(dat_for_analysis_sub)[1] <- "he_more_than_5yrs_than_her"
names(dat_for_analysis_sub)[2:5] <- c("HS","SomeColl","College","AdvDeg")
names(dat_for_analysis_sub)[6] <- "Age"

names(dat_for_analysis_sub)
```

Then to create training data and test data,

``` r
require("standardize")
set.seed(654321)
NN <- length(dat_for_analysis_sub$he_more_than_5yrs_than_her)

restrict_1 <- (runif(NN) < 0.1) # use 10% as training data, ordinarily this would be much bigger but start small
summary(restrict_1) # you should understand wtf this variable is for
dat_train <- subset(dat_for_analysis_sub, restrict_1)
dat_test <- subset(dat_for_analysis_sub, !restrict_1)

# again check this below, should be zero
sum( colSums(dat_train) == 0)
```

Now writing the formula is a bit of a pain. Would like to have
‘he_more_than_5yrs_than_her ~ HS + SomeColl + College + AdvDeg + Age +
State’ but that last term is no longer an easy factor but a mess of 50
dummies! Don’t copy-paste 50 times, instead:

``` r
formula_sobj <- reformulate( names(dat_for_analysis_sub[2:length(dat_for_analysis_sub)]), response = "he_more_than_5yrs_than_her")

sobj <- standardize(formula_sobj, dat_train, family = binomial) # standardized object

s_dat_test <- predict(sobj, dat_test)
```

Now your OLS and logit models can be run like this:

``` r
# OLS linear probability model
model_lpm1 <- lm(sobj$formula, data = sobj$data)
summary(model_lpm1)
pred_vals_lpm <- predict(model_lpm1, s_dat_test)
pred_model_lpm1 <- (pred_vals_lpm > mean(pred_vals_lpm))
table(pred = pred_model_lpm1, true = dat_test$he_more_than_5yrs_than_her )

# logit 
model_logit1 <- glm(sobj$formula, family = binomial, data = sobj$data)
summary(model_logit1)
pred_vals <- predict(model_logit1, s_dat_test, type = "response")
pred_model_logit1 <- (pred_vals > mean(pred_vals) )
table(pred = pred_model_logit1, true = dat_test$he_more_than_5yrs_than_her )
```

For now you wouldn’t see much saving – basically all that’s been done is
that we’ve standardized the X-variables of the model (you can run
`summary(sobj$data)` to verify). But OLS and logit models in R can
handle variables that are not standardized so there’s not yet any juice
from all that work. The juice comes later, when we run other procedures
that couldn’t manage un-standardized inputs. Which we’ll do in Lab 9.

We’ve also separated the data into training and test sets, so if you
compare your predicted values from here with the ones from previous,
you’d see some differences. We’ve made a somewhat stricter test of the
estimated model. Previously we’d evaluated how well the model could
predict the data it was trained on; this time we’re evaluating how well
it could predict on new data that it hadn’t seen in modeling.
