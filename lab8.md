Lab 8
================

### Econ B2000, MA Econometrics

### Kevin R Foster, the Colin Powell School at the City College of New York, CUNY

### Fall 2025

This is Part 2 of a 3-part series. Part 1 estimated OLS. Part 2 will
estimate logit models; Part 3 will estimate fancier machine learning
models.

Using the same couples data as last time, for this lab we’ll estimate
logit and probit models instead of just OLS as we did in Lab 7. But if
you use the same subgroup and same model, you can compare the
predictions from each method. Look at subgroups to see if there are
particular groups where the models are more confused. Look at the
tradeoff of false positive vs false negative. Are there explanatory
variables (features) that are consistently of little predictive value?
Can you find better ones?

Are these X-variables exogenous? As you add more, think about causality.

``` r
library(ggplot2)
library(tidyverse)
library(haven)

setwd("..//ACS_2021_PUMS//") # your directory structure will be different
load("ACS_2021_couples.RData")
```

This was the linear model from last time; now I’ve shown the table of
predicted values.

``` r
ols_out1 <- lm(he_more_than_5yrs_than_her ~ educ_hs + educ_somecoll + educ_college + educ_advdeg + AGE, data = trad_data)

pred_vals_ols1 <- predict(ols_out1, trad_data)
pred_model_ols1 <- (pred_vals_ols1 > mean(pred_vals_ols1))
table(pred = pred_model_ols1, true = trad_data$he_more_than_5yrs_than_her)
```

    ##        true
    ## pred         0      1
    ##   FALSE 180783  35078
    ##   TRUE  150394  46020

``` r
model_logit1 <- glm(he_more_than_5yrs_than_her ~ educ_hs + educ_somecoll + educ_college + educ_advdeg + AGE, data = trad_data, family = binomial)
summary(model_logit1)
```

    ## 
    ## Call:
    ## glm(formula = he_more_than_5yrs_than_her ~ educ_hs + educ_somecoll + 
    ##     educ_college + educ_advdeg + AGE, family = binomial, data = trad_data)
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)   -0.2637952  0.0186745  -14.13   <2e-16 ***
    ## educ_hs       -0.2825620  0.0151416  -18.66   <2e-16 ***
    ## educ_somecoll -0.3815243  0.0160547  -23.76   <2e-16 ***
    ## educ_college  -0.6128375  0.0161218  -38.01   <2e-16 ***
    ## educ_advdeg   -0.6760222  0.0173786  -38.90   <2e-16 ***
    ## AGE           -0.0140343  0.0002452  -57.23   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 408816  on 412274  degrees of freedom
    ## Residual deviance: 403701  on 412269  degrees of freedom
    ## AIC: 403713
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
pred_vals <- predict(model_logit1, trad_data, type = "response")
pred_model_logit1 <- (pred_vals > mean(pred_vals))
table(pred = pred_model_logit1, true = trad_data$he_more_than_5yrs_than_her)
```

    ##        true
    ## pred         0      1
    ##   FALSE 191616  37461
    ##   TRUE  139561  43637

You can play around to see if the `predvals > mean` cutoff is best or
use `> 0.5` or some other value. These give a table about how the models
predict.

And make sure to give me some better output, it’s time to stop dumping
all your output into one file but instead get thoughtful about
presenting results.
