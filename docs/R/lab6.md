---
layout: page
title: Survival analysis
permalink: /lab6/
parent: R
---


# Survival analysis

All data come from the study on the survival time of patients suffering from lung cancer. Section is given four 
files  with  data:  “part1_censored_XX.txt”,  “part1_uncensored_XX.txt”,  “part2_XX.txt”, 
“part3_XX.txt”,  and  “part4.txt”.

## Censored and uncensored data: Kaplan-Meier estimator
The aim of this laboratory was to implement by myself Kaplan-Meier estimator based on algorithm (instead of using already developed function from the library) using R language.

Both  of  them  refer  to  the  same  lung  cancer  patients:  in  the  first  one  information  concerning 
censoring  was  included,  but  in  the  second  one  it  was  not.  First  column  of  each  file  contains 
the time at which an event (death or censoring) occurred, while the second one consists of the type 
of the event (0 = dead, 1 = censoring). 
For the censored and uncensored datasets, KM estimator have been calculated.



```r
censored <- read.csv("part1_censored_08.txt",sep=" ")
uncensored <- read.csv("part1_uncensored_08.txt",sep=" ")

library(plyr)

source("survival_analysis_func.R")
df=survival_analysis_func(censored)

#uncensored
df_uncensored=survival_analysis_func(uncensored)
#For censored dataset
print(df)
```

```
##     t  n d c          s
## 1   0 60 0 0 1.00000000
## 2   1 60 3 0 0.95000000
## 3   2 57 3 0 0.90000000
## 4   3 54 5 0 0.81666667
## 5   4 49 4 0 0.75000000
## 6   5 45 4 0 0.68333333
## 7   6 41 7 1 0.56666667
## 8   7 33 1 0 0.54949495
## 9   8 32 2 0 0.51515152
## 10  9 30 6 0 0.41212121
## 11 10 24 2 0 0.37777778
## 12 11 22 6 0 0.27474747
## 13 12 16 1 0 0.25757576
## 14 13 15 2 0 0.22323232
## 15 15 13 1 0 0.20606061
## 16 18 12 1 0 0.18888889
## 17 20 11 1 0 0.17171717
## 18 22 10 2 1 0.13737374
## 19 25  6 1 0 0.11447811
## 20 30  5 1 0 0.09158249
## 21 34  4 1 2 0.06868687
## 22 47  1 1 0 0.00000000
```
For uncensored dataset
```r
print(df_uncensored)
```

```
##     t  n d c          s
## 1   0 60 0 0 1.00000000
## 2   1 60 3 0 0.95000000
## 3   2 57 3 0 0.90000000
## 4   3 54 5 0 0.81666667
## 5   4 49 4 0 0.75000000
## 6   5 45 4 0 0.68333333
## 7   6 41 8 0 0.55000000
## 8   7 33 1 0 0.53333333
## 9   8 32 2 0 0.50000000
## 10  9 30 6 0 0.40000000
## 11 10 24 2 0 0.36666667
## 12 11 22 6 0 0.26666667
## 13 12 16 1 0 0.25000000
## 14 13 15 2 0 0.21666667
## 15 15 13 1 0 0.20000000
## 16 18 12 1 0 0.18333333
## 17 20 11 1 0 0.16666667
## 18 22 10 4 0 0.10000000
## 19 25  6 1 0 0.08333333
## 20 30  5 1 0 0.06666667
## 21 34  4 1 0 0.05000000
## 22 35  3 1 0 0.03333333
## 23 45  2 1 0 0.01666667
## 24 47  1 1 0 0.00000000
```

```r
survival_analysis_func = function(censored) {
  censored <- censored[order(censored$Time), ]
  
  censored <-
    ddply(censored, .(censored$Time, censored$Censored), nrow)
  names(censored) <- c("Time", "Censored", "freq")
  
  df <- data.frame(
    t = integer(),
    n = integer(),
    d = integer(),
    c = integer(),
    s = double()
  )
  
  df[nrow(df) + 1, ] = c(0, sum(censored$freq), 0, 0, 1)
  
  while (nrow(censored) > 0) {
    if (censored$Censored[1] == 0) {
      df[nrow(df) + 1, ] <-c(censored$Time[1], 
                             sum(censored$freq),
                             censored$freq[1],
                             0,
                             df$s[nrow(df)] * (sum(censored$freq) - censored$freq[1]) / sum(censored$freq)
        )
    }
    else{
      df[nrow(df), 4] <- (df[nrow(df), 4] + 1)
    }
    censored = censored[-1, ]
  }
  return(df)
}
```

```r
library(ggplot2)
ggplot(data=df_uncensored,aes(x=t,y=s)) + geom_step()+ geom_step(data=df, aes(x=t,y=s), color="red")
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-1-1.png" style="display: block; margin: auto;" />

Censored is red curve, uncensored is black curve.
Censoring influence change the values of probability survival, but not dramatically. The uncensored lowered probability of survival in our case.

Censoring should be used in the survival analysis to get proper survival curve.

## Comaprison of two groups: log rank test

 These are the censored data from part I with 
one  additional  information  in  the  third  column:  the  general  stage  of  an  illness  at  the  time 
of diagnosis (either “good” or “bad”). First column contains the time at which an event (death 
or censoring) occurred, while the second one consists of the type of the event (0  =  dead, 1 = censoring).

**Plot of survival curves for good and bad stages**


```r
part2 <- read.csv("part2_08.txt",sep=" ")
part2good<-part2[part2$Stage=="Good",]
part2bad<-part2[part2$Stage=="Bad",]

part2good=survival_analysis_func(part2good)
part2bad=survival_analysis_func(part2bad)

ggplot(data=part2good,aes(x=t,y=s)) + geom_step()+ geom_step(data=part2bad, aes(x=t,y=s), color="red")
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-2-1.png" style="display: block; margin: auto;" />

Red colour curve represent Bad stage, on the other hand black curve present Good stage. In the Good stage, the survival is better. It seems that illness has influence on survival time.

**Log rank test**:

```r
#merge two df into one
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:plyr':
## 
##     arrange, count, desc, failwith, id, mutate, rename, summarise,
##     summarize
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
df_bad_good<-full_join(x = part2good,y=part2bad, by="t")

#sort by time
df_bad_good<-df_bad_good[order(df_bad_good$t),]

#eliminate NA from fields (take value above)
library(zoo)
```

```r
df_bad_good$n.x<-na.locf(df_bad_good$n.x, fromLast = TRUE,na.rm=FALSE)
df_bad_good$n.y<-na.locf(df_bad_good$n.y, fromLast = TRUE,na.rm=FALSE)
df_bad_good$s.x<-na.locf(df_bad_good$s.x, fromLast = TRUE,na.rm=FALSE)
df_bad_good$s.y<-na.locf(df_bad_good$s.y, fromLast = TRUE,na.rm=FALSE)

#eliminate in other columns
df_bad_good[is.na(df_bad_good)] <- 0

#calculate r1 and r2
df_bad_good$r.1<-df_bad_good$n.x-df_bad_good$c.x
df_bad_good$r.2<-df_bad_good$n.y-df_bad_good$c.y
print(df_bad_good)
```

```
##     t n.x d.x c.x       s.x n.y d.y c.y        s.y r.1 r.2
## 1   0  30   0   0 1.0000000  30   0   0 1.00000000  30  30
## 2   1  30   1   0 0.9666667  30   2   0 0.93333333  30  30
## 3   2  29   2   0 0.9000000  28   1   0 0.90000000  29  28
## 4   3  27   1   0 0.8666667  27   4   0 0.76666667  27  27
## 5   4  26   2   0 0.8000000  23   2   0 0.70000000  26  23
## 6   5  24   2   0 0.7333333  21   2   0 0.63333333  24  21
## 7   6  22   3   1 0.6333333  19   4   0 0.50000000  21  19
## 8   7  18   1   0 0.5981481  15   0   0 0.46666667  18  15
## 9   8  17   1   0 0.5629630  15   1   0 0.46666667  17  15
## 10  9  16   2   0 0.4925926  14   4   0 0.33333333  16  14
## 11 10  14   1   0 0.4574074  10   1   0 0.30000000  14  10
## 12 11  13   3   0 0.3518519   9   3   0 0.20000000  13   9
## 20 12  10   0   0 0.3166667   6   1   0 0.16666667  10   6
## 13 13  10   1   0 0.3166667   5   1   0 0.13333333  10   5
## 21 15   9   0   0 0.2814815   4   1   0 0.10000000   9   4
## 22 18   9   0   0 0.2814815   3   1   0 0.06666667   9   3
## 14 20   9   1   0 0.2814815   2   0   0 0.03333333   9   2
## 15 22   8   1   1 0.2462963   2   1   1 0.03333333   7   1
## 16 25   6   1   0 0.2052469   0   0   0 0.00000000   6   0
## 17 30   5   1   0 0.1641975   0   0   0 0.00000000   5   0
## 18 34   4   1   2 0.1231481   0   0   0 0.00000000   2   0
## 19 47   1   1   0 0.0000000   0   0   0 0.00000000   1   0
```

```r
#delete unnecessary column
df_bad_good <- df_bad_good[ -c(2,4:6,8)  ]

#merge d and r column
df_bad_good$d<-df_bad_good$d.x+df_bad_good$d.y
df_bad_good$r<-df_bad_good$r.1+df_bad_good$r.2

#calcaulte e and sigma
df_bad_good$e.1j<-df_bad_good$r.1*df_bad_good$d/df_bad_good$r
df_bad_good$e.2j<-df_bad_good$r.2*df_bad_good$d/df_bad_good$r
df_bad_good$sigma<-df_bad_good$r.1*df_bad_good$r.2*df_bad_good$d*(df_bad_good$r-df_bad_good$d)/(df_bad_good$r*df_bad_good$r*(df_bad_good$r-1))

df_bad_good = df_bad_good[-1,]
print(df_bad_good)
```

```
##     t d.x d.y        s.y r.1 r.2 d  r      e.1j      e.2j     sigma
## 2   1   1   2 0.93333333  30  30 3 60 1.5000000 1.5000000 0.7245763
## 3   2   2   1 0.90000000  29  28 3 57 1.5263158 1.4736842 0.7229917
## 4   3   1   4 0.76666667  27  27 5 54 2.5000000 2.5000000 1.1556604
## 5   4   2   2 0.70000000  26  23 4 49 2.1224490 1.8775510 0.9339858
## 6   5   2   2 0.63333333  24  21 4 45 2.1333333 1.8666667 0.9276768
## 7   6   3   4 0.50000000  21  19 7 40 3.6750000 3.3250000 1.4770673
## 8   7   1   0 0.46666667  18  15 1 33 0.5454545 0.4545455 0.2479339
## 9   8   1   1 0.46666667  17  15 2 32 1.0625000 0.9375000 0.4819808
## 10  9   2   4 0.33333333  16  14 6 30 3.2000000 2.8000000 1.2358621
## 11 10   1   1 0.30000000  14  10 2 24 1.1666667 0.8333333 0.4649758
## 12 11   3   3 0.20000000  13   9 6 22 3.5454545 2.4545455 1.1050767
## 20 12   0   1 0.16666667  10   6 1 16 0.6250000 0.3750000 0.2343750
## 13 13   1   1 0.13333333  10   5 2 15 1.3333333 0.6666667 0.4126984
## 21 15   0   1 0.10000000   9   4 1 13 0.6923077 0.3076923 0.2130178
## 22 18   0   1 0.06666667   9   3 1 12 0.7500000 0.2500000 0.1875000
## 14 20   1   0 0.03333333   9   2 1 11 0.8181818 0.1818182 0.1487603
## 15 22   1   1 0.03333333   7   1 2  8 1.7500000 0.2500000 0.1875000
## 16 25   1   0 0.00000000   6   0 1  6 1.0000000 0.0000000 0.0000000
## 17 30   1   0 0.00000000   5   0 1  5 1.0000000 0.0000000 0.0000000
## 18 34   1   0 0.00000000   2   0 1  2 1.0000000 0.0000000 0.0000000
## 19 47   1   0 0.00000000   1   0 1  1 1.0000000 0.0000000       NaN
```

```r
chi_square1=sum((df_bad_good$d.x-df_bad_good$e.1j)^2)/sum(df_bad_good$sigma[1:18])
print(chi_square1)
```

```
## [1] 0.6746379
```

In the last line, we achieved NA number. To calculate chi-square, it was missed due to zero value (it is dividing by 0). 

**Log rank test:**

Null hypothesis: No differences between survival curves. 

Alternative hypothesis: There is a statistically significant difference between the survival curves.

Critical value: 5.024>chi_square1>0.001

P-value is higher than 0.05 (0.411)

Assuming a 5% significance level, we do not have enough evidence to reject H0 and accept H1. Survival curves are the same.

Survival time diagnosed at good and bad stages doesn't differ significantly. By log rank test, we thought that assumption in previous was incorrect (that the curves are significantly different).

## Comparison of three or more groups: log-rank test
 These are the survival times of the lung 
cancer  patients  treated  with  one  of  three  different  therapies:  radiotherapy,  chemotherapy, 
and surgery.  First  column  contains  the  time  at  which  an  event  (death  or  censoring)  occurred, 
the second one consists of the type of the event (0 = dead, 1 = censoring), and the third one the type of treatment. 

**Plot of survival curves for different method of threatment (chemotherapy, radiotherapy and surgery).**


```r
part3 <- read.csv("part3_08.txt",sep=" ")
part3chem<-part3[part3$Therapy=="Chemotherapy",]
part3surgery<-part3[part3$Therapy=="Surgery",]
part3radio<-part3[part3$Therapy=="Radiotherapy",]

part3chem=survival_analysis_func(part3chem)
part3surgery=survival_analysis_func(part3surgery)
part3radio=survival_analysis_func(part3radio)

ggplot(data=part3chem,aes(x=t,y=s)) + geom_step()+ geom_step(data=part3surgery, aes(x=t,y=s), color="red")+geom_step(data=part3radio, aes(x=t,y=s), color="green")
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

From the chart, we can assume that the longest live of patient is connected with chemotherapy method (black colour). Second method is radiotherapy (greed colour) and the last surgery (red colour).

**Log rank test for all three survival curves:**




```r
#revrse 0 to 1, and 1 to 0 to proper use library
part3$Censored[part3$Censored>0] <- 2
part3$Censored[part3$Censored<1] <- 1
part3$Censored[part3$Censored>1] <- 0
print(survdiff(Surv(Time, Censored) ~ Therapy, data = part3))
```

```
## Call:
## survdiff(formula = Surv(Time, Censored) ~ Therapy, data = part3)
## 
##                       N Observed Expected (O-E)^2/E (O-E)^2/V
## Therapy=Chemotherapy 20       16    28.28     5.334    12.991
## Therapy=Radiotherapy 20       20    18.58     0.109     0.182
## Therapy=Surgery      20       18     7.14    16.510    22.789
## 
##  Chisq= 26.8  on 2 degrees of freedom, p= 1e-06
```

Null hypothesis: No differences between survival curves.

Alternative hypothesis: There are statistically significant differences between the survival curves.

**Conclusion**: the p-value gave us enough power to reject null hypothesis and accept alternative. It means that there are differences between survival curves.


```r
#teleradio and surgery
fit <- survfit(Surv(Time, Censored) ~ Therapy, data = part3[1:40,])

ggsurvplot(
  fit,
  conf.int = TRUE,
  surv.median.line = c('hv'), 
  data = part3, 
  pval = TRUE,
  pval.method = TRUE,
  risk.table = FALSE)
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-7-1.png" style="display: block; margin: auto;" />

Plot pair of therapies:


```r
#surgery and chemotherapy
fit <- survfit(Surv(Time, Censored) ~ Therapy, data = part3[21:60,])

ggsurvplot(
  fit,
  conf.int = TRUE,
  surv.median.line = c('hv'), 
  data = part3, 
  pval = TRUE,
  pval.method = TRUE,
  risk.table = FALSE)
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-8-1.png" style="display: block; margin: auto;" />


```r
#teleradio and surgery
fit <- survfit(Surv(Time, Censored) ~ Therapy, data = part3[c(1:20,41:60),])

ggsurvplot(
  fit,
  conf.int = TRUE,
  surv.median.line = c('hv'), 
  data = part3, 
  pval = TRUE,
  pval.method = TRUE,
  risk.table = FALSE)
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-9-1.png" style="display: block; margin: auto;" />

Results: the plots confirm that are differences between methods.

**Log rank test for pairs:**


```r
#teleradio and chemotherapy
print(survdiff(Surv(Time, Censored) ~ Therapy, data = part3[c(1:20,41:60),]))
```

```
## Call:
## survdiff(formula = Surv(Time, Censored) ~ Therapy, data = part3[c(1:20, 
##     41:60), ])
## 
##                       N Observed Expected (O-E)^2/E (O-E)^2/V
## Therapy=Chemotherapy 20       16     22.7      1.97      5.81
## Therapy=Radiotherapy 20       20     13.3      3.35      5.81
## 
##  Chisq= 5.8  on 1 degrees of freedom, p= 0.02
```

```r
#surgery and chemotherapy
print(survdiff(Surv(Time, Censored) ~ Therapy, data = part3[21:60,]))
```

```
## Call:
## survdiff(formula = Surv(Time, Censored) ~ Therapy, data = part3[21:60, 
##     ])
## 
##                       N Observed Expected (O-E)^2/E (O-E)^2/V
## Therapy=Chemotherapy 20       16    25.97      3.83      21.4
## Therapy=Surgery      20       18     8.03     12.38      21.4
## 
##  Chisq= 21.4  on 1 degrees of freedom, p= 4e-06
```

```r
#teleradio and surgery
print(survdiff(Surv(Time, Censored) ~ Therapy, data = part3[1:40,]))
```

```
## Call:
## survdiff(formula = Surv(Time, Censored) ~ Therapy, data = part3[1:40, 
##     ])
## 
##                       N Observed Expected (O-E)^2/E (O-E)^2/V
## Therapy=Radiotherapy 20       20     27.3      1.94      8.76
## Therapy=Surgery      20       18     10.7      4.95      8.76
## 
##  Chisq= 8.8  on 1 degrees of freedom, p= 0.003
```

Null hypothesis: No differences between survival curves. 

Alternative hypothesis: There are statistically significant differences between the survival curves.

In all three cases, we have enough power to reject null hypothesis and accept alternative. That means, the method have influence for patient survival. 

## Cox proportional hazard model
These are the survival times of the lung cancer 
patients  accompanied  with  the  information  about  three  more  features:  age  (in  years),  sex 
(baseline:  Female),  and  calories  consumed  at  meals.  First  column  contains  the  time  at  which 
an event (death or censoring) occurred, while the second one consists of the type of the event (1 = dead, 0 = censoring). 
Dataset info:

```r
part4 <- read.csv("part4.txt",sep=" ")
summary(part4)
```

```
##       Time          Censoring           Age            Sex     
##  Min.   :   5.0   Min.   :0.0000   Min.   :39.00   Female: 67  
##  1st Qu.: 163.0   1st Qu.:0.0000   1st Qu.:57.00   Male  :114  
##  Median : 259.0   Median :1.0000   Median :64.00               
##  Mean   : 298.9   Mean   :0.7403   Mean   :62.68               
##  3rd Qu.: 387.0   3rd Qu.:1.0000   3rd Qu.:70.00               
##  Max.   :1022.0   Max.   :1.0000   Max.   :82.00               
##     Calories     
##  Min.   :  96.0  
##  1st Qu.: 635.0  
##  Median : 975.0  
##  Mean   : 928.8  
##  3rd Qu.:1150.0  
##  Max.   :2600.0
```

Sex column is factorial (with male and female as a gender). Age and Calories are continuous variable. In previous part, dead and censoring factorials had swapped labels. In this case (1 = dead, 0 = censoring). 

**Division into train and test dataset:**

```r
smp_size <- floor(0.8 * nrow(part4))
set.seed(120, sample.kind = "Rejection")
train_ind <- sample(seq_len(nrow(part4)), size = smp_size)

train <- part4[train_ind, ]
test <- part4[-train_ind, ]
```

We need to be aware, that in new verion >=3.6.0 there is different type of sampling method. Samples from console and knit might we different despite using same seed number.

**Build full Cox proportional hazard model on the training set:**


```r
model<-coxph(Surv(Time, Censoring) ~ Age + Sex + Calories, train) 
print(model)
```

```
## Call:
## coxph(formula = Surv(Time, Censoring) ~ Age + Sex + Calories, 
##     data = train)
## 
##                coef  exp(coef)   se(coef)      z      p
## Age       0.0179783  1.0181408  0.0124150  1.448 0.1476
## SexMale   0.3750064  1.4550007  0.2104115  1.782 0.0747
## Calories -0.0001560  0.9998440  0.0002631 -0.593 0.5533
## 
## Likelihood ratio test=7.27  on 3 df, p=0.06384
## n= 144, number of events= 107
```

```r
summary(model)
```

```
## Call:
## coxph(formula = Surv(Time, Censoring) ~ Age + Sex + Calories, 
##     data = train)
## 
##   n= 144, number of events= 107 
## 
##                coef  exp(coef)   se(coef)      z Pr(>|z|)  
## Age       0.0179783  1.0181408  0.0124150  1.448   0.1476  
## SexMale   0.3750064  1.4550007  0.2104115  1.782   0.0747 .
## Calories -0.0001560  0.9998440  0.0002631 -0.593   0.5533  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
##          exp(coef) exp(-coef) lower .95 upper .95
## Age         1.0181     0.9822    0.9937     1.043
## SexMale     1.4550     0.6873    0.9633     2.198
## Calories    0.9998     1.0002    0.9993     1.000
## 
## Concordance= 0.598  (se = 0.032 )
## Likelihood ratio test= 7.27  on 3 df,   p=0.06
## Wald test            = 7.02  on 3 df,   p=0.07
## Score (logrank) test = 7.09  on 3 df,   p=0.07
```

Null model:


```r
modelnull<-coxph(Surv(Time, Censoring) ~ 1, train) 
summary(modelnull)
```

```
## Call:  coxph(formula = Surv(Time, Censoring) ~ 1, data = train)
## 
## Null model
##   log likelihood= -437.4219 
##   n= 144
```

**Likelihood Ratio Test:**

H0: (null model)

Ha: (model with Sex, Calories and Age features)

p-value: 0.06383463 (from summary function, alternative way: 1 - pchisq(-2*(-437.4219--433.7880), df=3))

We cannot reject null hypothesis and accept alternative. 

Conclusion: depending on the sampling method (despite seed number), the result from LRT might be different (as in this example).


**Wald test:**

H0: the coefficient is equal zero

Ha: the coefficient is not equal zero

In summary model all coefficients have p-value over 0.05 (prior to summary table). It means that in all cases we don't have enough power to reject null hypothesis. The closest p-value to the 0.05 is connected with Sex feature.

**Hazard Ratios**

HR can be found by summary function.

```r
ggforest(model)
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-16-1.png" style="display: block; margin: auto;" />

Hazard ratio by ggforest give us hazard with p-value (same as from summary function). In this case, Age and Calories are "unsignificant" predictors. In Sex predictor, Female is reference base. In Male population, the hazard ratio is highest, but whiskers include 1 value (as confidence interval). That means, we don't have enough power to say that Sex coresponds significantly to the hazard ratio. 


**Cox proportional hazard model with a brute force method:**

For eight different model has been created Cox PH (all possible combination of features):

```r
model1<-coxph(Surv(Time, Censoring) ~ Age + Sex + Calories, train)
model2<-coxph(Surv(Time, Censoring) ~ Age + Calories, train)
model3<-coxph(Surv(Time, Censoring) ~ Age + Sex, train)
model4<-coxph(Surv(Time, Censoring) ~ Sex + Calories, train)
model5<-coxph(Surv(Time, Censoring) ~ Age, train)
model6<-coxph(Surv(Time, Censoring) ~ Sex, train)
model7<-coxph(Surv(Time, Censoring) ~ Calories, train)
model8<-coxph(Surv(Time, Censoring) ~ 1, train)

x<-c(1,2,3,4,5,6,7,8)
y<-c(extractAIC(model1)[2],extractAIC(model2)[2],extractAIC(model3)[2],extractAIC(model4)[2],extractAIC(model5)[2],extractAIC(model6)[2],extractAIC(model7)[2],extractAIC(model8)[2])
plot(x,y)
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-17-1.png" style="display: block; margin: auto;" />

The axis-x is number of model (as in R script above), the axis-y is AIC value per each model.

Instead of using BIC function, it is possible to calculate by own BIC value. Documentation of extractAIC function, give us: AIC = - 2  * log L + k * edf

While BIC k=log(n), where n is number of observations.
From the model{X}$loglik, it is possible to get log likelihood of the model and null model. 


```r
print(-2*model1$loglik[2]+model1$iter*log(model1$n))
```

```
## [1] 882.4855
```

```r
print(-2*model1$loglik[2]+model1$iter*log(model1$nevent))
```

```
## [1] 881.5946
```

```r
print(BIC(model1))
```

```
## [1] 881.5946
```

**Ready BIC function get only not censored value!**

```r
x<-c(1,2,3,4,5,6,7,8)
y<-c()
y[1]<-(-2*model1$loglik[2]+model1$iter*log(model1$n))
y[2]<-(-2*model2$loglik[2]+model2$iter*log(model2$n))
y[3]<-(-2*model3$loglik[2]+model3$iter*log(model3$n))
y[4]<-(-2*model4$loglik[2]+model4$iter*log(model4$n))
y[5]<-(-2*model5$loglik[2]+model5$iter*log(model5$n))
y[6]<-(-2*model6$loglik[2]+model6$iter*log(model6$n))
y[7]<-(-2*model7$loglik[2]+model7$iter*log(model7$n))
y[8]<-(-2*model8$loglik)
plot(x,y)
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-19-1.png" style="display: block; margin: auto;" />

BIC "penalize" larger models and has a larger pentalty than AIC. In this situation, BIC penalty is too strong (model8 as a null model).

From the AIC and BIC chart, some conclusion can be made about the "best" candidate to the model. In model3 and model6, Sex feature is take into account and the AIC value small in both cases. In prevoius part, from Hazard Ratio, it was concluded that Sex cathegory might have the higest impact on the precition lung cancer.

**The lowest AIC value is in model3 (Age + Sex).** In our opinion it could be the best model to predict lung cancer.


**Plot survival for best chosen model for training and test dataset**:

```r
model3<-coxph(Surv(Time, Censoring) ~ Age + Sex, train)
Cox_curve <- survfit(model3)
plot(Cox_curve)
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-20-1.png" style="display: block; margin: auto;" />

```r
model3_1<-coxph(Surv(Time, Censoring) ~ Age + Sex, test)
Cox_curve_1 <- survfit(model3_1)
plot(Cox_curve_1)
```

<img src="{{site.url}}/assets/images/lab6_files/figure-html/unnamed-chunk-20-2.png" style="display: block; margin: auto;" />

The x-axis is Time, the y-axis is probability of survival. As the test set containts less number of observations, the shape of curve is similar to the "stairs". Confidence intervals for test set are wider.

**Conclusion:** Survival of lung cancer patient may depends on the gender. Female has more chance to have longer life. It may be correlated with higher risk of the dust/chemicals in industry, where ratio of male workers is higher.
