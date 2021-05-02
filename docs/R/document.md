---
layout: page
title: Advanced clustering 
permalink: /document/
parent: R
---

# Advanced clustering on single-cell RNA-sequencing

The data used for this report are the results of the single-cell RNA sequencing.  That is the state-of-the-art method of measuring the expression levels in a particular cell. This kind of experiment generates huge data:  expression levels of thousands of genes are measured for hundreds or thousands of cells. Each cell is also described with the annotation data. In the case of these classes, every cell is annotated with the tissue or organ of origin. All cells were collected from the same mouse. This study aims to distinguish between cells coming from different tissues based on the measured expression levels. 

## Part I
**Hierarchical clustering: Gaussing Mixture Model (GMM) decomposition as a method of feature selection**

The aim of this study is to distinguish between cells coming from different tissues on the basis of the measured expression levels. [single-cell RNA-requencing]

**1. Perform hierarchical clustering and draw the dendrogram. Comment on the results.**



```r
load("sl8.RData") 
d<-dist(df)
```

```
## Warning in dist(df): NAs introduced by coercion
```

```r
model<-hclust(d)
plot(model, labels=df[,dim(df)[2]], cex = 0.5) 
```

<img src="{{site.url}}/assets/images/document_files/figure-html/unnamed-chunk-2-1.png" width="672" style="display: block; margin: auto;" />

Clustering by usage dendrogram gave us unsatisfactory results. The tree has no significant cluster meaning. It doesn't' divide into a separate node with the same human body tissue.

**2. Draw  the  heatmap  of  the  first  1000  features  with  the  dendrogram  corresponding  to  cells (not genes). Comment on the results. Do expression levels of all genes vary across different cells?**

```r
heatmap(as.matrix(df[,1:1000]),scale='none',labRow = df[,dim(df)[2]],)
```

<img src="{{site.url}}/assets/images/document_files/figure-html/unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

Expression levels vary across different cells but not in all genes. In some cases, the variance is equal to zero. Those genes should not take account in the further stages of the investigation. 

**3. Remove from the dataset those genes, for which the expression levels are equal for all cells.**

```r
myFunction = function(x){
  length(unique(x))==1
}

xx<-apply(df,2,myFunction)
df_removed<-df[,which(xx==FALSE)]
```

**4. Compute  mean  expression  levels  in  log 10   scale  for  the remaining set  of  genes.  Draw the histogram of those values with the number of bins equal to the square root of the number of genes.**

```r
means<-colMeans(df_removed[,1:(dim(df_removed)[2]-1)])

root<-sqrt(dim(df_removed)[2]-1)

means<-(log10(means))
hist(means,breaks=seq(min(means),max(means),l=root))
```

<img src="{{site.url}}/assets/images/document_files/figure-html/unnamed-chunk-5-1.png" style="display: block; margin: auto;" />

**5. Perform GMM decomposition of the mean values in log 10  scale. Plot the components. **

```r
modelGMM<-Mclust(means)

#model paramaters
modelGMM[13]
```

```
## $parameters
## $parameters$pro
## [1] 0.04491080 0.09792484 0.09837609 0.17864913 0.16227475 0.12394991 0.15412848
## [8] 0.13978599
## 
## $parameters$mean
##          1          2          3          4          5          6          7 
## 0.09996775 0.11099161 0.13549821 0.19475897 0.28978262 0.38344168 0.48780542 
##          8 
## 0.61028058 
## 
## $parameters$variance
## $parameters$variance$modelName
## [1] "V"
## 
## $parameters$variance$d
## [1] 1
## 
## $parameters$variance$G
## [1] 8
## 
## $parameters$variance$sigmasq
## [1] 3.414188e-06 3.966316e-05 2.191551e-04 1.226703e-03 1.619461e-03
## [6] 2.082662e-03 6.274158e-03 2.126732e-02
## 
## $parameters$variance$scale
## [1] 3.414188e-06 3.966316e-05 2.191551e-04 1.226703e-03 1.619461e-03
## [6] 2.082662e-03 6.274158e-03 2.126732e-02
```

**6. We assume that genes with low expression levels are the noise and cannot be informative in terms of differences between cells. Remove those genes from the dataset. To do that, set the cut-off value as the intersection of two components with the lowest means.**

```r
gmm_thres <- function(mu1,mu2,s1,s2, p1, p2) {
  a=(s1^2-s2^2)/(2*s1^2*s2^2)
  b=(mu1*s2^2-mu2*s1^2)/(s1^2*s2^2)
  c=(-mu1^2*s2^2+mu2^2*s1^2)/(2*s1^2*s2^2)-log((s1*p2)/(s2*p1))
  
  delta=b^2-4*a*c
  
  x_1 = (-b+sqrt(delta))/(2*a)
  x_2 = (-b-sqrt(delta))/(2*a)
  result = c(x_1,x_2)
  
  return(result)
}

#get index of the lowest element
index_1<-sort(modelGMM$parameters$mean, decreasing=F)[1]
index1<-match(c(index_1),modelGMM$parameters$mean)
index_2<-sort(modelGMM$parameters$mean, decreasing=F)[2]
index2<-match(c(index_2),modelGMM$parameters$mean)

p1=modelGMM$parameters$pro[index1]
p2=modelGMM$parameters$pro[index2]

s1=modelGMM$parameters$variance$sigmasq[index1]
s2=modelGMM$parameters$variance$sigmasq[index2]

mu1=modelGMM$parameters$mean[index1]
mu2=modelGMM$parameters$mean[index2]

#result of cut off
result<-gmm_thres(mu1,mu2,s1,s2, p1, p2)

print(result)
```

```
##          1          1 
## 0.09892942 0.10084149
```

```r
if(result[1]>index_1 & result[1]<index_2){
  means_upt<-means[means>=result[1]]
}else{
  means_upt<-means[means>=result[2]]
}
```

The higher value from the quadratic equation should be used to set the cut-off point. The parabolic is quite tight when crossing zero value.

**7. Compute variances of expression levels in log 10  scale for the remaining set of genes. Draw the histogram of those values with the number of bins equal to the square root of the number of genes.**

```r
name<-attr(means_upt, "names")
means_upt<-df[ , which(names(df) %in% name)]

#calc variance
Variance<-sapply(means_upt, var)

root<-sqrt(dim(as.matrix(Variance))[1])
Variance<-(log10(Variance))
hist(Variance,breaks=seq(min(Variance),max(Variance),l=root))
```

<img src="{{site.url}}/assets/images/document_files/figure-html/unnamed-chunk-8-1.png" style="display: block; margin: auto;" />

**8. Perform GMM decomposition of the variance values in log 10  scale. Plot the components.**

```r
modelGMM<-Mclust(Variance)

#model paramaters
modelGMM[13]
```

```
## $parameters
## $parameters$pro
## [1] 0.03086523 0.27654924 0.21082032 0.16807964 0.16819192 0.14549366
## 
## $parameters$mean
##          1          2          3          4          5          6 
## -1.3646853 -0.3473932  0.3059969  0.6061295  0.8245434  0.6979975 
## 
## $parameters$variance
## $parameters$variance$modelName
## [1] "V"
## 
## $parameters$variance$d
## [1] 1
## 
## $parameters$variance$G
## [1] 6
## 
## $parameters$variance$sigmasq
## [1] 0.03566175 0.19755241 0.05039356 0.01841838 0.01052116 0.04256266
## 
## $parameters$variance$scale
## [1] 0.03566175 0.19755241 0.05039356 0.01841838 0.01052116 0.04256266
```

**9. We  assume  that  genes  with  high  variances  of  expression  levels  are  the  most  informative in terms of differences between cells. Keep only those genes in the dataset. To do that, set the cut-off value as the intersection of two components with the highest variances.**

```r
gmm_thres <- function(mu1,mu2,s1,s2, p1, p2) {
  a=(s1^2-s2^2)/(2*s1^2*s2^2)
  b=(mu1*s2^2-mu2*s1^2)/(s1^2*s2^2)
  c=(-mu1^2*s2^2+mu2^2*s1^2)/(2*s1^2*s2^2)-log((s1*p2)/(s2*p1))
  
  delta=b^2-4*a*c
  
  x_1 = (-b+sqrt(delta))/(2*a)
  x_2 = (-b-sqrt(delta))/(2*a)
  result = c(x_1,x_2)
  
  return(result)
}

#get index of the lowest element
index_1<-sort(modelGMM$parameters$mean, decreasing=T)[1]
index1<-match(c(index_1),modelGMM$parameters$mean)
index_2<-sort(modelGMM$parameters$mean, decreasing=T)[2]
index2<-match(c(index_2),modelGMM$parameters$mean)

p1=modelGMM$parameters$pro[index1]
p2=modelGMM$parameters$pro[index2]

s1=modelGMM$parameters$variance$sigmasq[index1]
s2=modelGMM$parameters$variance$sigmasq[index2]

mu1=modelGMM$parameters$mean[index1]
mu2=modelGMM$parameters$mean[index2]

#result of cut off
result<-gmm_thres(mu1,mu2,s1,s2, p1, p2)

print(result)
```

```
##         5         5 
## 0.7943896 0.8711685
```

```r
if(result[1]>index_1 & result[1]<index_2){
  Variance<-Variance[Variance>=result[1]]
}else{
  Variance<-Variance[Variance>=result[2]]
}
```

**10. Perform hierarchical clustering and draw the dendrogram using only the remaining genes. Comment on the results.**

```r
name<-attr(Variance, "names")
df_mod<-df[ , which(names(df) %in% name)]

#add column labels
df_mod<-cbind(df_mod, df_removed[,(dim(df_removed)[2])])

d<-dist(df_mod)
```

```
## Warning in dist(df_mod): NAs introduced by coercion
```

```r
model<-hclust(d)
plot(model, labels=df_mod[,dim(df_mod)[2]], cex = 0.7) 
```

<img src="{{site.url}}/assets/images/document_files/figure-html/unnamed-chunk-11-1.png" style="display: block; margin: auto;" />

The most problematic tissue is related to Marrow and Spleen tissue. They seem to be similar, and clusters have problems with proper clusterization.

**11. Draw  the  heatmap  of  the  remaining  dataset  with  the  dendrogram  corresponding  to  cells (not genes). Comment on the results. Do expression levels of all genes vary across different cells?**

```r
heatmap(as.matrix(df_mod[,-dim(df_mod)[2]]),scale='none',labRow = df_mod[,dim(df_mod)[2]],)
```

<img src="{{site.url}}/assets/images/document_files/figure-html/unnamed-chunk-12-1.png" style="display: block; margin: auto;" />

Expression level may vary across different cells. Especially it is visible in Colon, that cells have different expression based on heatmap but in some cases, visually they have similar expression level.

## Part II
**Gaussing Mixture Model (GMM) decomposition as a clustering method**

**1. Perform GMM decomposition of the expression levels of gene Perp**

```r
df4<-df_part4[,grep("Perp", names(df_part4))]

modelGMM<-Mclust(df4)
modelGMM[13]
```

```
## $parameters
## $parameters$pro
## [1] 0.58342898 0.01630221 0.03028410 0.04121962 0.05248204 0.07040616 0.10382587
## [8] 0.06861966 0.03343134
## 
## $parameters$mean
##         1         2         3         4         5         6         7         8 
##  1.279866  4.526921  6.447820  7.825105  9.031420  9.991102 10.969615 12.053546 
##         9 
## 13.899574 
## 
## $parameters$variance
## $parameters$variance$modelName
## [1] "E"
## 
## $parameters$variance$d
## [1] 1
## 
## $parameters$variance$G
## [1] 9
## 
## $parameters$variance$sigmasq
## [1] 0.08907447
```

```r
root<-sqrt(dim(as.matrix(df4))[1])
cluster<-df4
hist(cluster,breaks=seq(min(cluster),max(cluster),l=root))
```

<img src="{{site.url}}/assets/images/document_files/figure-html/unnamed-chunk-13-1.png" style="display: block; margin: auto;" />

**2. Investigate  the  composition  of  each  cluster  with  regard  to  tissues  (present  the  results in the table).**

```r
uni<-unique(df_part4[,dim(df_part4)[2]])
df4<-as.data.frame(df4)
df4$Tissue <- df_part4[,dim(df_part4)[2]]

table(df4$Tissue, modelGMM$classification)
```

```
##          
##             1   2   3   4   5   6   7   8   9
##   Bladder  54   1   1   3   7  10  11   0   0
##   Colon    39  11  20  30  33  18   2   0   0
##   Marrow  310   2   3   3   0   0   0   0   0
##   Skin      1   0   0   0   4  17  42  34   8
##   Spleen   88   0   0   0   0   0   0   0   0
##   Tongue    1   0   1   0   1  11  34  25  20
```

Marrow and Spleen have the highest impact on the frequency appearance expression gene in the histogram. They are similar. Higher expression values are related to Tongue and Skin. Clustering Colon is problematic due to belonging to many clusters (various expression levels).

Obtained clusters are heterogeneous.
 
