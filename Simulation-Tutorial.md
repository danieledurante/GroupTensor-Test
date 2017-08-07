Simulation Studies
================
Daniele Durante

Description
-----------
As described in the [`README.md`](https://github.com/danieledurante/GroupTensor-Test/blob/master/README.md) file, this tutorial contains the general guidelines and code to reproduce the simulation studies considered in **Section 4** of the paper. In particular, we provide information on how to **simulate the data**, detailed `R` code to **perform posterior inference and testing** under the proposed dependent mixture of tensor factorizations, and guidelines to **reproduce Figure 2 in the paper**.

Simulate the Data under the Three Scenarios in the Paper
--------------------------------------
--------------------------------------

We consider three simulation studies to evaluate the empirical performance of the proposed methodologies in several scenarios of interest, characterized by different types of dependence between the multivariate categorical random variable *Y* and the grouping variable *X*. Details and discussion on how the data are simulated can be found in **Section 4** of the paper. **All the steps below should be run in the same order as they are presented**.

Let us first clean the working directory and upload useful `R` libraries.

``` r
rm(list=ls())
library(reshape)
library(gtools)
```

-----------
#### Scenario 1

**DESCRIPTION**: In this first scenario there is no dependence between the multivariate categorical random variable *Y* and the grouping variable *X*. Hence there are no group differences in the marginals, and no group differences in the bivariates. However, to evaluate the flexibility of the proposed model, we define a challenging representation for the probabilistic generative mechanism associated with *Y*. In particular, a subset of the variables are simply generated from independent multinomials with probabilities from a Dirichlet *Dir(10,10,10,10)*. The remaining variables are instead simulated from joint probability mass function assigning probability *0.1* to the configuration *(1,1,..,1)*, probability *0.1* to the configuration *(2,2,...,2)*, probability *0.1* to the configuration *(3,3,...,3)*, probability *0.1* to the configuration *(4,4,...,4)* and probability *0.6* to the remaining configurations in equal proportion. **This generative mechanism of *Y* is the same in the two groups**.

To provide reproducible results we first set a seed.

``` r
set.seed(123)
```

Consistent with the above discussion let us first select the indicators for the variables in *Y* generated from independent multinomials, and those generated from the joint probability mass function.

``` r
#####################################################################
#Indicators for the variables generated from independent multinomials
sel_indep_multinom <- c(2,3,4,6,7,8,9,11,13,14)

#####################################################################
#Indicators for the variables not generated from independent multinomials
sel_joint <- c(1,5,10,12,15)
```

We now create the probability mass functions required to generate the data consistent with the aforementioned generative process. In particular, the **grouping variable** is generated from a binary random variable with equal probabilities.
``` r
pi_X_0 <- c(0.5,0.5)
```

The **variables generated from independent multinomials** simply require their marginal probability mass functions to be simulated. We define these quantities in a *px4* matrix `pi_Y_0_multinom` containing the marginal probabilities for all the *p* variables. Note that, consistent with the generative mechanisms for the variables simulated from the joint probability mass function, their marginals will be equal to *(0.25,0.25,0.25,0.25)*.
``` r
pi_Y_0_multinom <- matrix(0.25,15,4)
for (j in 1:length(sel_indep_multinom)){
pi_Y_0_multinom[sel_indep_multinom[j],] <- rdirichlet(1,c(10,10,10,10))}

pi_Y_0_multinom
```
    ##            [,1]      [,2]      [,3]      [,4]
    ## [1,]  0.2500000 0.2500000 0.2500000 0.2500000
    ## [2,]  0.2163868 0.3727044 0.1379757 0.2729332
    ## [3,]  0.4148120 0.2932725 0.1603739 0.1315416
    ## [4,]  0.3038951 0.2369572 0.2399417 0.2192060
    ## [5,]  0.2500000 0.2500000 0.2500000 0.2500000
    ## [6,]  0.1815423 0.3211108 0.2819822 0.2153648
    ## [7,]  0.2470281 0.2698337 0.2013407 0.2817975
    ## [8,]  0.1919418 0.2392209 0.2418513 0.3269860
    ## [9,]  0.2067295 0.2185640 0.2928735 0.2818330
    ## [10,] 0.2500000 0.2500000 0.2500000 0.2500000
    ## [11,] 0.3040820 0.2508902 0.2803605 0.1646673
    ## [12,] 0.2500000 0.2500000 0.2500000 0.2500000
    ## [13,] 0.2345739 0.2380512 0.3593460 0.1680289
    ## [14,] 0.3063116 0.1977999 0.2625715 0.2333171
    ## [15,] 0.2500000 0.2500000 0.2500000 0.2500000


Finally, the **variables generated from the joint probability mass function** require a specification for all the probabilities of the different configurations. 
``` r
pi_Y_0_joint <- array(0.6/(4^5-4),c(rep(4,5)))
pi_Y_0_joint[1,1,1,1,1] <- 0.1
pi_Y_0_joint[2,2,2,2,2] <- 0.1
pi_Y_0_joint[3,3,3,3,3] <- 0.1
pi_Y_0_joint[4,4,4,4,4] <- 0.1

#Vectorized probability table
vec_pi_Y_0_joint <- as.matrix(melt(pi_Y_0_joint))

head(vec_pi_Y_0_joint)
 ##       X1 X2 X3 X4 X5        value
 ## [1,]  1  1  1  1  1 0.1000000000
 ## [2,]  2  1  1  1  1 0.0005882353
 ## [3,]  3  1  1  1  1 0.0005882353
 ## [4,]  4  1  1  1  1 0.0005882353
 ## [5,]  1  2  1  1  1 0.0005882353
 ## [6,]  2  2  1  1  1 0.0005882353

tail(vec_pi_Y_0_joint)
 ##          X1 X2 X3 X4 X5        value
 ## [1019,]  3  3  4  4  4 0.0005882353
 ## [1020,]  4  3  4  4  4 0.0005882353
 ## [1021,]  1  4  4  4  4 0.0005882353
 ## [1022,]  2  4  4  4  4 0.0005882353
 ## [1023,]  3  4  4  4  4 0.0005882353
 ## [1024,]  4  4  4  4  4 0.1000000000
```

Let us now **generate the data**. First we define the number of variables `p` and the sample size `n`
``` r
n <- 400
p <- 15
```

- The **grouping variable *X*** is generated from a binary random variable. **Note that** the simulated groups are kept the same in all the three simulation scenarios, and therefore will be simulated only once.
``` r
x_group <- sample(c(1:2),n,replace=TRUE,prob=pi_X_0)
```
- As discussed before, to simulate the **multivariate categorical random variable *Y***, part of the variables are generated from independent multinomials, whereas the remaining variables come from the joint probability mass function previously defined.
``` r
tensor_data <- matrix(0,n,p)

for (i in 1:n){
for (j in 1:(length(sel_indep_multinom))){
tensor_data[i,sel_indep_multinom[j]] <- sample(c(1:4),1,replace=TRUE,prob=pi_Y_0_multinom[sel_indep_multinom[j],])}

tensor_data[i,sel_joint] <- c(vec_pi_Y_0_joint[sample(c(1:dim(vec_pi_Y_0_joint [1]),1,replace=TRUE,prob=vec_pi_Y_0_joint[,6]),1:5])}
```

Finally let us **save** the simulated data in **Scenario 1**.
``` r
save(tensor_data,x_group,file="Scenario1.RData")
```

### Scenario 1

**DESCRIPTION**: In this first scenario there is no dependence between the multivariate categorical random variable *Y* and the grouping variable *X*. Hence there are no group differences in the marginals, and no group differences in the bivariates. However, to evaluate the flexibility of the proposed model, we define a challenging representation for the probabilistic generative mechanism associated with *Y*. In particular, a subset of the variables are simply generated from independent multinomials with probabilities from a Dirichlet *Dir(10,10,10,10)*. The remaining variables are instead simulated from joint probability mass function assigning probability *0.1* to the configuration *(1,1,..,1)*, probability *0.1* to the configuration *(2,2,...,2)*, probability *0.1* to the configuration *(3,3,...,3)*, probability *0.1* to the configuration *(4,4,...,4)* and probability *0.6* to the remaining configurations in equal proportion. **This generative mechanism of *Y* is the same in the two groups**.

To provide reproducible results we first set a seed.

``` r
set.seed(123)
```

Consistent with the above discussion let us first select the indicators for the variables in *Y* generated from independent multinomials, and those generated from the joint probability mass function.

``` r
#####################################################################
#Indicators for the variables generated from independent multinomials
sel_indep_multinom <- c(2,3,4,6,7,8,9,11,13,14)

#####################################################################
#Indicators for the variables not generated from independent multinomials
sel_joint <- c(1,5,10,12,15)
```

We now create the probability mass functions required to generate the data consistent with the aforementioned generative process. In particular, the **grouping variable** is generated from a binary random variable with equal probabilities.
``` r
pi_X_0 <- c(0.5,0.5)
```

The **variables generated from independent multinomials** simply require their marginal probability mass functions to be simulated. We define these quantities in a *px4* matrix `pi_Y_0_multinom` containing the marginal probabilities for all the *p* variables. Note that, consistent with the generative mechanisms for the variables simulated from the joint probability mass function, their marginals will be equal to *(0.25,0.25,0.25,0.25)*.
``` r
pi_Y_0_multinom <- matrix(0.25,15,4)
for (j in 1:length(sel_indep_multinom)){
pi_Y_0_multinom[sel_indep_multinom[j],] <- rdirichlet(1,c(10,10,10,10))}

pi_Y_0_multinom
```
    ##            [,1]      [,2]      [,3]      [,4]
    ## [1,]  0.2500000 0.2500000 0.2500000 0.2500000
    ## [2,]  0.2163868 0.3727044 0.1379757 0.2729332
    ## [3,]  0.4148120 0.2932725 0.1603739 0.1315416
    ## [4,]  0.3038951 0.2369572 0.2399417 0.2192060
    ## [5,]  0.2500000 0.2500000 0.2500000 0.2500000
    ## [6,]  0.1815423 0.3211108 0.2819822 0.2153648
    ## [7,]  0.2470281 0.2698337 0.2013407 0.2817975
    ## [8,]  0.1919418 0.2392209 0.2418513 0.3269860
    ## [9,]  0.2067295 0.2185640 0.2928735 0.2818330
    ## [10,] 0.2500000 0.2500000 0.2500000 0.2500000
    ## [11,] 0.3040820 0.2508902 0.2803605 0.1646673
    ## [12,] 0.2500000 0.2500000 0.2500000 0.2500000
    ## [13,] 0.2345739 0.2380512 0.3593460 0.1680289
    ## [14,] 0.3063116 0.1977999 0.2625715 0.2333171
    ## [15,] 0.2500000 0.2500000 0.2500000 0.2500000


Finally, the **variables generated from the joint probability mass function** require a specification for all the probabilities of the different configurations. 
``` r
pi_Y_0_joint <- array(0.6/(4^5-4),c(rep(4,5)))
pi_Y_0_joint[1,1,1,1,1] <- 0.1
pi_Y_0_joint[2,2,2,2,2] <- 0.1
pi_Y_0_joint[3,3,3,3,3] <- 0.1
pi_Y_0_joint[4,4,4,4,4] <- 0.1

#Vectorized probability table
vec_pi_Y_0_joint <- as.matrix(melt(pi_Y_0_joint))

head(vec_pi_Y_0_joint)
 ##       X1 X2 X3 X4 X5        value
 ## [1,]  1  1  1  1  1 0.1000000000
 ## [2,]  2  1  1  1  1 0.0005882353
 ## [3,]  3  1  1  1  1 0.0005882353
 ## [4,]  4  1  1  1  1 0.0005882353
 ## [5,]  1  2  1  1  1 0.0005882353
 ## [6,]  2  2  1  1  1 0.0005882353

tail(vec_pi_Y_0_joint)
 ##          X1 X2 X3 X4 X5        value
 ## [1019,]  3  3  4  4  4 0.0005882353
 ## [1020,]  4  3  4  4  4 0.0005882353
 ## [1021,]  1  4  4  4  4 0.0005882353
 ## [1022,]  2  4  4  4  4 0.0005882353
 ## [1023,]  3  4  4  4  4 0.0005882353
 ## [1024,]  4  4  4  4  4 0.1000000000
```

Let us now **generate the data**. First we define the number of variables `p` and the sample size `n`
``` r
n <- 400
p <- 15
```

- The **grouping variable *X*** is generated from a binary random variable. **Note that** the simulated groups are kept the same in all the three simulation scenarios, and therefore will be simulated only once.
``` r
x_group <- sample(c(1:2),n,replace=TRUE,prob=pi_X_0)
```
- As discussed before, to simulate the **multivariate categorical random variable *Y***, part of the variables are generated from independent multinomials, whereas the remaining variables come from the joint probability mass function previously defined.
``` r
tensor_data <- matrix(0,n,p)

for (i in 1:n){
for (j in 1:(length(sel_indep_multinom))){
tensor_data[i,sel_indep_multinom[j]] <- sample(c(1:4),1,replace=TRUE,prob=pi_Y_0_multinom[sel_indep_multinom[j],])}

tensor_data[i,sel_joint] <- c(vec_pi_Y_0_joint[sample(c(1:dim(vec_pi_Y_0_joint [1]),1,replace=TRUE,prob=vec_pi_Y_0_joint[,6]),1:5])}
```

Finally let us **save** the simulated data in **Scenario 1**.
``` r
save(tensor_data,x_group,file="Scenario1.RData")
```

### Scenario 3

Description

``` r

```


Perform Posterior Inference
--------------------------------------

#### Scenario 1

Description

``` r

```

#### Scenario 2

Description

``` r

```

#### Scenario 3

Description

``` r

```

Reproduce Figure 2 in the Paper
--------------------------------------
