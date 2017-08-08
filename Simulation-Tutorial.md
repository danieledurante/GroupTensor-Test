Simulation Studies
================
Daniele Durante

Description
-----------
As described in the [`README.md`](https://github.com/danieledurante/GroupTensor-Test/blob/master/README.md) file, this tutorial contains the general guidelines and code to reproduce the simulation studies considered in **Section 4** of the paper. In particular, we provide information on how to **simulate the data**, detailed `R` code to **perform posterior inference and testing** under the proposed dependent mixture of tensor factorizations, and guidelines to **reproduce Figure 2 in the paper**.

Simulate the Data under the Three Scenarios in the Paper
--------------------------------------
We consider **three simulation studies** to evaluate the empirical performance of the proposed methodologies in several scenarios of interest, characterized by different types of dependence between the multivariate categorical random variable *Y*, and the grouping variable *X*. Details and discussion on how the data are simulated can be found in **Section 4** of the paper. **All the steps below should be run in the same order as they are presented**.

Let us first clean the working directory and upload useful `R` libraries.

``` r
rm(list=ls())
library(reshape)
library(gtools)
```

-----------
### Scenario 1

**Description**: In this first scenario there is no dependence between the multivariate categorical random variable *Y*, and the grouping variable *X*. Hence, **there are no group differences in the marginals, and no group differences in the bivariates**. However, to evaluate the flexibility of the proposed model, we define a challenging representation for the probabilistic generative mechanism associated with *Y*. In particular, a subset of the variables are simply generated from independent multinomials with probabilities from a Dirichlet *Dir(10,10,10,10)*. The remaining variables are instead simulated from the joint probability mass function assigning probability *0.1* to the configuration *(1,1,..,1)*, probability *0.1* to the configuration *(2,2,...,2)*, probability *0.1* to the configuration *(3,3,...,3)*, probability *0.1* to the configuration *(4,4,...,4)*, and probability *0.6* to the remaining configurations in equal proportion. **This generative mechanism of *Y* is the same in the two groups**.

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

Let us now **generate the data**. First we define the number of variables `p`, and the sample size `n`.

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

tensor_data[i,sel_joint] <- c(vec_pi_Y_0_joint[sample(c(1:dim(vec_pi_Y_0_joint)[1]),1,replace=TRUE,prob=vec_pi_Y_0_joint[,6]),1:5])}
```

Finally let us **save** the simulated data in **Scenario 1**.

``` r
save(tensor_data,x_group,file="Scenario1.RData")
```

-----------
### Scenario 2

**Description**: In this second scenario there is dependence between the multivariate categorical random variable *Y*, and the grouping variable *X*. Here, **there are both group differences in the marginals, and in the bivariates**. The simulation procedure is the same as before for the variables generated from the independent multinomials, with the only exception of variable 2 and 8, whose probability mass function is allowed to change with groups. In fact, in group *X=1* we let their probability mass function be *(0.45,0.45,0.05,0.05)*, whereas in group *X=2* we consider *(0.05,0.05,0.45,0.45)*. The variables previously generated from the joint probability mass function keep the same generative mechanism as in Scenario 1, in group *X=1*. Instead, in group *X=2* these variables are now generated from independent multinomials with probability mass function *(0.25,0.25,0.25,0.25)*. Hence, for these variables we will only observe group differences in the bivariates, but not in the marginals. 

To provide reproducible results we first set a seed.

``` r
set.seed(123)
```

Consistent with the above discussion let us first select the indicators for the variables in *Y* generated from independent multinomials, and those generated from the joint probability mass function.

``` r
############################################################################################
#Indicators for the variables generated from independent multinomials not varying with groups
sel_indep_multinom_1 <- c(3,4,6,7,9,11,13,14)

############################################################################################
#Indicators for the variables generated from independent multinomials varying with groups
sel_indep_multinom_2 <- c(2,8)

#####################################################################
#Indicators for the variables not generated from independent multinomials
sel_joint <- c(1,5,10,12,15)
```

We now create the probability mass functions required to generate the data consistent with the aforementioned generative process. The **grouping variable** has been already generated in Scenario 1, and therefore do not require additional simulations.

The **variables generated from independent multinomials** simply require their marginal probability mass functions to be simulated. These marginals are kept the same as in Scenario 1, with the only exception of variables 2 and 8, whose marginals now vary with groups. Hence, we create now two matrices `pi_Y_0_multinom_1`, and `pi_Y_0_multinom_2`, containing the marginal probabilities for all the *p* variables in the two groups. Note that, consistent with the generative mechanism for the variables simulated from the joint probability mass function, their marginals will be equal to *(0.25,0.25,0.25,0.25)*.

``` r
pi_Y_0_multinom_1 <- pi_Y_0_multinom
pi_Y_0_multinom_2 <- pi_Y_0_multinom

for (j in 1:length(sel_indep_multinom_2)){
pi_Y_0_multinom_1[sel_indep_multinom_2[j],] <- c(0.45,0.45,0.05,0.05)	
pi_Y_0_multinom_2[sel_indep_multinom_2[j],] <- c(0.05,0.05,0.45,0.45)}
```

Finally, the **variables generated from the joint probability mass function** require a specification for all the probabilities of the different configurations. This setting is the same as in Scenario 1.

``` r
pi_Y_0_joint <- array(0.6/(4^5-4),c(rep(4,5)))
pi_Y_0_joint[1,1,1,1,1] <- 0.1
pi_Y_0_joint[2,2,2,2,2] <- 0.1
pi_Y_0_joint[3,3,3,3,3] <- 0.1
pi_Y_0_joint[4,4,4,4,4] <- 0.1

#Vectorized probability table
vec_pi_Y_0_joint <- as.matrix(melt(pi_Y_0_joint))
```

Let us now **generate the data**. First we define the number of variables `p`, and the sample size `n`.

``` r
n <- 400
p <- 15
```

The **grouping variable *X*** has been already simulated in Scenario 1. The **multivariate categorical random variable *Y***, is instead simulated according to the above description for the Scenario 2. In particular:
- Simulate the variables generated from the independent multinomials not varying with groups.

``` r
tensor_data <- matrix(0,n,p)

for (i in 1:n){
for (j in 1:(length(sel_indep_multinom_1))){
tensor_data[i,sel_indep_multinom_1[j]] <- sample(c(1:4),1,replace=TRUE,prob=pi_Y_0_multinom_1[sel_indep_multinom_1[j],])}}
```

- Simulate the variables with indices in `sel_joint` as in Scenario 1 for group *X=1*, and from independent multinomials with probability mass function *(0.25,0.25,0.25,0.25)* in group *X=2*. 

``` r
for (i in 1:n){
if (x_group[i]==1){		
tensor_data[i,sel_joint]<-c(vec_pi_Y_0_joint[sample(c(1:dim(vec_pi_Y_0_joint)[1]),1,replace=TRUE,prob=vec_pi_Y_0_joint[,6]),1:5])} else {
for (j in 1:(length(sel_joint))){
tensor_data[i,sel_joint[j]]<-sample(c(1:4),1,replace=TRUE,prob=pi_Y_0_multinom_1[sel_joint[j],])}}}
```

- Simulate the variables 2 and 8 from independent multinomials with probabilities varying with groups.

``` r
for (i in 1:n){
if (x_group[i]==1){	
for (j in 1:(length(sel_indep_multinom_2))){
tensor_data[i,sel_indep_multinom_2[j]]<-sample(c(1:4),1,replace=TRUE,prob=pi_Y_0_multinom_1[sel_indep_multinom_2[j],])}} 
else {
for (j in 1:(length(sel_indep_multinom_2))){
tensor_data[i,sel_indep_multinom_2[j]]<-sample(c(1:4),1,replace=TRUE,prob=pi_Y_0_multinom_2[sel_indep_multinom_2[j],])}}}
```

Finally let us **save** the simulated data in **Scenario 2**.

``` r
save(tensor_data,x_group,file="Scenario2.RData")
```

-----------
### Scenario 3

**Description**: Also in this final scenario there is dependence between the multivariate categorical random variable *Y*, and the grouping variable *X*. Here, **there are only group differences in the bivariates**. Hence, we maintain the same generative process of Scenario 2, with the only exception that now the marginal probability mass functions for variables 2 and 8 are kept constant across groups—as in Scenario 1.

To provide reproducible results we first set a seed.

``` r
set.seed(123)
```

Consistent with the above discussion let us first select the indicators for the variables in *Y* generated from independent multinomials, and those generated from the joint probability mass function.

``` r
############################################################################################
#Indicators for the variables generated from independent multinomials
sel_indep_multinom<-c(2,3,4,6,7,8,9,11,13,14)

#####################################################################
#Indicators for the variables not generated from independent multinomials
sel_joint <- c(1,5,10,12,15)
```

We now create the probability mass functions required to generate the data consistent with the aforementioned generative process. The **grouping variable** has been already generated in Scenario 1, and therefore do not require additional simulations.

The **variables generated from independent multinomials** simply require their marginal probability mass functions to be simulated. These marginal probabilities are the same as in Scenario 1, and therefore can be found in the matrix `pi_Y_0_multinom`. Note that, consistent with the generative mechanisms for the variables simulated from the joint probability mass function, their marginals will be equal to *(0.25,0.25,0.25,0.25)*.

The **variables generated from the joint probability mass function** require a specification for all the probabilities of the different configurations. This setting is the same as in Scenario 1.

``` r
pi_Y_0_joint <- array(0.6/(4^5-4),c(rep(4,5)))
pi_Y_0_joint[1,1,1,1,1] <- 0.1
pi_Y_0_joint[2,2,2,2,2] <- 0.1
pi_Y_0_joint[3,3,3,3,3] <- 0.1
pi_Y_0_joint[4,4,4,4,4] <- 0.1

#Vectorized probability table
vec_pi_Y_0_joint <- as.matrix(melt(pi_Y_0_joint))
```

Let us now **generate the data**. First we define the number of variables `p`, and the sample size `n`.

``` r
n <- 400
p <- 15
```

The **grouping variable *X*** has been already simulated in Scenario 1. The **multivariate categorical random variable *Y***, is instead simulated according to the above description for the Scenario 3. In particular:

``` r
tensor_data <- matrix(0,n,p)

for (i in 1:n){
if (x_group[i]==1){	
for (j in 1:p){
tensor_data[i,j] <- sample(c(1:4),1,replace=TRUE,prob=pi_Y_0_multinom[j,])}
} else {
for (j in 1:(length(sel_indep_multinom))){
tensor_data[i,sel_indep_multinom[j]]<-sample(c(1:4),1,replace=TRUE,prob=pi_Y_0_multinom[sel_indep_multinom[j],])}
tensor_data[i,sel_joint]<-c(vec_pi_Y_0_joint[sample(c(1:dim(vec_pi_Y_0_joint)[1]),1,replace=TRUE,prob=vec_pi_Y_0_joint[,6]),1:5])}}
```

Finally let us **save** the simulated data in **Scenario 3**.

``` r
save(tensor_data,x_group,file="Scenario3.RData")
```


Perform Posterior Inference
--------------------------------------
Posterior computation under the dependent mixture of tensor factorizations requires the function `gibbs_tensor()` in the source file [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R). This function implements the Gibbs sampler described in **Algorithm 1** of the paper, taking as inputs:

- `Y_response`: an *nxp* matrix containing the values of the *p* categorical variables observed for the *n* units.
- `x_predictor`: a vector with the group memberships for the *n* units.
- `prior`: a list of the hyperparameters discussed in **Section 3.1**, and the number of mixture components *H*.
- `N_sampl`: the number of MCMC samples required.
- `seed`: a seed to ensure reproducibility.

The function `gibbs_tensor()` outputs the posterior samples for the parameters of the model described in **Section 2** of the paper.

Let us perform posterior inference for the three simulations, and compute the posterior samples of the Cramer's V coefficient required for the local tests on the marginals and the bivariates—see Section 2.1 in the paper. To do this we will use the functions `cramer_marginals()` and `cramer_bivariates()` in the file [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R), and described in the [`README.md`](https://github.com/danieledurante/GroupTensor-Test/blob/master/README.md) file.

-----------
### Scenario 1
To perform posterior computation set a working directory containing the file [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R). Once this is done, clean the work space, and upload the source functions [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R) along with useful libraries, and the data.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")

load("Scenario1.RData")
```

Let us now set the hyperparameters as in **Section 3** of the paper.

``` r
prior_model <- list(H=20, 
a_dir_y=matrix(1/length(unique(c(tensor_data))),dim(tensor_data)[2],length(unique(c(tensor_data)))),
a_dir_x=rep(1/length(unique(x_group)),length(unique(x_group))),
p_H_0=0.5)
```

Based on these settings and the data, we can now perform posterior computation using the function  `gibbs_tensor()`, and save the output.

``` r
fit <- gibbs_tensor(Y_response=tensor_data,x_predictor=x_group,prior=prior_model,N_sampl=5000,seed=123)
save(fit, file="Posterior_samples_Scenario1.RData")
```

Once the MCMC samples for the parameters of the statistical model are available, we can compute the posterior samples of the **Cramer's V coefficients for the tests on the marginals**. To do this, clean first the working directory and upload useful data and samples.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")
load("Scenario1.RData")
load("Posterior_samples_Scenario1.RData")
```

As shown in equation (7) in **Section 2.1** of the paper, to obtain the posterior samples of the Cramer's V coefficients for the tests on the marginals, we need to compute several functionals of our model. To do this, run the code below.

``` r
################################################################################
#DEFINE USEFUL DIMENSIONS
N_sampl <- dim(fit$pi_y_post)[4]
H <- dim(fit$pi_y_post)[3]
d_x <- dim(fit$pi_x_post)[1]
d_y <- dim(fit$pi_y_post)[2]
p <- dim(fit$pi_y_post)[1]
n <- dim(tensor_data)[1]

################################################################################
#DEFINE USEFUL QUANTITIES
nu <- fit$nu_post
pi_y <- fit$pi_y_post
pi_x <- fit$pi_x_post

################################################################################
#OBTAIN MARGINALS IN THE TWO GROUPS AND GLOBAL MARGINAL
#-----------------------------------------------------------------------------#
#Marginals two group
pi_y_1 <- array(0,c(p,d_y,N_sampl))
pi_y_2 <- array(0,c(p,d_y,N_sampl))

for (t in 1:N_sampl){
for (h in 1:H){	
pi_y_1[,,t] <- pi_y_1[,,t]+nu[1,h,t]*pi_y[,,h,t]
pi_y_2[,,t] <- pi_y_2[,,t]+nu[2,h,t]*pi_y[,,h,t]	
}	
print(t)}

#-----------------------------------------------------------------------------#
#Global Marginal
pi_y_marg <- array(0,c(p,d_y,N_sampl))
for (t in 1:N_sampl){
pi_y_marg[,,t] <- pi_x[1,t]*pi_y_1[,,t]+pi_x[2,t]*pi_y_2[,,t]}
```

Finally compute the posterior samples of the Cramer's V coefficients for the tests on the marginals, and save them along with other useful quantities.

``` r
test_marginal <- cramer_marginals(pi_y_group1=pi_y_1,pi_y_group2=pi_y_2,pi_y_marginal=pi_y_marg,pi_x_predictor=pi_x)
save(test_marginal,file="Posterior_cramer_marginal_Scenario1.RData")
```

Similar steps are required to obtain the posterior samples of the **Cramer's V coefficients for the tests on the bivariates**. Hence, clean first the working directory and upload useful data and samples.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")
load("Scenario1.RData")
load("Posterior_samples_Scenario1.RData")
```

As shown in equation (8) in **Section 2.1** of the paper, to obtain the posterior samples of the Cramer's V coefficients for the tests on the bivariates, we need to compute several functionals of our model. To do this, run the code below.

``` r
################################################################################
#DEFINE USEFUL DIMENSIONS
N_sampl <- dim(fit$pi_y_post)[4]
H <- dim(fit$pi_y_post)[3]
d_x <- dim(fit$pi_x_post)[1]
d_y <- dim(fit$pi_y_post)[2]
p <- dim(fit$pi_y_post)[1]
n <- dim(tensor_data)[1]

################################################################################
#DEFINE USEFUL QUANTITIES
nu <- fit$nu_post
pi_y <- fit$pi_y_post
pi_x <- fit$pi_x_post

################################################################################
#OBTAIN BIVARIATES IN THE TWO GROUPS AND GLOBAL BIVARIATE
#-----------------------------------------------------------------------------#
#Bivariate two groups
pi_y_biv_1 <- array(0,c(p,p,d_y,d_y,N_sampl))
pi_y_biv_2 <- array(0,c(p,p,d_y,d_y,N_sampl))

for (t in 1:N_sampl){
for (j_1 in 1:p){
for (j_2 in 1:p){
for (h in 1:H){	
pi_y_biv_1[j_1,j_2,,,t] <- pi_y_biv_1[j_1,j_2,,,t]+nu[1,h,t]*(pi_y[j_1,,h,t]%*%t(pi_y[j_2,,h,t]))
pi_y_biv_2[j_1,j_2,,,t] <- pi_y_biv_2[j_1,j_2,,,t]+nu[2,h,t]*(pi_y[j_1,,h,t]%*%t(pi_y[j_2,,h,t]))}}}	
print(t)}

#-----------------------------------------------------------------------------#
#Global Bivariate
pi_y_biv <- array(0,c(p,p,d_y,d_y,N_sampl))

for (t in 1:N_sampl){
pi_y_biv[,,,,t] <- pi_x[1,t]*pi_y_biv_1[,,,,t]+pi_x[2,t]*pi_y_biv_2[,,,,t]}
```

Finally compute the posterior samples of the Cramer's V coefficients for the tests on the bivariates, and save them along with other useful quantities.

``` r
test_bivariate <- cramer_bivariates(pi_y_biv_group1=pi_y_biv_1,pi_y_biv_group2=pi_y_biv_2,pi_y_biv_marginal=pi_y_biv,pi_x_predictor=pi_x)
save(test_bivariate,file="Posterior_cramer_bivariate_Scenario1.RData")
```

-----------
### Scenario 2
Posterior computation, and calculation of the posterior samples for the Cramer's V coefficients for the local tests proceed as in Scenario 1 above. Therefore:

Clean workspace, and upload source functions and data.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")

load("Scenario2.RData")
```

Set the hyperparameters.

``` r
prior_model <- list(H=20, 
a_dir_y=matrix(1/length(unique(c(tensor_data))),dim(tensor_data)[2],length(unique(c(tensor_data)))),
a_dir_x=rep(1/length(unique(x_group)),length(unique(x_group))),
p_H_0=0.5)
```

Perform posterior computation using the function  `gibbs_tensor()`, and save the output.

``` r
fit <- gibbs_tensor(Y_response=tensor_data,x_predictor=x_group,prior=prior_model,N_sampl=5000,seed=123)
save(fit, file="Posterior_samples_Scenario2.RData")
```

Compute the posterior samples of the **Cramer's V coefficients for the tests on the marginals**. To do this, clean first the working directory and upload useful data and samples.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")
load("Scenario2.RData")
load("Posterior_samples_Scenario2.RData")
```

Obtain relevant functionals to compute the **Cramer's V coefficients for the tests on the marginals**.

``` r
################################################################################
#DEFINE USEFUL DIMENSIONS
N_sampl <- dim(fit$pi_y_post)[4]
H <- dim(fit$pi_y_post)[3]
d_x <- dim(fit$pi_x_post)[1]
d_y <- dim(fit$pi_y_post)[2]
p <- dim(fit$pi_y_post)[1]
n <- dim(tensor_data)[1]

################################################################################
#DEFINE USEFUL QUANTITIES
nu <- fit$nu_post
pi_y <- fit$pi_y_post
pi_x <- fit$pi_x_post

################################################################################
#OBTAIN MARGINALS IN THE TWO GROUPS AND GLOBAL MARGINAL
#-----------------------------------------------------------------------------#
#Marginals two group
pi_y_1 <- array(0,c(p,d_y,N_sampl))
pi_y_2 <- array(0,c(p,d_y,N_sampl))

for (t in 1:N_sampl){
for (h in 1:H){	
pi_y_1[,,t] <- pi_y_1[,,t]+nu[1,h,t]*pi_y[,,h,t]
pi_y_2[,,t] <- pi_y_2[,,t]+nu[2,h,t]*pi_y[,,h,t]	
}	
print(t)}

#-----------------------------------------------------------------------------#
#Global Marginal
pi_y_marg <- array(0,c(p,d_y,N_sampl))
for (t in 1:N_sampl){
pi_y_marg[,,t] <- pi_x[1,t]*pi_y_1[,,t]+pi_x[2,t]*pi_y_2[,,t]}
```

Finally compute the posterior samples of the Cramer's V coefficients for the tests on the marginals, and save them along with other useful quantities.

``` r
test_marginal <- cramer_marginals(pi_y_group1=pi_y_1,pi_y_group2=pi_y_2,pi_y_marginal=pi_y_marg,pi_x_predictor=pi_x)
save(test_marginal,file="Posterior_cramer_marginal_Scenario2.RData")
```

Compute the posterior samples of the **Cramer's V coefficients for the tests on the bivariates**. To do this, clean first the working directory and upload useful data and samples.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")
load("Scenario2.RData")
load("Posterior_samples_Scenario2.RData")
```

Obtain relevant functionals to compute the the **Cramer's V coefficients for the tests on the bivariates**.

``` r
################################################################################
#DEFINE USEFUL DIMENSIONS
N_sampl <- dim(fit$pi_y_post)[4]
H <- dim(fit$pi_y_post)[3]
d_x <- dim(fit$pi_x_post)[1]
d_y <- dim(fit$pi_y_post)[2]
p <- dim(fit$pi_y_post)[1]
n <- dim(tensor_data)[1]

################################################################################
#DEFINE USEFUL QUANTITIES
nu <- fit$nu_post
pi_y <- fit$pi_y_post
pi_x <- fit$pi_x_post

################################################################################
#OBTAIN BIVARIATES IN THE TWO GROUPS AND GLOBAL BIVARIATE
#-----------------------------------------------------------------------------#
#Bivariate two groups
pi_y_biv_1 <- array(0,c(p,p,d_y,d_y,N_sampl))
pi_y_biv_2 <- array(0,c(p,p,d_y,d_y,N_sampl))

for (t in 1:N_sampl){
for (j_1 in 1:p){
for (j_2 in 1:p){
for (h in 1:H){	
pi_y_biv_1[j_1,j_2,,,t] <- pi_y_biv_1[j_1,j_2,,,t]+nu[1,h,t]*(pi_y[j_1,,h,t]%*%t(pi_y[j_2,,h,t]))
pi_y_biv_2[j_1,j_2,,,t] <- pi_y_biv_2[j_1,j_2,,,t]+nu[2,h,t]*(pi_y[j_1,,h,t]%*%t(pi_y[j_2,,h,t]))}}}	
print(t)}

#-----------------------------------------------------------------------------#
#Global Bivariate
pi_y_biv <- array(0,c(p,p,d_y,d_y,N_sampl))

for (t in 1:N_sampl){
pi_y_biv[,,,,t] <- pi_x[1,t]*pi_y_biv_1[,,,,t]+pi_x[2,t]*pi_y_biv_2[,,,,t]}
```

Finally compute the posterior samples of the Cramer's V coefficients for the tests on the bivariates, and save them along with other useful quantities.

``` r
test_bivariate <- cramer_bivariates(pi_y_biv_group1=pi_y_biv_1,pi_y_biv_group2=pi_y_biv_2,pi_y_biv_marginal=pi_y_biv,pi_x_predictor=pi_x)
save(test_bivariate,file="Posterior_cramer_bivariate_Scenario2.RData")
```

-----------
### Scenario 3
Posterior computation, and calculation of the posterior samples for the Cramer's V coefficients for the local tests proceed as in Scenario 1 above. Therefore:

Clean workspace, and upload source functions and data.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")

load("Scenario3.RData")
```

Set the hyperparameters.

``` r
prior_model <- list(H=20, 
a_dir_y=matrix(1/length(unique(c(tensor_data))),dim(tensor_data)[2],length(unique(c(tensor_data)))),
a_dir_x=rep(1/length(unique(x_group)),length(unique(x_group))),
p_H_0=0.5)
```

Perform posterior computation using the function  `gibbs_tensor()`, and save the output.

``` r
fit <- gibbs_tensor(Y_response=tensor_data,x_predictor=x_group,prior=prior_model,N_sampl=5000,seed=123)
save(fit, file="Posterior_samples_Scenario3.RData")
```

Compute the posterior samples of the **Cramer's V coefficients for the tests on the marginals**. To do this, clean first the working directory and upload useful data and samples.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")
load("Scenario3.RData")
load("Posterior_samples_Scenario3.RData")
```

Obtain relevant functionals to compute the **Cramer's V coefficients for the tests on the marginals**

``` r
################################################################################
#DEFINE USEFUL DIMENSIONS
N_sampl <- dim(fit$pi_y_post)[4]
H <- dim(fit$pi_y_post)[3]
d_x <- dim(fit$pi_x_post)[1]
d_y <- dim(fit$pi_y_post)[2]
p <- dim(fit$pi_y_post)[1]
n <- dim(tensor_data)[1]

################################################################################
#DEFINE USEFUL QUANTITIES
nu <- fit$nu_post
pi_y <- fit$pi_y_post
pi_x <- fit$pi_x_post

################################################################################
#OBTAIN MARGINALS IN THE TWO GROUPS AND GLOBAL MARGINAL
#-----------------------------------------------------------------------------#
#Marginals two group
pi_y_1 <- array(0,c(p,d_y,N_sampl))
pi_y_2 <- array(0,c(p,d_y,N_sampl))

for (t in 1:N_sampl){
for (h in 1:H){	
pi_y_1[,,t] <- pi_y_1[,,t]+nu[1,h,t]*pi_y[,,h,t]
pi_y_2[,,t] <- pi_y_2[,,t]+nu[2,h,t]*pi_y[,,h,t]	
}	
print(t)}

#-----------------------------------------------------------------------------#
#Global Marginal
pi_y_marg <- array(0,c(p,d_y,N_sampl))
for (t in 1:N_sampl){
pi_y_marg[,,t] <- pi_x[1,t]*pi_y_1[,,t]+pi_x[2,t]*pi_y_2[,,t]}
```

Finally compute the posterior samples of the Cramer's V coefficients for the tests on the marginals, and save them along with other useful quantities.

``` r
test_marginal <- cramer_marginals(pi_y_group1=pi_y_1,pi_y_group2=pi_y_2,pi_y_marginal=pi_y_marg,pi_x_predictor=pi_x)
save(test_marginal,file="Posterior_cramer_marginal_Scenario3.RData")
```

Compute the posterior samples of the **Cramer's V coefficients for the tests on the bivariates**. To do this, clean first the working directory and upload useful data and samples.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")
load("Scenario3.RData")
load("Posterior_samples_Scenario3.RData")
```

Obtain relevant functionals to compute the the **Cramer's V coefficients for the tests on the bivariates**.

``` r
################################################################################
#DEFINE USEFUL DIMENSIONS
N_sampl <- dim(fit$pi_y_post)[4]
H <- dim(fit$pi_y_post)[3]
d_x <- dim(fit$pi_x_post)[1]
d_y <- dim(fit$pi_y_post)[2]
p <- dim(fit$pi_y_post)[1]
n <- dim(tensor_data)[1]

################################################################################
#DEFINE USEFUL QUANTITIES
nu <- fit$nu_post
pi_y <- fit$pi_y_post
pi_x <- fit$pi_x_post

################################################################################
#OBTAIN BIVARIATES IN THE TWO GROUPS AND GLOBAL BIVARIATE
#-----------------------------------------------------------------------------#
#Bivariate two groups
pi_y_biv_1 <- array(0,c(p,p,d_y,d_y,N_sampl))
pi_y_biv_2 <- array(0,c(p,p,d_y,d_y,N_sampl))

for (t in 1:N_sampl){
for (j_1 in 1:p){
for (j_2 in 1:p){
for (h in 1:H){	
pi_y_biv_1[j_1,j_2,,,t] <- pi_y_biv_1[j_1,j_2,,,t]+nu[1,h,t]*(pi_y[j_1,,h,t]%*%t(pi_y[j_2,,h,t]))
pi_y_biv_2[j_1,j_2,,,t] <- pi_y_biv_2[j_1,j_2,,,t]+nu[2,h,t]*(pi_y[j_1,,h,t]%*%t(pi_y[j_2,,h,t]))}}}	
print(t)}

#-----------------------------------------------------------------------------#
#Global Bivariate
pi_y_biv <- array(0,c(p,p,d_y,d_y,N_sampl))

for (t in 1:N_sampl){
pi_y_biv[,,,,t] <- pi_x[1,t]*pi_y_biv_1[,,,,t]+pi_x[2,t]*pi_y_biv_2[,,,,t]}
```

Finally compute the posterior samples of the Cramer's V coefficients for the tests on the bivariates, and save them along with other useful quantities.

``` r
test_bivariate <- cramer_bivariates(pi_y_biv_group1=pi_y_biv_1,pi_y_biv_group2=pi_y_biv_2,pi_y_biv_marginal=pi_y_biv,pi_x_predictor=pi_x)
save(test_bivariate,file="Posterior_cramer_bivariate_Scenario3.RData")
```


Reproduce Figure 2 in the Paper
--------------------------------------
To reproduce Figure 2 in the paper, clean first the working directory and upload useful libraries.

``` r
rm(list=ls())
library(gtools)
library(coda)
library(gdata)
library(reshape)
library(ggplot2)
library(RColorBrewer)
library(gridExtra)
source("Core_Functions.R")
```

Once these preliminary operations are made, load the posterior samples of the Cramer's V coefficients for the local tests previously obtained under the three scenarios.

``` r
#--------------------------------------------------------------------------------
#SCENARIO 1
load("Posterior_cramer_marginal_Scenario1.RData")
load("Posterior_cramer_bivariate_Scenario1.RData")

cramer_marginal1 <- test_marginal$cramer_margin
cramer_bivariate1 <- test_bivariate$cramer_bivariate

#--------------------------------------------------------------------------------
#SCENARIO 2
load("Posterior_cramer_marginal_Scenario2.RData")
load("Posterior_cramer_bivariate_Scenario2.RData")

cramer_marginal2 <- test_marginal$cramer_margin
cramer_bivariate2 <- test_bivariate$cramer_bivariate

#--------------------------------------------------------------------------------
#SCENARIO 3
load("Posterior_cramer_marginal_Scenario3.RData")
load("Posterior_cramer_bivariate_Scenario3.RData")

cramer_marginal3 <- test_marginal$cramer_margin
cramer_bivariate3 <- test_bivariate$cramer_bivariate
```

Let us also set useful quantities, including the **burn-in** of the MCMC chains.

``` r
p <- dim(cramer_marginal1)[1]
MCMC_sample <- dim(cramer_marginal1)[2]
MCMC_burn <- 1001
```

Once this has been done, the `ggplot` code to reproduce the results for the test on the bivariates—representing the lower panels in Figure 2—is provided below.

``` r
matr_1 <- matrix(0,p,p)
lowerTriangle(matr_1) <- lowerTriangle(apply(cramer_bivariate1[,,MCMC_burn:MCMC_sample]>0.2,c(1,2),mean))
matr_1 <- matr_1+t(matr_1)
diag(matr_1) <- NA
matr.dat1 <- melt(matr_1)
matr.dat1 <- matr.dat1[-which(is.na(matr.dat1[, 3])),]
matr.dat1 <- data.frame(matr.dat1)
matr.dat1$X1 <- factor(matr.dat1$X1,levels=rev(c(1:p)))
matr.dat1$X1 <- droplevels(matr.dat1$X1)
matr.dat1$X2 <- factor(matr.dat1$X2)
matr.dat1$flag <- cut(matr.dat1$value,breaks=c(-Inf,0.95,Inf),labels=c("","x"))

matr_2 <- matrix(0,p,p)
lowerTriangle(matr_2) <- lowerTriangle(apply(cramer_bivariate2[,,MCMC_burn:MCMC_sample]>0.2,c(1,2),mean))
matr_2 <- matr_2+t(matr_2)
diag(matr_2) <- NA
matr.dat2 <- melt(matr_2)
matr.dat2 <- matr.dat2[-which(is.na(matr.dat2[, 3])),]
matr.dat2 <- data.frame(matr.dat2)
matr.dat2$X1 <- factor(matr.dat2$X1,levels=rev(c(1:p)))
matr.dat2$X1 <- droplevels(matr.dat2$X1)
matr.dat2$X2 <- factor(matr.dat2$X2)
matr.dat2$flag <- cut(matr.dat2$value,breaks=c(-Inf,0.95,Inf),labels=c("","x"))

matr_3 <- matrix(0,p,p)
lowerTriangle(matr_3) <- lowerTriangle(apply(cramer_bivariate3[,,MCMC_burn:MCMC_sample]>0.2,c(1,2),mean))
matr_3 <- matr_3+t(matr_3)
diag(matr_3) <- NA
matr.dat3 <- melt(matr_3)
matr.dat3 <- matr.dat3[-which(is.na(matr.dat3[, 3])),]
matr.dat3 <- data.frame(matr.dat3)
matr.dat3$X1 <- factor(matr.dat3$X1,levels=rev(c(1:p)))
matr.dat3$X1 <- droplevels(matr.dat3$X1)
matr.dat3$X2 <- factor(matr.dat3$X2)
matr.dat3$flag <- cut(matr.dat3$value,breaks=c(-Inf,0.95,Inf),labels=c("","x"))

matr.dat <-r bind(matr.dat1,matr.dat2,matr.dat3)
matr.dat$g1 <- (c(rep("'SCENARIO 1.  Estimated pr('~rho[jj*minute]>0.2~')'",dim(matr.dat)[1]/3),rep("'SCENARIO 2.  Estimated pr('~rho[jj*minute]>0.2~')'",dim(matr.dat)[1]/3),rep("'SCENARIO 3.  Estimated pr('~rho[jj*minute]>0.2~')'",dim(matr.dat)[1]/3)))

Bivariate <- ggplot(matr.dat, aes(X2, X1, fill = value)) +   geom_tile(color="grey") +  scale_fill_gradientn(colors=brewer.pal(9,"Greys")) +  scale_x_discrete()  +  labs(x = "", y = "") +theme_bw()+facet_wrap(~g1, labeller = label_parsed,ncol=3)+ theme(axis.text.x = element_text(size=6.5),axis.text.y = element_text(size=6.5))+ theme(legend.title=element_blank(),plot.margin=unit(c(0.1,0.1,-0.3,-0.3), "cm"),panel.background = element_rect(fill = brewer.pal(9,"Greys")[2]) )+geom_text(aes(label=flag), color="white", size=3)
```

The `ggplot` code to reproduce the results for the test on the marginals—representing the uppers panels in Figure 2—is instead provided below.

``` r
marg_1 <- data.frame(melt(apply(cramer_marginal1[,MCMC_burn:MCMC_sample]>0.2,1,mean)))
marg_1$var <- c(1:p)
#just for aestetical reasons
marg_1$value <- marg_1$value+0.005

marg_2 <- data.frame(melt(apply(cramer_marginal2[,MCMC_burn:MCMC_sample]>0.2,1,mean)))
marg_2$var <- c(1:p)

marg_3 <- data.frame(melt(apply(cramer_marginal3[,MCMC_burn:MCMC_sample]>0.2,1,mean)))
marg_3$var <- c(1:p)
marg_3$value <- marg_3$value+0.005

matr.dat <- rbind(marg_1,marg_2,marg_3)
matr.dat$g1 <- (c(rep("'SCENARIO 1.  Estimated pr('~rho[j]>0.2~')'",dim(matr.dat)[1]/3),rep("'SCENARIO 2.  Estimated pr('~rho[j]>0.2~')'",dim(matr.dat)[1]/3),rep("'SCENARIO 3.  Estimated pr('~rho[j]>0.2~')'",dim(matr.dat)[1]/3)))

Margin <- ggplot(matr.dat, aes(factor(var),y=value,fill=value)) + geom_bar(stat="identity",colour="black",size=0.3)+  scale_fill_gradientn(colors=brewer.pal(9,"Greys")[2:9]) + labs(x = "", y = "") + theme_bw() +facet_wrap(~g1, labeller = label_parsed,ncol=3) + theme(axis.text.x = element_text(size=6.5),axis.text.y = element_text(size=6.5)) + theme(legend.title=element_blank(),plot.margin=unit(c(0.1,0.1,0.1,-0.3), "cm") ) + geom_hline(yintercept = 0.95,color="gray",,linetype=2)
```

Finally, joining the plots `Bivariate` and `Margin` via

``` r
grid.arrange(Margin,Bivariate,ncol=1)
```
provides the Figure below.

![](https://github.com/danieledurante/GroupTensor-Test/blob/master/Images/simulation.jpg)
