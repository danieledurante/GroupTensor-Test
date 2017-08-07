Application to the ANES Dataset
================
Daniele Durante

Description
-----------
As described in the [`README.md`](https://github.com/danieledurante/GroupTensor-Test/blob/master/README.md) file, this tutorial contains general guidelines and code to perform the analyses for the [ANES](http://electionstudies.org/studypages/anes_timeseries_2016/anes_timeseries_2016.htm) application in **Section 5** of the paper. In particular, we provide information on how to **download and clean the data**, detailed **R code to perform posterior inference and testing** under the proposed dependent mixture of tensor factorizations, and **guidelines to reproduce Figure 3 and 4** in the paper.

Upload and Clean the Data from the ANES
--------------------------------------
As discussed in **Section 5** of the paper, we apply the proposed methodologies to a subset of the 2016 polls data from the American National Election Studies ([ANES](http://electionstudies.org/studypages/anes_timeseries_2016/anes_timeseries_2016.htm)).  

> Our aim is to **understand if the voters feelings and opinions for Hillary Clinton and Donald Trump, change with their preference for Hillary Clinton or Bernie Sanders expressed in the 2016 Democratic Presidential primaries**. In particular, the multivariate categorical data under analysis comprise five different feelings along with five specific personality traits for each of the two Presidential candidates, thereby providing a total of *p=20* categorical opinions collected for each unit. There are *567* voters who expressed their preference for Hillary Clinton, and *386* who chose Bernie Sanders voters  in the 2016  primaries. 


**We cannot redistribute the data**, but they can be downloaded at the [Data Center](http://electionstudies.org/studypages/download/datacenter_all_NoData.html) of the ANES. Download will require a registration but is completely free. We focus on the `Stata` version of the dataset called **2016 Time Series Study**. The download provides a file `anes_timeseries_2016_Stata12.dta` which contains a large dataset, along with a file `anes_timeseries_2016_varlist.csv` with the names of the observed variables. Additional information can be found in the [User's and Codebook](http://electionstudies.org/studypages/anes_timeseries_2016/anes_timeseries_2016_userguidecodebook.pdf) file, and in the [Pre-Election Questionnaire](http://electionstudies.org/studypages/anes_timeseries_2016/anes_timeseries_2016_qnaire_pre.pdf).

To **clean the dataset**, first set the working directory where the downloaded files `anes_timeseries_2016_Stata12.dta` and `anes_timeseries_2016_varlist.csv` are placed. Once this has been done, **clean the workspace, and load the data along with useful libraries**.

``` r
rm(list=ls())

library(foreign)
library(lattice)
library(gridExtra)

dati <- read.dta('anes_timeseries_2016_Stata12.dta')
variable_names <- read.csv('anes_timeseries_2016_varlist.csv',stringsAsFactors = FALSE)
names(dati) <- variable_names$`X.1`
```
Note that when uploading the data, a warning appears. However this warning is related to a variable which is not considered in this application.

The dataframe `dati` contain a massive amount of information and observed items. In our motivating application, the variable defining the **group membership—i.e. the voting preferences at the Presidential primaries—is the variable 49**. The multivariate categorical data on the **voters feelings and opinions for Hillary Clinton and Donald Trump, are instead found in the variables from 160 to 169, and 217 to 226**. Let us create a dataset having only these variables.
   
``` r
group_variable <- 49
selected_variables <- c(160:169,217:226)

response <- dati[,group_variable]
tensor_dat <- dati[,selected_variables]
```

**For a subset of the voters we observe categories `-9. Refused` or `-8. Don't know (FTF only)`** in some variables measuring feelings and opinions for Hillary Clinton and Donald Trump. **We hold these voters out from our analysis** since they do not provide information of their evaluations and preferences.

``` r
for(j in 1:NCOL(tensor_dat))
{tensor_dat[,j] <- as.character( tensor_dat[,j])}

response <- as.character(response)

sel = c()
for( i in 1:NROW(tensor_dat))
{if(all(tensor_dat[i,] != '-9. Refused') &
     all(tensor_dat[i,] != "-8. Don't know (FTF only)"))
  {sel <- c(sel,i)}}

tensor_dat <- tensor_dat[sel,]
response <- response[sel]
```

The grouping variable denoting the voters preferences at the Presidential primaries, contain also information on the Republican primaries. Since our focus is only on the Democratic primaries, we **select only the voters who chose either Hillary Clinton or Bernie Sanders.**

``` r
sel_resp <- which(response=="1. Hillary Clinton" | response=="2. Bernie Sanders")
tensor_dat <- tensor_dat[sel_resp,]
response <- response[sel_resp]
```

Finally, let us **transform the above variables into factors, and save them**.

``` r
for(j in 1:NCOL(tensor_dat))
{tensor_dat[,j] <- factor(tensor_dat[,j])}
response <- factor(response)
table(response)
```
    ##      response   
    ##      1. Hillary Clinton  2. Bernie Sanders 
    ##                     567                386
    
    
``` r
save(tensor_dat,response,file = 'Political.RData')
```

Perform Posterior Inference
--------------------------------------
Posterior computation under the dependent mixture of tensor factorizations requires the function `gibbs_tensor()` in the source file [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R). This function implements the Gibbs sampler described in **Algorithm 1** of the paper, taking as inputs:
- `Y_response`: an *nxp* matrix containing the values of the *p* categorical variables observed for the *n* units.
- `x_predictor`: a vector with the group memberships for the *n* units.
- `prior_model`: a list of the hyperparameters discussed in **Section 3.1** and the number of mixture components *H*.
- `N_sampl`: the number of MCMC samples required.
- `seed`: a seed to ensure reproducibility.
The function `gibbs_tensor()` outputs the posterior samples for the parameters of the model described in **Section 2** of the paper.

Let us perform posterior inference for the application, and compute the posterior samples of the Cramer's V coefficient required for the local tests on the marginals and the bivariates—see Section 2.1 in the paper. To do this we will use the functions `cramer_marginals()` and `cramer_bivariates()` in the file [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R), and described in the [`README.md`](https://github.com/danieledurante/GroupTensor-Test/blob/master/README.md) file.


To perform posterior computation set a working directory containing the file [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R). Once this is done, clean the work space, and upload the source functions [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R) along with useful libraries, and the data.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")

load("Political.RData")
tensor_data < -matrix(0,dim(tensor_dat)[1],dim(tensor_dat)[2])
for (j in 1:dim(tensor_dat)[2]){
tensor_data[,j] <- c((tensor_dat[,j]))}
x_group <- c(response)
```

Let us now set the hyperparameters as in **Section 5** of the paper.

``` r
prior_model <- list(H=20, 
a_dir_y=matrix(1/length(unique(c(tensor_data))),dim(tensor_data)[2],length(unique(c(tensor_data)))),
a_dir_x=rep(1/length(unique(x_group)),length(unique(x_group))),
p_H_0=0.5)
```

Based on these settings and the data, we can now perform posterior computation using the function  `gibbs_tensor()`, and save the output.

``` r
fit <- gibbs_tensor(Y_response=tensor_data,x_predictor=x_group,prior=prior_model,N_sampl=5000,seed=123)
save(fit, file="Posterior_samples_Application.RData")
```

Once the MCMC samples for the parameters of the statistical model are available, we can compute the posterior samples of the **Cramer's V coefficients for the tests on the marginals**. To do this, clean first the working directory and upload useful data and samples.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")
load("Political.RData")
load("Posterior_samples_Application.RData")
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
pi_y_2[,,t] <- pi_y_2[,,t]+nu[2,h,t]*pi_y[,,h,t]}	
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
save(test_marginal,file="Posterior_cramer_marginal_Application.RData")
```

Similar steps are required to obtain the posterior samples of the **Cramer's V coefficients for the tests on the bivariates**. Hence, clean first the working directory and upload useful data and samples.

``` r
rm(list=ls())
library(gtools)
source("Core_Functions.R")
load("Political.RData")
load("Posterior_samples_Application.RData")
```

As shown in equation (8) in **Section 2.1** of the paper, to obtain the posterior samples of the Cramer's V coefficients for the tests on the bivariates, we need to compute several functionals of our model. To do this, run the code below.

``` r
#DEFINE USEFUL DIMENSIONS
N_sampl <- dim(fit$pi_y_post)[4]
H <- dim(fit$pi_y_post)[3]
d_x <- dim(fit$pi_x_post)[1]
d_y <- dim(fit$pi_y_post)[2]
p <- dim(fit$pi_y_post)[1]

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
pi_y_biv_2[j_1,j_2,,,t] <- pi_y_biv_2[j_1,j_2,,,t]+nu[2,h,t]*(pi_y[j_1,,h,t]%*%t(pi_y[j_2,,h,t]))}}}}

#-----------------------------------------------------------------------------#
#Global Bivariate
pi_y_biv <- array(0,c(p,p,d_y,d_y,N_sampl))

for (t in 1:N_sampl){
pi_y_biv[,,,,t] <- pi_x[1,t]*pi_y_biv_1[,,,,t]+pi_x[2,t]*pi_y_biv_2[,,,,t]}
```

Finally compute the posterior samples of the Cramer's V coefficients for the tests on the bivariates, and save them along with other useful quantities.

``` r
test_bivariate <- cramer_bivariates(pi_y_biv_group1=pi_y_biv_1,pi_y_biv_group2=pi_y_biv_2,pi_y_biv_marginal=pi_y_biv,pi_x_predictor=pi_x)
save(test_bivariate,file="Posterior_cramer_bivariate_Application.RData")
```

Reproduce Figures 3 and 4 in the Paper
--------------------------------------
