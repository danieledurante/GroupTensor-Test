Application to the ANES Dataset
================
Daniele Durante

Description
-----------
As described in the [`README.md`](https://github.com/danieledurante/GroupTensor-Test/blob/master/README.md) file, this tutorial contains general guidelines and code to perform the analyses for the [ANES](http://electionstudies.org/studypages/anes_timeseries_2016/anes_timeseries_2016.htm) application in **Section 5** of the paper. In particular, we provide information on how to **download and clean the data**, detailed **R code to perform posterior inference and testing** under the proposed dependent mixture of tensor factorizations, and **guidelines to reproduce Figure 3 and 4** in the paper.

Upload and Clean the Data from the ANES
--------------------------------------
As discussed in **Section 5** of the paper, we apply the proposed methodologies to a subset of the 2016 polls data from the American National Election Studies ([ANES](http://electionstudies.org/studypages/anes_timeseries_2016/anes_timeseries_2016.htm)).  

> Our aim is to **understand if the voters feelings and opinions for Hillary Clinton and Donald Trump, change with their preference for Hillary Clinton or Bernie Sanders expressed in the 2016 Democratic Presidential primaries**. In particular, the multivariate categorical data under analysis comprise five different feelings along with five specific personality traits for each of the two Presidential candidates, thereby providing a total of *p=20* categorical opinions collected for each unit. There are *567* voters who expressed their preference for Hillary Clinton, and $386$ who chose Bernie Sanders voters  in the 2016  primaries. 


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

save(tensor_dat,response,file = 'Political.RData')
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

Reproduce Figures 3 and 4 in the Paper
--------------------------------------
