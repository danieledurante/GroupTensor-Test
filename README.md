# GroupTensor-Test: Bayesian inference on group differences in multivariate categorical data


This repository is associated with the paper [Russo M., Durante D. and Scarpa B. (2018). **Bayesian inference on group differences in multivariate categorical data**. *Computational Statistics & Data Analysis*, 126, 136-149](https://doi.org/10.1016/j.csda.2018.04.010), and aims at providing detailed materials to reproduce the main analyses in the paper. 

The documentation is organized in two main sections described below.  

- [`Simulation-Tutorial.md`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Simulation-Tutorial.md): contains the general guidelines and code to reproduce the simulation studies considered in **Section 4** of the paper. In particular, we provide information on how to **simulate the data**, detailed `R` code to **perform posterior inference and testing** under the proposed dependent mixture of tensor factorizations, and guidelines to **reproduce Figure 2 in the paper**.


- [`Application-Tutorial.md`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Application-Tutorial.md): contains the general guidelines and code to perform the analyses for the [ANES](http://electionstudies.org/studypages/anes_timeseries_2016/anes_timeseries_2016.htm) application in **Section 5** of the paper. In particular, we provide information on how to **download and clean the data**, detailed `R` code to **perform posterior inference and testing** under the proposed dependent mixture of tensor factorizations, and guidelines to **reproduce Figure 3 and 4 in the paper**.


All the above analyses are performed with a **MacBook Pro (OS X El Capitan, version 10.11.6)**, using a `R` version **3.3.2**. In the repository we also made available the file [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R), which contains the core functions to perform the above analyses. In particular:

- `gibbs_tensor()` contains the **Markov Chain Monte Carlo (MCMC) routine to perform posterior computation** for the proposed dependent mixture of tensor factorizations described in Section 2.1 of the paper. The prior distributions are described in Section 3.1. The steps of this function are outlined in **Algorithm 1** in the paper.
- `cramer_marginals()` computes the **posterior samples of the Cramer’s V coefficients** for assessing evidence of group differences in the **marginals**—taking as input the posterior samples produced by the function `gibbs_tensor()`. See Section 2.2 of the paper for a description. 
- `cramer_bivariates()` computes the **posterior samples of the Cramer’s V coefficients** for assessing evidence of group differences in the **bivariates**—taking as input the posterior samples produced by the function `gibbs_tensor()`. See Section 2.2 of the paper for a description. 

All the above functions rely on a **basic and reproducible `R` implementation**, mostly meant to provide a clear understanding of the computational routines and steps associated with the proposed model. Hence, the code may require a non-negligible amount of time in large datasets. **Optimized computational routines relying on C++ coding can be easily considered.** Note also that the current codes in [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R) are written for the case of **two group comparisons** among **categorical random variables having the same number of categories**. Generalizations to include multiple groups and categorical random variables having different numbers of categories, require minor modifications on the functions in the file [`Core_Functions.R`](https://github.com/danieledurante/GroupTensor-Test/blob/master/Core_Functions.R).
