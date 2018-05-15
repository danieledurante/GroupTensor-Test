###############################################################################
###############################################################################
###############################################################################
# GIBBS SAMPLE ################################################################
###############################################################################
###############################################################################
###############################################################################
gibbs_tensor<-function(Y_response,x_predictor,prior,N_sampl,seed){

set.seed(seed)
tensor_data<-Y_response
x_group<-x_predictor

###############################################################################
# SET USEFUL DIMENSIONS #######################################################
###############################################################################
	
#-----------------------------------------------------------------------------#
#Number of categories of x ordered from 1,...,d_x
d_x<-length(unique(x_group))

#-----------------------------------------------------------------------------#
#Number of categories of each y_j ordered from 1,...,d_j
#(Here we consider the simple case in which all the response variables have the 
#same number of categories)
d_y<-length(unique(c(tensor_data)))

#-----------------------------------------------------------------------------#
#Number of categorical response variables
p<-dim(tensor_data)[2]

#-----------------------------------------------------------------------------#
#Sample size
n<-dim(tensor_data)[1]	

#-----------------------------------------------------------------------------#
#Size of each predictor group
n_x<-c(table(x_group))

###############################################################################
# DEFINE HYPERPARAMETERS SETTINGS #############################################
###############################################################################

#-----------------------------------------------------------------------------#
#Upper bound number of mixture components
H<-prior$H

#-----------------------------------------------------------------------------#
#Hyperparameters for the conditional mixing probabilities
a_dir_nu<-rep(1/H,H)

#-----------------------------------------------------------------------------#
#Hyperparameters Dirichlet for the Multinomial kernels
a_dir_y<-prior$a_dir_y

#-----------------------------------------------------------------------------#
#Hyperparameters Dirichlet for \pi_x
a_dir_x<-prior$a_dir_x

#-----------------------------------------------------------------------------#
#Prior probability of the null hypothesis
p_H_O<-prior$p_H_0


################################################################################
# CREATE ALLOCATION MATRICES ###################################################
################################################################################
#-----------------------------------------------------------------------------#
#Marginals of Y in each mixture component
pi_y<-array(0,c(p,d_y,H,N_sampl))

#-----------------------------------------------------------------------------#
#z_i (grouping variable) z[i,]: Group of unit i at a given iterat. i=1,...,n
Group<-matrix(0,n,N_sampl)

#-----------------------------------------------------------------------------#
#Pr(z_i=h | -) for h=1,...,H and i=1,...,n
P_Group<-array(0,c(n,H,N_sampl))

#-----------------------------------------------------------------------------#
#nu with nu[x,h,]=Pr(z=h|x)
nu<-array(0,c(d_x,H,N_sampl))

#-----------------------------------------------------------------------------#
#Sufficient statistic for each step Y[,,h]=Y^{(h)}
Y<-array(0,c(p,d_y,H))

#-----------------------------------------------------------------------------#
#Other useful matrices for implementing the Gibbs sampler
P_Group_Temp<-matrix(0,n,H)
Sel_Group<-rep(0,H)

#-----------------------------------------------------------------------------#
#Multivariate categorical data in dummy form
Tens_dummy<-array(0,c(p,d_y,n))
for (i in 1:n){
for (j in 1:p){
Tens_dummy[j,tensor_data[i,j],i]<-1}}	

#-----------------------------------------------------------------------------#
#Marginals of x
pi_x<-matrix(0,d_x,N_sampl)

#-----------------------------------------------------------------------------#
#Testing indicator. T=0 global independence, T=1 global dependence
T_test<-matrix(0,1,N_sampl)

################################################################################
# INITIALIZE QUANTITIES ########################################################
################################################################################
#-----------------------------------------------------------------------------#
for (h in 1:H){
for (j in 1:p){
pi_y[j,,h,1]<-a_dir_y[j,]}}

#-----------------------------------------------------------------------------#
nu[,,1]<-rep(1/H,H)

#-----------------------------------------------------------------------------#
Group[,1]<-sample(c(1:H),n,replace=TRUE)

################################################################################
# GIBBS ALGORITHM ##############################################################
################################################################################

for (t in 2:N_sampl){
################################################################################
################################################################################
#COMPUTE SIZE OF EACH GROUP
for (h in 1:H){Sel_Group[h]<-sum(Group[,t-1]==h)}

################################################################################
################################################################################
#CREATE MATRICES OF COUNTS
for (h in 1:H){
tensor_temp<-tensor_data[which(Group[,t-1]==h),]	

if (Sel_Group[h]>1){
for (j in 1:p){
for (k in 1:d_y){
Y[j,k,h]<-sum(tensor_temp[,j]==k)
}}} else {
for (j in 1:p){
for (k in 1:d_y){
Y[j,k,h]<-sum(tensor_temp[j]==k)
}}}

}

################################################################################
################################################################################
#UPDATE MARGINALS OF Y IN EACH COMPONENT
for (h in 1:H){
################################################################################
#Sample from full conditional given the data for non-empty groups
if (Sel_Group[h]>0){

#-----------------------------------------------------------------------------#
#Sample pi_y
for (j in 1:p){
pi_y[j,,h,t]<-rdirichlet(1,a_dir_y[j,]+Y[j,,h])
}

################################################################################
#Sample from unconditional for empty groups
} else {
	
#-----------------------------------------------------------------------------#
#Sample pi_y	
for (j in 1:p){
pi_y[j,,h,t]<-rdirichlet(1,a_dir_y[j,])
}}

}

################################################################################
################################################################################
#UPDATE CONDITIONAL PROBABILITIES OF THE LATENT GROUPING VARIABLE
for (i in 1:n){
for (h in 1:H){
P_Group_Temp[i,h]<-sum(log(apply(Tens_dummy[,,i]*pi_y[,,h,t],1,sum)))+log(nu[x_group[i],h,t-1])
}}

for (h in 1:H){
P_Group[,h,t]<-exp(-log(1+apply(exp(P_Group_Temp[,-h]-P_Group_Temp[,h]),1,sum)))}

#-----------------------------------------------------------------------------#
#Sample the component indicator variable for each unit
for (i in 1:n){
Group[i,t]<-sample(c(1:H),1,prob=c(P_Group[i,,t]))}

################################################################################
################################################################################
#UPDATE THE TESTING VARIABLE T=0 independent, T=1 dependent
#useful quantities
Group_matr_test<-matrix(0,n,H)

for (n_i in 1:n){
Group_matr_test[n_i,Group[n_i,t]]<-1}

hyperpar<-lgamma(sum(a_dir_nu))-(sum(lgamma(a_dir_nu)))
data_0<-sum(lgamma(a_dir_nu+apply(Group_matr_test,2,sum)))-(lgamma(sum(a_dir_nu)+n))

data_1<-matrix(0,d_x,1)
for (k in 1:d_x){
data_1[k,]<-sum(lgamma(a_dir_nu+apply(Group_matr_test[x_group==k,],2,sum)))-(lgamma(sum(a_dir_nu)+n_x[k]))}

P_0_given_Data<-hyperpar+data_0
P_1_given_Data<-sum(hyperpar+data_1)

Pr_dep<-1/(1+(p_H_O/(1-p_H_O))*exp(P_0_given_Data-P_1_given_Data))
T_test[,t]<-rbinom(1,1,Pr_dep)


################################################################################
################################################################################
#UPDATE THE GROUP-SPECIFIC MIXING PROBABILITIES
if (T_test[,t]==0){
nu[,,t]<-matrix(rdirichlet(1,a_dir_nu+apply(Group_matr_test,2,sum)),d_x,H,byrow=TRUE)
} else {
for (k in 1:d_x){
nu[k,,t]<-rdirichlet(1,a_dir_nu+apply(Group_matr_test[x_group==k,],2,sum))	
}	
}

################################################################################
################################################################################
#UPDATE THE MARGINAL PROBABILITIES OF THE GROUP VARIABLE X
pi_x[,t]<-rdirichlet(1,a_dir_x+n_x)

###############################################################################
#PRINT: ITERATIONS, CURRENT pr(H_1|data), NUMBER OF NON-EMPTY LATENT CLUSTERS	
if (t%%25 == 0){ 
cat(paste("Iteration:", t, " P_H1:",Pr_dep, " H_fill:", 
    length(unique(Group[,t])),"\n",sep = ""))}
}

###############################################################################
#SAVE THE USEFUL QUANTITIES FOR POSTERIOR INFERENCE AND TESTING
list(nu_post=nu,pi_x_post=pi_x,pi_y_post=pi_y,Group_index=Group,Testing=T_test)

}












###############################################################################
###############################################################################
###############################################################################
# CRAMER MARGINALS ############################################################
###############################################################################
###############################################################################
###############################################################################
cramer_marginals<-function(pi_y_group1,pi_y_group2,pi_y_marginal,pi_x_predictor){


###############################################################################
# DEFINE USEFUL QUANTITIES ####################################################
###############################################################################
pi_y_1<-pi_y_group1
pi_y_2<-pi_y_group2
pi_y_marg<-pi_y_marginal
pi_x<-pi_x_predictor

##############################################################################
# CREATE ALLOCATION MATRIX ###################################################
##############################################################################

cramer_margin<-matrix(,dim(pi_y_marg)[1],dim(pi_y_marg)[3])

##############################################################################
#COMPUTE THE POSTERIOR SAMPLES FOR THE CRAMER V ON THE MARGINALS #############
##############################################################################

for (t in 1:dim(pi_y_marg)[3]){

cramer_margin[,t]<-sqrt(pi_x[1,t]*apply((pi_y_1[,,t]-pi_y_marg[,,t])^2/(pi_y_marg[,,t]),1,sum)+pi_x[2,t]*apply((pi_y_2[,,t]-pi_y_marg[,,t])^2/(pi_y_marg[,,t]),1,sum))

###############################################################################
#PRINT THE NUMBER OF ITERATIONS	
if (t%%25 == 0){ 
cat(paste("Iteration:", t,"\n",sep = ""))}}

###############################################################################
#SAVE THE USEFUL QUANTITIES FOR POSTERIOR INFERENCE AND TESTING
list(pi_y_1=pi_y_1,pi_y_2=pi_y_2,cramer_margin=cramer_margin)

}
















###############################################################################
###############################################################################
###############################################################################
# CRAMER BIVARIATE ############################################################
###############################################################################
###############################################################################
###############################################################################
cramer_bivariates<-function(pi_y_biv_group1,pi_y_biv_group2,pi_y_biv_marginal,pi_x_predictor){


###############################################################################
# DEFINE USEFUL QUANTITIES ####################################################
###############################################################################
pi_y_biv_1<-pi_y_biv_group1
pi_y_biv_2<-pi_y_biv_group2
pi_y_biv<-pi_y_biv_marginal
pi_x<-pi_x_predictor

##############################################################################
# CREATE ALLOCATION MATRIX ###################################################
##############################################################################

cramer_bivariate<-array(0,c(dim(pi_y_biv)[1],dim(pi_y_biv)[1],dim(pi_y_biv)[5]))

##############################################################################
# COMPUTE THE POSTERIOR SAMPLES FOR THE CRAMER V ON THE BIVARIATES ###########
##############################################################################

for (t in 1:dim(pi_y_biv)[5]){
for (j_1 in 2:dim(pi_y_biv)[1]){
for (j_2 in 1:(j_1-1)){		
cramer_bivariate[j_1,j_2,t]<-sqrt(pi_x[1,t]*sum((c(pi_y_biv_1[j_1,j_2,,,t])-c(pi_y_biv[j_1,j_2,,,t]))^2/(c(pi_y_biv[j_1,j_2,,,t])))+pi_x[2,t]*sum((c(pi_y_biv_2[j_1,j_2,,,t])-c(pi_y_biv[j_1,j_2,,,t]))^2/(c(pi_y_biv[j_1,j_2,,,t]))))}}

###############################################################################	
if (t%%25 == 0){ 
cat(paste("Iteration:", t,"\n",sep = ""))}}

###############################################################################
#SAVE THE USEFUL QUANTITIES FOR POSTERIOR INFERENCE AND TESTING
list(pi_y_biv_1=pi_y_biv_1,pi_y_biv_2=pi_y_biv_2,cramer_bivariate=cramer_bivariate)

}



