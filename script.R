# Introduction to Probabilistic Programming with Stan
# StanCon 2024

# install.packages("remotes")
# remotes::install_github("stan-dev/cmdstanr")
library(cmdstanr)
# remotes::install_github("stan-dev/posterior")
library(posterior)

# utility function for sampling from cmdstanr models

get_samples <- function(model_code,data=list(),seed=123,chains=4) {
    f <- write_stan_file(model_code)
    model_x <- cmdstan_model(f)
    samples_1 <- model_x$sample(
        data = data,
        seed = seed,
        chains = chains,
        parallel_chains = chains,
        refresh = 0
    )
    pr <- as_draws_rvars( samples_1$draws() )
    p <- list()
    for ( i in 1:length(pr) )
        p[[ names(pr)[i] ]] <- draws_of( pr[[i]] )
    return(p)
}

# 1 - Variables, constraints, distributions and relationships

model_code_1 <- "
parameters{
    real< lower=5 , upper=25 > duration;
}
"

f <- write_stan_file(model_code_1)
model_1 <- cmdstan_model(f)

samples_1 <- model_1$sample(
    data = list(),
    seed = 123,
    chains = 4,
    parallel_chains = 4
)

samples_1$summary()

post <- as_draws_df( samples_1$draws() )

plot(post$duration,xlab="sample",ylab="duration")

plot(density(post$duration,adj=0.1),xlab="duration",main="")

## add distribution - "The average duration is 10 minutes"
model_code_2 <- "
parameters{
    real< lower=0 > duration;
}
model{
    duration ~ exponential(1.0/10.0);
}
"

post2 <- get_samples(model_code_2)

plot(post2$duration,xlab="sample",ylab="duration")

plot(density(post2$duration,adj=0.1),xlab="duration",main="")

mean(post2$duration)

## add data - duration observed N times, what is the average?
model_code_3 <- "
data{
    int N;
    array[N] real<lower=0> y; // observed durations
}
parameters{
    real< lower=0 > avg_duration;
}
model{
    y ~ exponential(1.0/avg_duration);
}
"

N <- 10
set.seed(123)
post3 <- get_samples(
    model_code_3,
    data=list(N=N,y=rexp(N,1/10))
)

mean(post3$avg_duration)
median(post3$avg_duration)
th <- 100
x <- post3$avg_duration
plot(density(x[x<th]),xlab="avg duration",main="",lwd=2,col=4,xlim=c(0,100))

## relationship - duration associated with location x
model_code_4 <- "
data{
    int N;                    // number of observations
    int M;                    // number of locations
    array[N] real<lower=0> y; // observed durations
    array[N] int x;           // location (categorical)
}
parameters{
    array[M] real<lower=0> avg_duration;
}
model{
    y ~ exponential(avg_duration[x]^(-1));
}
"

N <- 20
set.seed(123)
avg_durations <- c(10,20)
x <- sample(1:2,size=N,replace=TRUE)
y_sim <- rexp(N,1/avg_durations[x])
post4 <- get_samples(
    model_code_4,
    data=list(N=N,M=2,y=y_sim,x=x)
)

x <- post4$avg_duration
plot(density(x[,1]),xlab="avg duration",main="",lwd=2,col=4,xlim=c(0,100))
rethinking::dens(x[,2],add=TRUE,lwd=2)


#############################################################
# 2 - Generative models

## cat adoptions data
## library(rethinking)
## data(AustinCats)
## d <- AustinCats

## read data from github, so you don't have to install rethinking package
urlfile <- 'https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/AustinCats.csv'
d <- read.csv2(urlfile)

dat <- list(
    N = nrow(d),
    days = d$days_to_event,
    adopted = ifelse( d$out_event=="Adoption" , 1 , 0 ),
    color = ifelse( d$color=="Black" , 1 , 2 ) )

## plot individual cats as lines

n <- 100
idx <- sample(1:dat$N,size=n)
ymax <- max(dat$days[idx])
plot(NULL,xlim=c(0,ymax),ylim=c(1,n),xlab="days observed",ylab="cat")
for ( i in 1:n ) { 
    j <- idx[i]
    cat_color <- ifelse( dat$color[j]==1 , "black" , "orange" )
    lines( c(0,dat$days[j]) , c(i,i) , lwd=4 , col=cat_color )
    if ( dat$adopted[j]==1 ) points( dat$days[j] , i , pch=16 , cex=1.5 , col=cat_color )
}

## how should we model these data? think about how they were generated
## there is process and observation
## we start with process of adoption and then add observation (censoring)

sim_cats1 <- function(n=1e3,p=c(0.1,0.2)) {
    color <- sample(c(1,2),size=n,replace=TRUE)
    days <- rgeom( n , p[color] ) + 1
    return(list(N=n,days=days,color=color,adopted=rep(1,n)))
}

cat_code1 <- "
// observed adoptions only
data{
    int N;
    array[N] int adopted; // 1/0 indicator
    array[N] int days;       // days until event
    array[N] int color;   // 1=black, 2=other
}
parameters{
    vector<lower=0,upper=1>[2] p;
}
model{
    p ~ beta(1,10);
    for ( i in 1:N ) {
        real P = p[color[i]];
        if ( adopted[i]==1 ) {
            target += log( (1-P)^(days[i]-1) * P );
        } else {
            // something here
        }
    }
}
"

cat_code1b <- "
// observed adoptions only, version using target for p ~ beta(1,10)
data{
    int N;
    array[N] int adopted; // 1/0 indicator
    array[N] int days;       // days until event
    array[N] int color;   // 1=black, 2=other
}
parameters{
    vector<lower=0,upper=1>[2] p;
}
model{
    target += beta_lpdf(p|1,10);
    for ( i in 1:N ) {
        real P = p[color[i]];
        if ( adopted[i]==1 ) {
            target += log( (1-P)^(days[i]-1) * P );
        } else {
            // something here
        }
    }
}
"

## test
p <- c(0.1,0.15)
sim_dat <- sim_cats1(p=p)
postx <- get_samples(cat_code1,data=sim_dat)

dens( postx$p[,1] , lwd=3 , xlab="probability of adoption" , xlim=c(0.07,0.2)  )
dens( postx$p[,2] , add=TRUE , lwd=3 , col="orange" )
abline(v=p[1],lwd=2)
abline(v=p[2],lwd=2,col="orange")

## real sample
postx <- get_samples(cat_code1,data=dat)

dens( postx$p[,1] , lwd=3 , xlab="probability of adoption" , xlim=c(0.019,0.03) , ylim=c(0,1400) )
dens( postx$p[,2] , add=TRUE , lwd=3 , col="orange" )

dens( 1/postx$p[,1] , lwd=3 , xlab="expected days until adoption" , xlim=c(35,64) , ylim=c(0,1) )
dens( 1/postx$p[,2] , add=TRUE , lwd=3 , col="orange" )

diff <- 1/postx$p[,1] - 1/postx$p[,2]
quantile( diff )
dens( diff , xlab="extra days for black cat" , lwd=3 )

quantile(postx$p[,1])
quantile(postx$p[,2])

## now add observation (censoring)

sim_cats2 <- function(n=1e3,p=c(0.01,0.02),cens=50) {
    color <- sample(c(1,2),size=n,replace=TRUE)
    days <- rgeom( n , p[color] ) + 1
    adopted <- ifelse( days < cens , 1 , 0 )
    days <- ifelse( adopted==1 , days , cens )
    return(list(N=n,days=days,color=color,adopted=adopted))
}

cat_code2 <- "
// all events, including censored
data{
    int N;
    array[N] int adopted; // 1/0 indicator
    array[N] int days;    // days until event
    array[N] int color;   // 1=black, 2=other
}
parameters{
    vector<lower=0,upper=1>[2] p;
}
model{
    p ~ beta(1,10);
    for ( i in 1:N ) {
        real P = p[color[i]];
        if ( adopted[i]==1 ) {
            target += log( (1-P)^(days[i]-1) * P );
        } else {
            target += log( (1-P)^days[i] );
        }
    }
}
"

## test
sim_dat <- sim_cats2()
postx <- get_samples(cat_code2,data=sim_dat)

dens( postx$p[,1] , lwd=3 , xlab="probability of adoption" , xlim=c(0.007,0.025) , ylim=c(0,600) )
dens( postx$p[,2] , add=TRUE , lwd=3 , col="orange" )
abline(v=0.01,lwd=2); abline(v=0.02,lwd=2,col="orange")

## test new data with censoring on previous model
sim_dat <- sim_cats2()
postx <- get_samples(cat_code1,data=sim_dat)

dens( postx$p[,1] , lwd=3 , xlab="probability of adoption" , xlim=c(0.007,0.06) , ylim=c(0,170) )
dens( postx$p[,2] , add=TRUE , lwd=3 , col="orange" )
abline(v=0.01,lwd=2); abline(v=0.02,lwd=2,col="orange")

## real data
postx <- get_samples(cat_code2,data=dat)

quantile( postx$p[,1] )
quantile( postx$p[,2] )

quantile( 1 / post$lambda[,1] )
quantile( 1 / post$lambda[,2] )

dens( postx$p[,1] , lwd=3 , xlab="probability of adoption" , xlim=c(0.015,0.023) , ylim=c(0,1900) )
dens( postx$p[,2] , add=TRUE , lwd=3 , col="orange" )

dens( 1/postx$p[,1] , lwd=3 , xlab="expected days until adoption" , xlim=c(35,63) , ylim=c(0,1) )
dens( 1/postx$p[,2] , add=TRUE , lwd=3 , col="orange" )


dens( 1/postx$p[,1] , lwd=3 , xlab="expected days until adoption" , xlim=c(45,63) , ylim=c(0,1) )
dens( 1/postx$p[,2] , add=TRUE , lwd=3 , col="orange" )


#############################################################
# 3 - Workflow

## prior predictive distributuon
## repeatedly sample from prior, simulate observations

## prior samples
n <- 100
p1 <- rbeta(n,1,10)
p2 <- rbeta(n,1,10)

sim_cats2 <- function(n=1e3,p=c(0.01,0.02),cens=50) {
    color <- sample(c(1,2),size=n,replace=TRUE)
    days <- rgeom( n , p[color] ) + 1
    adopted <- ifelse( days < cens , 1 , 0 )
    days <- ifelse( adopted==1 , days , cens )
    return(list(N=n,days=days,color=color,adopted=adopted))
}

prior_days <- sapply( 1:n , function(i) sim_cats2(1,p=c(p1[i],p2[i]))$days )

plot( prior_days , xlab="simulated cat" , ylab="days" , pch=ifelse(prior_days==50,1,16) )


## posterior predictive distributuon
## sample from posterior, simulate observations
## problem with this example: need to impute censored values
## so we'll simulate kaplan-meier curves to compare to empirical curve

postx <- get_samples(cat_code2,data=dat)

## plot empirical K-M curves 

library(survival)
sfit <- survfit(Surv(days, adopted) ~ color, data = dat)
plot( sfit, lty = 1 , lwd=0.1 , col=c("black","orange") , xlim=c(0,90) , xlab="days" , ylab="proportion un-adopted" ) 

## simulate and draw

cols <- c( col.alpha("black",0.5) , col.alpha("orange",0.5) )
n <- 12
for ( i in 1:n ) {
    days_rep <- sim_cats2(n=1e3,p=postx$p[i,],cens=200)
    xfit <- survfit(Surv(days, adopted) ~ color, data = days_rep)
    lines( xfit , lwd=1 , col=cols )
}

## overlay empirical curves
lines( sfit , lwd=5 , col=c('white','white') )
lines( sfit , lwd=3 , col=c('black','orange') )
