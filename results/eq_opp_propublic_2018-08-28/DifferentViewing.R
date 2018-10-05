
library(ggplot2)
true_name = character()
false_name = character()
for (i in 1: 5) {
  for (j in 1: 5) {
    for (k in 1:5) {
      true_name = c(true_name, paste("FairGPopp*, 0-TNR=", i*0.1 + 0.5, ", 1-TNR=", 
                                     j*0.1 + 0.5, ", TPR=", 
                                     k*0.1 + 0.5, sep = ""))
      false_name = c(false_name, paste("FairGPopp, 0-TNR=", i*0.1 + 0.5, ", 1-TNR=", 
                                       j*0.1 + 0.5, ", TPR=", 
                                       k*0.1 + 0.5, sep = ""))
    }
  }
}
algos = c(true_name,false_name)


TNR0setting = c("0-TNR=0.6","0-TNR=0.7","0-TNR=0.8","0-TNR=0.9","0-TNR=1")
TNR1setting = c("1-TNR=0.6","1-TNR=0.7","1-TNR=0.8","1-TNR=0.9","1-TNR=1")
TPRsetting = c("TPR=0.6", "TPR=0.7","TPR=0.8","TPR=0.9","TPR=1" )

# dataset setting
file_name = list("propublica-recidivism_race","propublica-recidivism_sex")
file_attribute = list( "race", "sex")
N_file = length(file_name)



# df = read.csv("/Users/zc223/PycharmProjects/fairness-comparison/results/eq_opp_propublic_2018-06-14/propublica-recidivism_race_numerical-binsensitive.csv", check.names = FALSE) %>%
df = read.csv(str_c(file_name[1], "_numerical-binsensitive.csv"), check.names=FALSE)%>%
  filter(algorithm %in% true_name, TNR0 %in% TNR0setting, TNR1%in%TNR1setting, TPRset %in% TPRsetting)

var1 = "race-TPRDiff"
var2 = "accuracy"
target_view = "TPRset"

x_var = as.name(var1)
y_var = as.name(var2)
viewing = as.name(target_view)

plot_title = paste("From the view of", target_view)

ggplot(df, aes_q(x=x_var, y=y_var, colour=viewing)) + geom_point() +
  labs(y=var2, x=var1, title=plot_title)