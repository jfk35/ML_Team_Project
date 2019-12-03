setwd("C:\\Users\\jorda\\Workspace\\mban\\Code\\ML_Team_Project")
library(dplyr)
library(tidyr)
library(caret)

crimes <- read.csv("Data/fairClassificationData.csv")

crimes <- crimes %>% filter(is_recid != -1)
crimes <- crimes %>% select(sex, age_cat, race, juv_fel_count, juv_misd_count, juv_other_count, priors_count, days_b_screening_arrest, c_days_from_compas, c_charge_degree,is_recid)
crimes <- crimes %>% mutate(y = ifelse(is_recid==0,-1, 1))

oneHot <- dummyVars(" ~ sex + age_cat + race + c_charge_degree", data = crimes)
encoded <- data.frame(predict(oneHot, newdata = crimes))
outMatrix <- cbind(crimes, encoded)
outMatrix <- outMatrix%>% select(-sex, -age_cat, -race, -is_recid, -c_charge_degree) %>% mutate(groupAssignment = ifelse(race.African.American == 1, 2, 1) )

write.csv(outMatrix, "fairClassificationData.csv")
