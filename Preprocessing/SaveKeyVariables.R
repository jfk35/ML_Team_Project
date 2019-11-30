setwd("C:\\Users\\jorda\\Workspace\\mban\\Code\\ML_Team_Project")

crimesFull <- read.csv("cleanedCrimesData.csv")
crimesFull <- crimesFull %>% select(pctilleg, pctkids2par, numilleg, racepctwhite, racepctblack, pctpersdensehouse, pctfam2par, pctvacantboarded, numstreet, pcthouseless3br, malepctdivorce, mixedCommunity, immigrantCommunity, violentcrimesperpopulation)

write.csv(crimesFull, "fairRegressionData.csv")

