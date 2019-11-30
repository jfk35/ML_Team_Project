
setwd("C:\\Users\\jorda\\Workspace\\mban\\Code\\ML_Team_Project")
library(dplyr)
library(tidyr)


crimes <- read.csv("Data/communities.data")
crimes <- crimes %>% select(-state, -county, -community, -fold, -communityname, -lemasswornft, -lemasswftperpop, -lemasgangunitdeployed, -lemasswftfieldops, -lemasswftfieldperpop, -lemastotalreq, -lemastotreqperpop, -lemaspctofficedrugunits, -lemaspctpoliceonpatrol, -policecars, -policeperpop, -policeoperbudget, -policebudgetperpop, -policeaveotworked, -numkindsdrugsseized, -officersassgndrugunits, -pctpoliceminority, -pctpoliceasian, -pctpolicewhite, -pctpoliceblack, -racialmatchcommpolice, -policereqperofficer, -pctpolicehisp)

meanasian = mean(crimes$racepctasian)
meanblack = mean(crimes$racepctblack)
meanhisp = mean(crimes$racepcthisp)
meanimmig = mean(crimes$pctpoprecentimmig)


crimes <- crimes %>% mutate(mixedCommunity = ifelse((racepcthisp > meanhisp) | (racepctblack > meanblack), 2, 1), immigrantCommunity = ifelse(pctpoprecentimmig > meanimmig, 2, 1))

write.csv(crimes, "cleanedCrimesData.csv")

