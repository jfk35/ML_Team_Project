
setwd("C:\\Users\\jorda\\Workspace\\mban\\Code\\ML_Team_Project")
library(dplyr)
library(tidyr)
library(ROCR)
library(class)
library(caret)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(randomForest) #for bagging and random forests 
library(gbm) #for boosting


crimesFull = read.csv("cleanedCrimesData.csv")
crimesFull <- crimesFull %>% select(-mixedCommunity, -immigrantCommunity, -X, -otherpercap)

set.seed(657) 
split = createDataPartition(crimesFull$violentcrimesperpopulation, p = 0.65, list = FALSE) 
train = crimesFull[split,] 
test = crimesFull[-split,] 


#Medium ntrees, medium shrinkage, medium interaction depth
n.trees2 = 30000          
shrinkage2 = 0.0005
interaction.depth2 = 6
minobs = 25

# Again, we can manually control the parameters
boost.mod2 = gbm(violentcrimesperpopulation~.,data=train,distribution = "gaussian",n.minobsinnode = minobs, n.trees=n.trees2, shrinkage=shrinkage2, interaction.depth=interaction.depth2)

# Save the influence for the report
influence2 = varImp(boost.mod2, n.trees2)



predTrain <- predict(boost.mod2, train, n.trees= n.trees2)
predTest <- predict(boost.mod2, test, n.trees= n.trees2)

trainSSE <- sum((predTrain- train$violentcrimesperpopulation)^2)
trainSST <- sum((mean(train$violentcrimesperpopulation)- train$violentcrimesperpopulation)^2)
trainOSR2 <- 1-trainSSE/trainSST

testSSE <- sum((predTest- test$violentcrimesperpopulation)^2)
testSST <- sum((mean(train$violentcrimesperpopulation)- test$violentcrimesperpopulation)^2)
testOSR2 <- 1-testSSE/testSST





#Use the recitation code to train the out-of-bag predictions, setting ntree and nodesize specifically
train.rf.oob <- train(x = train %>% select(-violentcrimesperpopulation),
                      y = train$violentcrimesperpopulation,
                      method="rf",
                      ntree=80,
                      nodesize=25,
                      tuneGrid=data.frame(mtry=1:40),          # use default nodesize & ntree
                      trControl=trainControl(method="oob"))
# Plot Results as a Line Graph
ggplot(train.rf.oob$results, aes(x=mtry, y=Rsquared)) +
  geom_point(size=5) +
  theme_bw() +
  xlab("Number of variables per split") +
  ylab("Out-of-bag R2") +
  ggtitle("Cross-Validation: OOB R2 vs Number of Vars Per Split")+
  scale_x_continuous(breaks=1:80, name="mtry") +
  theme(axis.title=element_text(size=18), axis.text=element_text(size=12))

best.mtry <- train.rf.oob$bestTune[[1]]
print(best.mtry)
rf.cv = randomForest(violentcrimesperpopulation~., data=train, ntree=80,mtry=best.mtry,nodesize=25)    
importance.rf <- data.frame(imp=importance(rf.cv))

predTrainRF <- predict(rf.cv, train, n.trees= n.trees2)
predTestRF <- predict(rf.cv, test, n.trees= n.trees2)

trainSSERF <- sum((predTrainRF- train$violentcrimesperpopulation)^2)
trainSSTRF <- sum((mean(train$violentcrimesperpopulation)- train$violentcrimesperpopulation)^2)
trainOSR2RF <- 1-trainSSERF/trainSSTRF



testSSERF <- sum((predTestRF- test$violentcrimesperpopulation)^2)
testSSTRF <- sum((mean(train$violentcrimesperpopulation)- test$violentcrimesperpopulation)^2)
testOSR2RF <- 1-testSSERF/testSSTRF




