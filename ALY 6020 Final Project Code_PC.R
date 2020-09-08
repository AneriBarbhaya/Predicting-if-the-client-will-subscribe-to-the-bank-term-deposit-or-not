#STEP 1- collecting & importing the data

#{r loading dataset and settling data in proper format}
bank <- read.csv("bank-additional-full.csv", sep = ";", stringsAsFactors = T)
View(bank)

#{r reviewing structure}
str(bank)
#{r checking for missing values}
library(DataExplorer)
plot_missing(bank)

#{summaryof dataset r}
summary(bank)

library(e1071)
library(gmodels)

##### Naive Bayes classification.
#STEP 2- exploring and preparing the data
#checking proportion of each category of y
table(bank$y)

#Data Cleaning
#colSums will calculate number of rows each column has with the value as NA
colSums(is.na(bank) | bank == "" )
#we are removing the fields with value NA 
bank <- na.omit(bank)

set.seed(123)
# Shuffle the dataset, call the result shuffled
n=nrow(bank)
shuffled <- bank[sample(n),]
#creating training and test datasets (70% training 30% testing)
bank_train<-shuffled[1:round(0.7 * n),]
bank_test<-shuffled[(round(0.7*n)+1):n, ]

bank_train_labels <- shuffled[1:round(0.7 * n),]$y
bank_test_labels <- shuffled[(round(0.7*n)+1):n, ]$y

prop.table(table(bank_train_labels))
prop.table(table(bank_test_labels))

#STEP 3- Training a model on the data
#building the classifier
bankmodel <- naiveBayes(y~., data = bank_train)
bankmodel

#STEP 4- evaluating model performance
#making predictions
pred=predict(bankmodel,newdata=bank_test[-21])

CrossTable(bank_test$y, pred ,prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,dnn = c('actual term deposit subscription', 'predicted term deposit subscription'))
conf=table(bank_test$y,pred)
conf
#Accuracy = TP+TN/TP+FP+FN+TN
#it is simply a ratio of correctly predicted observation to the total observations
accuracy<- 100*(conf[1]+conf[4])/sum(conf)
print(accuracy)

#STEP 5- improving model performance
bankmodel2 <- naiveBayes(y~., data = bank_train, laplace=1)
pred=predict(bankmodel2, newdata=bank_test[-21])

CrossTable(bank_test$y, pred ,prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,dnn = c('actual term deposit subscription', 'predicted term deposit subscription'))
conf_lp=table(bank_test$y,pred)
conf_lp
accuracy<- 100*(conf_lp[1]+conf_lp[4])/sum(conf_lp)
print(accuracy)


##### decision trees
#{r setting data in random order}
set.seed(150)
sample.bank <-bank
#{r Partitioning data into 80% of training and remaining 20% for testing}
library(C50)
library(caret)
library(lattice)
library(ggplot2)
partitioned.sample.bank <- createDataPartition(
  sample.bank$y,
  times = 1,
  p = 0.8,
  list = F
)

#train vs test
sample.train = sample.bank[partitioned.sample.bank, ]
sample.test = sample.bank[-partitioned.sample.bank, ]
#{r making sure that data is in right proportion}
prop.table(table(sample.train$y))
prop.table(table(sample.test$y))
#{r number of columns}
ncol(sample.train)
#{r Decision tree of training dataset with 93.5% accuracy}
dt.sample.train.model <- C5.0(sample.train[-21], as.factor(sample.train$y))

dt.sample.train.model
summary(dt.sample.train.model)
#{r Predicting model}
dt.sample.train.predict <- predict(dt.sample.train.model, sample.test)

#{r Confusion Matrix of predicted training data and test data}
library(gmodels)
confusionMatrix(as.factor(sample.test$y), dt.sample.train.predict)
CrossTable(sample.test$y, dt.sample.train.predict,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default'))

#{r Performance Improvement by Boosting: Decision Tree of training data with 10 trials and 2.8% boost}
dt.sample.train.boost <- C5.0(sample.train[-21], as.factor(sample.train$y), trials = 10)
dt.sample.train.boost
summary(dt.sample.train.boost)

#{r predicting boosted model and comparing with test data resulting in 91.51% accuracy}
dt.sample.train.predict.boosted <- predict(dt.sample.train.boost, sample.test)


#lets check a crosstable comparison
confusionMatrix(as.factor(sample.test$y), dt.sample.train.predict.boosted)
CrossTable(sample.test$y, dt.sample.train.predict.boosted,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default'))

#{r Reducing False Positives which are more important than False Negatives}
error_cost <- matrix(c(0, 4, 2, 0), nrow = 2)
error_cost
dt.sample.train.boost <- C5.0(sample.train[-21], as.factor(sample.train$y), costs = error_cost)
bank.pred <- predict(dt.sample.train.boost, sample.test)
confusionMatrix(as.factor(sample.test$y), bank.pred)
CrossTable(sample.test$y, bank.pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default'))


#####random forest#
library(randomForest)
set.seed(17)
bank.rf<- randomForest(y~.,data = bank,
                       mtry = 5, importance = TRUE,
                       do.trace = 100)
print(bank.rf)

#model comparison#
#10 times error of random forest#
library(ipred)
set.seed(131)
error.bankRF<-numeric(10)
for (i in 1:10) error.bankRF[i] <-
  errorest(y ~ .,data = bank, 
           model = randomForest, mtry =5)$error
summary(error.bankRF)
#importance #
par(mfrow = c(1,2))
for (i in 1:2)
  plot(sort(bank.rf$importance[,i],dec = TRUE),
       type = "h", main = paste("Measure", i))

#label#
for (i in 1:2){
  imp<-as.data.frame(sort(bank.rf$importance[,i],dec = TRUE))
  plot(imp[,1],ylab='',type = "h", main = paste(colnames(bank.rf$importance)[i]))
  text(imp[,1],rownames(imp),cex=0.8,pos =3, col='red')
}

#label name is too long, thus use the table to show the order#
bank.rf$importance[order(bank.rf$importance[,1],decreasing=T),]
bank.rf$importance[order(bank.rf$importance[,2],decreasing=T),]




#References:
#https://rpubs.com/shekar07/413543
#https://rstudio-pubs-static.s3.amazonaws.com/259406_8349e4a1c11f40859fbdcd5736d295e8.html
#https://www.researchgate.net/publication/263054095_Bank_Direct_Marketing_Analysis_of_Data_Mining_Techniques