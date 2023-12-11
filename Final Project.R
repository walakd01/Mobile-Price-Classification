
data <- read.csv("C:\\Users\\Admin\\Documents\\train.csv")


library(tidyverse)
library(tidymodels)
library(ggplot2)
library(skimr)
library(tidyr)
library(GGally)
library(reshape2)
library(corrplot)
library(nnet)
library(ROCR)
library(gains)


glimpse(data)
hist(data$mobile_wt)

data$price_range <- as.factor(data$price_range)
data_train$price_range <- as.factor(data_train$price_range)
data_test$price_range <- as.factor(data_test$price_range)


#Distribution of the battery power
ggplot(data, aes(x = battery_power)) +
  geom_histogram(binwidth = 100, fill = "blue", color = "black", aes(y = ..density..)) +
  labs(title = "Histogram of Battery Power",
       x = "Battery Power",
       y = "Density")+ 
  theme(
    text = element_text(size = 16),  # Set the overall text size
    axis.text = element_text(size = 12),  # Set the axis text size
    axis.title = element_text(size = 14),  # Set the axis title size
    title = element_text(size = 16)  # Set the plot title size
  )



#Box plot of the RAM vs Price class
ggplot(data, aes(x = as.factor(price_range), y = ram)) +
  geom_boxplot(fill = "skyblue", color = "darkblue") +
  labs(x = "Price Class", y = "RAM") +
  ggtitle("Boxplot of RAM across Price Classes")+
  theme(
    text = element_text(size = 20),  # Set the overall text size
    axis.text = element_text(size = 20),  # Set the axis text size
    axis.title = element_text(size = 20),  # Set the axis title size
    title = element_text(size = 20)  # Set the plot title size
  )



#Scatter plot of Battery power and talk time
ggplot(data, aes(x = battery_power, y = talk_time)) +
  geom_point() +
  labs(title = "Scatter battery power and talk time", x = "battery power", y = "talk time", fill="blue")
#Correlation coefficient between battery power and talk time
correlation1 <- cor(data$battery_power, data$talk_time)

# Print the correlation coefficient
print(correlation1)



#Scatter plot of front camera and primary camera
ggplot(data, aes(x = fc, y = pc)) +
  geom_point() +
  labs(title = "front camera and primary camera", x = "front camera", y = "primary camera", fill="fc")
#Correlation coefficient between battery power and talk time
correlation2 <- cor(data$fc, data$pc)

# Print the correlation coefficient
print(correlation2)



#Box plot of the clock speed vs number of cores
ggplot(data, aes(x = as.factor(n_cores), y = clock_speed)) +
  geom_boxplot(fill = "skyblue", color = "darkblue") +
  labs(x = "number of cores", y = "clock speed") +
  ggtitle("Boxplot of number of cores  vs clock speed")+
  theme(
    text = element_text(size = 20), 
    axis.text = element_text(size = 20),  
    axis.title = element_text(size = 20),  
    title = element_text(size = 20)  
  )



#Plot of 3G and 4G
ggplot(data, aes(x = factor(three_g), fill = factor(four_g))) +
  geom_bar(position = "dodge", stat = "count") +
  labs(title = "3G and 4G Presence",
       x = "3G Support",
       y = "Count",
       fill = "4G Support") +
  scale_fill_manual(values = c("0" = "red", "1" = "lightblue"), name = "4G Support") +
  theme_minimal()



#Box plot of internal memory by 4g support
ggplot(data, aes(x = factor(four_g), y = int_memory, fill = factor(four_g))) +
  geom_boxplot() +
  labs(title = "Boxplot of Internal Memory by 4G Support",
       x = "4G Support",
       y = "Internal Memory",
       fill = "4G Support") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "red"))+
  theme(
    text = element_text(size = 20), 
    axis.text = element_text(size = 20),  
    axis.title = element_text(size = 20),  
    title = element_text(size = 20)  )



#Correlation matrix heat map
cor_matrix <- cor(data[, c(1, 14, 3, 6, 17, 20)])
par(mar = c(1, 1, 1, 1))
corrplot(cor_matrix, addCoef.col = TRUE, number.cex = 0.75,)


#Proportion of phones with mobiles that support 4g
ggplot(data, aes(x = "", fill = factor(four_g))) +
  geom_bar(width = 1, stat = "count") +
  coord_polar(theta = "y") +
  labs(title = "Proportion of Mobile with 4G Support",
       fill = "4G Support") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "red"))



model1 <- lm(price_range ~ ., data = data)
summary(model1)

#Sample linear model to check the important variables
model2 <- lm(price_range ~ battery_power + clock_speed + int_memory + px_height + px_width + ram, data = data)
summary(model2)


#Sample linear model to check the important variables
model3 <- lm(price_range ~ battery_power + int_memory + px_height + px_width + ram, data = data)
summary(model3)


AIC <- c(AIC(model1), AIC(model2), AIC(model3))
AIC



##Model preparation and variable selection
#Selecting the important variables for summary statistics
data_select <- data %>%
  select(c(battery_power, clock_speed, int_memory, px_height, px_width, ram, price_range))

#Summary Statistics
data_select %>%
  skim()


##Training and evaluating the multi-class classifier
##Multinomial Regression
# Split data 70%-30% into training set and test set
set.seed(4040)
data_split <- data_select %>% 
  initial_split(prop = 0.70, strata = price_range)

# Extract data in each split
data_train <- training(data_split)
data_test <- testing(data_split)

# Print the number of observations in each split
cat("Training cases: ", nrow(data_train), "\n",
    "Test cases: ", nrow(data_test), sep = "")

###Multinomial regression
multireg_price <- multinom_reg(penalty = 1) %>%
  set_engine("nnet") %>%
  set_mode ("classification")

#Train the multinomial regression model without any preprocessing
set.seed(50)
data_train$price_range <- as.factor(data_train$price_range)
multireg_fit <- multireg_price %>%
  fit(price_range ~ ., data = data_train)

#Print the model
summary (multireg_fit)



#Predicting the test data
results <- data_test %>% select(price_range) %>%
  bind_cols(multireg_fit %>%
              predict (new_data = data_test)) %>%
  bind_cols(multireg_fit %>%
              predict(new_data = data_test, type = "prob"))

#Print the predicted values
results %>%
  slice_head(n = 10)

#Confusion matrix
##data$price_range <- as.factor(data$price_range)
results %>%
confusion_matrix <- results %>%
 conf_mat(price_range, .pred_class)
  ##
# Print the confusion matrix
print(confusion_matrix)


##update_geom_defaults(geom = "tile", new = list(color = "black", alpha = 0.7))
# Visualize confusion matrix
results %>% 
 conf_mat(price_range, .pred_class) %>% 
      autoplot(type = "heatmap")

# Make an ROC_CURVE
rocss<-results %>% 
  roc_curve(price_range, c(.pred_0, .pred_1, .pred_2, .pred_3)) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = .level)) +
  geom_abline(lty = 2, color = "gray80", size = 0.9) +
  geom_path(show.legend = T, alpha = 0.6, size = 1.2) +
  coord_equal()

rocss


#######################################################################

# Predicting a new result with Logistic Regression
model_predd = predict(multireg_fit, new_data = data_test, type = "class") 
model_predd

model_pred <- cbind(data_test, model_predd)

# Data frame with actual and predicted values
model_results <- data.frame(price_range = data_test$price_range, 
                            Predicted = model_pred)
model_results

model_gains = gains(as.numeric(data_test$price_range), as.numeric(model_pred$.pred_class), 10)
model_gains 
names(model_gains)
plot(model_gains)



# Extract columns from gains object into data frame
model_gains_df <- data.frame(depth = model_gains$depth,
                            mean.prediction = model_gains$mean.prediction,
                            mean.response = model_gains$mean.resp)


# Plot of Depth against mean responses
ggplot(model_gains_df, aes(x = depth)) +
  geom_line(aes(y = mean.prediction, color = "Predicted"), linetype = "dashed") +
  geom_line(aes(y = mean.response, color = "Actual")) +
  labs(x = "Depth",
       y = "Mean Response",
       title = "Mean Response of the classification") +
  scale_color_manual(name = "Response",
                     values = c("Predicted" = "red", 
                                "Actual" = "blue")) +
  theme_minimal()




#Calculating AUC did not work because it is a multiclass ( more than 2 classifications)
model_roc.pred = prediction(as.numeric(data_test$price_range), as.numeric(model_pred$.pred_class))

# Calculate AUC
auc_model <- performance(model_roc.pred, measure = "auc")@y.values[[1]]


#######################################################
new <- read.csv("C:\\Users\\Admin\\Documents\\test.csv")
nrow(new)
pred2 <- predict (multireg_fit, new_data = new, type = "class")

Predict_mobile_price_range <- cbind(new, pred2)

write.csv(Predict_mobile_price_range, "mobile.csv")


saveRDS(multireg_fit, "multireg_fit.rds")
