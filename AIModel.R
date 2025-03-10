library(caret)
library(randomForest)
library(gbm)
library(plumber)
library(jsonlite)

# Load the dataset
dataset <- read.csv("car_price_prediction.csv")

# Convert categorical variables to factors
dataset[sapply(dataset, is.character)] <- lapply(dataset[sapply(dataset, is.character)], as.factor)

# Normalize numerical features
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

numerical_features <- sapply(dataset, is.numeric)
dataset[numerical_features] <- lapply(dataset[numerical_features], normalize)

# Split into training and test sets
set.seed(123)
train_index <- createDataPartition(dataset$Price, p = 0.8, list = FALSE)
train_data <- dataset[train_index, ]
test_data <- dataset[-train_index, ]

# Train a Random Forest model
rf_model <- randomForest(Price ~ ., data = train_data, ntree = 500, mtry = 5)

# Train a Gradient Boosting Model
gbm_model <- gbm(Price ~ ., data = train_data, distribution = "gaussian", n.trees = 500, interaction.depth = 5, shrinkage = 0.1, cv.folds = 5)

# Save models
saveRDS(rf_model, "rf_model.rds")
saveRDS(gbm_model, "gbm_model.rds")

# Create API using plumber
predict_price <- function(model_type, features_json) {
  features <- fromJSON(features_json)
  features <- as.data.frame(features)
  features[sapply(features, is.character)] <- lapply(features[sapply(features, is.character)], as.factor)
  features[sapply(features, is.numeric)] <- lapply(features[sapply(features, is.numeric)], normalize)
  
  if (model_type == "rf") {
    model <- readRDS("rf_model.rds")
  } else {
    model <- readRDS("gbm_model.rds")
  }
  
  predicted_price <- predict(model, features)
  return(predicted_price)
}

# Plumber API setup
pr <- plumb("plumber.R")
pr$handle("POST", "/predict", function(req, res) {
  model_type <- req$body$model
  features_json <- req$body$features
  res$body <- list(predicted_price = predict_price(model_type, features_json))
})

pr$run(port = 8000)

# Function for user input prediction
predict_price_interactive <- function() {
  cat("Enter Car Brand: ")
  brand <- as.character(readline())
  cat("Enter Car Model: ")
  model <- as.character(readline())
  cat("Enter Year: ")
  year <- as.numeric(readline())
  cat("Enter Engine Size: ")
  engine_size <- as.numeric(readline())
  cat("Enter Fuel Type: ")
  fuel_type <- as.character(readline())
  cat("Enter Transmission Type: ")
  transmission <- as.character(readline())
  cat("Enter Mileage: ")
  mileage <- as.numeric(readline())
  cat("Enter Number of Doors: ")
  doors <- as.numeric(readline())
  cat("Enter Owner Count: ")
  owner_count <- as.numeric(readline())
  
  user_data <- data.frame(Brand = factor(brand, levels = levels(dataset$Brand)),
                          Model = factor(model, levels = levels(dataset$Model)),
                          Year = normalize(year),
                          Engine_Size = normalize(engine_size),
                          Fuel_Type = factor(fuel_type, levels = levels(dataset$Fuel_Type)),
                          Transmission = factor(transmission, levels = levels(dataset$Transmission)),
                          Mileage = normalize(mileage),
                          Doors = normalize(doors),
                          Owner_Count = normalize(owner_count))
  
  user_data <- dummyVars(" ~ .", data = user_data) %>% predict(user_data) %>% as.data.frame()
  predicted_price <- predict(rf_model, user_data)
  
  cat("Predicted Car Price: ", predicted_price, "\n")
}

# Call function for interactive prediction
predict_price_interactive()
