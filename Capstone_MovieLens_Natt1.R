## 1 Overview

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Loading library
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
library(kableExtra)

## 2 Methods/ Analysis 

# Check that edx dataset looks fine
head(edx)

# Include summary to drill down data in each column
summary(edx)

# Get distinct numbers of userId, movieId and enre combinations
distinct <- edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId),
            n_genres=n_distinct(genres))
distinct

# Ratings Mean
rating_mean <- mean(edx$rating)
rating_mean

# Plot distribution of movie ratings
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, fill = "gold", color="white") +
  xlab("Rating") +
  ylab("Count") +
  ggtitle("Movie Ratings Distribution") +
  theme(plot.title = element_text(hjust = 0.5))  # center the title

# User Count by Number of Ratings
edx %>% 
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.125, fill = "mediumpurple4", color="white") +
  scale_x_log10() +
  xlab("# Ratings") +
  ylab("# Users") +
  ggtitle("Users by Number of Ratings") +
  theme(plot.title = element_text(hjust = 0.5))

# Movie Count by Number of Ratings
edx %>% 
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(binwidth = 0.25, fill = "midnightblue", color="white") +
  scale_x_log10() +
  xlab("# Ratings") +
  ylab("# Movies") +
  ggtitle("Movies by Number of Ratings") +
  theme(plot.title = element_text(hjust = 0.5))

#### Movie Effects
# Calculate the average rating per movie and the rating frequency
avg_movie <- edx %>% 
  group_by(title) %>% 
  summarize(count_movie_rating=n(), avg_movie_rating = mean(rating), .groups = 'drop') %>%
  arrange(desc(count_movie_rating))
head(avg_movie)

# Plot average rating variation with smoothing
avg_movie %>% ggplot(aes(count_movie_rating, avg_movie_rating)) +
  geom_point(color = "gold") + 
  geom_smooth(method="loess") + 
  ggtitle("Average Movie Rating vs. Rating Frequency") +
  theme(plot.title = element_text(hjust = 0.5))

#### User Effects
# Calculate average ratings using number of ratings given by users
avg_user <- edx %>% 
  group_by(userId) %>% 
  summarize(count_user_rating=n(), avg_user_rating = mean(rating), .groups = 'drop') %>% 
  filter(count_user_rating > 100) %>%
  arrange(desc(count_user_rating))
head(avg_user)

#### Time Effects
# Convert time stamp to a datetime object and round it to the weekly unit
# Plot Timestamp (Week) vs. Average Ratings

edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating), .groups = 'drop') %>%
  ggplot(aes(date, rating)) +
  geom_point(color="chocolate") +
  geom_smooth() +
  ggtitle("Timestamp (Week) vs. Average Ratings")

#### Genre Effects
# Derive individual genres and calculate their mean and SD for ratings
edx %>% 
  separate_rows(genres, sep ="\\|") %>% 
  group_by(genres) %>%
  summarize(n = n(), avg_rating = mean(rating), se = sd(rating)/sqrt(n()), .groups = 'drop') %>%
  filter(n >= 1000) %>% 
  mutate(genres = reorder(genres, avg_rating)) %>%
  ggplot(aes(x = genres, y = avg_rating, ymin = avg_rating - 2*se, ymax = avg_rating + 2*se)) + 
  geom_point(color = "darkgreen") +
  geom_errorbar(color = "tomato4") + 
  ggtitle ("Mean Rating of Individual Genres with SD Error Bars") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5))

## 3 Results and Discussion 

# Split edx into train and test sets
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-edx_test_index,] 
test_set <- edx[edx_test_index,]

# Define RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

### 3.1 Base Model Using Mean Rating

# Base Model using Mean Rating
mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu_hat)
print(c('The Base Model RMSE :', naive_rmse))

# Save results to tibble for model comparison
rmse_results <- tibble(method = "Base Model - Mean Rating", 
                       RMSE_train_set = naive_rmse, RMSE_validation="NA")
rmse_results

### 3.2 Movie Effects Model

# Calculate b_i using the training set 
mu <- mean(train_set$rating)
movie_avgs <- train_set %>%
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu), .groups = 'drop')

# Plot distribution of movie bias
qplot(b_i, data = movie_avgs, bins = 20, 
      color = I("white"), fill=I("royalblue4"),
      main = 'Distribution of movie bias') + 
  theme(plot.title = element_text(hjust = 0.5))

# Predict rating due to movie effects
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  pull(b_i)

# Calculate RMSE due to movie effects
rmse_movies <- RMSE(test_set$rating, predicted_ratings)
print(c('The Movie Effects Model RMSE :', rmse_movies))

# Add new results to tibble for comparison
rmse_results <- add_row(rmse_results, method = "Movie Effects Model", 
                        RMSE_train_set = rmse_movies, RMSE_validation="NA")
rmse_results 

### 3.3 Movie and User Effects Model

# Plot user bias distribution
train_set %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating), .groups = 'drop') %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 20, color="white", fill = "magenta4") +
  ggtitle("User Bias Distribution") +
  theme(plot.title = element_text(hjust = 0.5))

# Calculate b_u using the training set 
user_avgs <- train_set %>%  
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i), .groups = 'drop')

# Predict ratings due to movie + user effects
predicted_ratings_bu <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE due to movie + user effects
rmse_movies_users <- RMSE(test_set$rating, predicted_ratings_bu)

# Add new results to tibble for comparison
rmse_results <- add_row(rmse_results, method = "Movie + User Effects Model", 
                        RMSE_train_set = rmse_movies_users, RMSE_validation="NA")
rmse_results

### 3.4 Movie-User-Time Effects Model

# Calculate time effects (b_t) using the training set
time_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u), .groups = 'drop')

# Predict ratings due to movie + user + time effects
predicted_ratings_bt <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(time_avgs, by='date') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  pull(pred)
  
# Calculate RMSE due to movie + user + time effects
rmse_movie_user_time <- RMSE(test_set$rating, predicted_ratings_bt)

# Add new results to tibble for comparison
rmse_results <- add_row(rmse_results, method = "Movie + User + Time Effects Model", 
                        RMSE_train_set = rmse_movie_user_time, RMSE_validation="NA")
rmse_results

### 3.5 Regularized Movie-User Effects Model

# Predict by regularization, movie and user effects model

lambdas <- seq(0,10,0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + l), .groups = 'drop')
  
  b_u <- train_set %>%
    left_join(b_i, by='movieId') %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n() + l), .groups = 'drop')
  
  predicted_ratings_lam <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i +  b_u) %>% 
    pull(pred)
  
  return(RMSE(predicted_ratings_lam, test_set$rating)) 
})

# Plot RMSE against lambdas to find optimal lambda
plot(lambdas, rmses, main="RMSE vs. Lambda", col="seagreen4")

# Obtain lambda that minimizes RMSE
lambda <- lambdas[which.min(rmses)] 
lambda

# Obtain RMSE from regularized user and movie effects
rmse_movie_user_reg <- min(rmses)

# Add new results to tibble for comparison
rmse_results <- add_row(rmse_results, method="Regularized Movie + User Effects Model", 
                        RMSE_train_set = rmse_movie_user_reg, RMSE_validation="NA")
rmse_results

### 3.6 Matrix Factorization with GD

if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
set.seed(123, sample.kind = "Rounding") # a randomized algorithm
library(recosystem)

# Convert the train and test sets into recosystem input format
train_reco <-  with(train_set, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))
test_reco  <-  with(test_set,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))

# Create the model object
r <-  recosystem::Reco()

# Determine optimal tuning parameters 'opts'

#opts <- r$tune(train_reco, opts = list(dim = c(10, 20, 30), 
#                                       lrate = c(0.1, 0.2),
#                                       costp_l2 = c(0.01, 0.1), 
#                                       costq_l2 = c(0.01, 0.1),
#                                       nthread  = 4, niter = 10))

# In order to avoid performance issues, the 'opts' chunk has been pre-run to derive the optimal tuning parameters as per below:

opts <- r$tune(train_reco, opts = list(dim = c(10),
                                       lrate = c(0.1),
                                       costp_l2 = c(0.1), 
                                       costq_l2 = c(0.1),
                                       nthread  = 4,
                                       niter = 10,
                                       verbose = TRUE))

# Train the algorithm  
r$train(train_reco, opts = c(opts$min, nthread = 4, niter = 10))

# Calculate the predicted values (ratings)
y_hat_reco <-  r$predict(test_reco, out_memory())
head(y_hat_reco, 10)

# Matrix Factorization - recosystem
rmse_matfac <- RMSE(test_set$rating, y_hat_reco)
print(c('The Matrix Factorization RMSE :', rmse_matfac))

# Add new results to tibble for comparison
rmse_results <- add_row(rmse_results, method="Matrix Factorization GD Model", 
                        RMSE_train_set = rmse_matfac, RMSE_validation="NA")
rmse_results

### 3.7 Final Validation

# For the final RMSE calculation, we use edx as train set and validation as test set

#### Matrix Factorization Model

# Predict final validation with the validation set

set.seed(123, sample.kind = "Rounding")

# Convert 'edx' and 'validation' sets to recosystem input format
edx_reco <-  with(edx, data_memory(user_index = userId, 
                                   item_index = movieId, 
                                   rating = rating))
validation_reco  <-  with(validation, data_memory(user_index = userId, 
                                                  item_index = movieId, 
                                                  rating = rating))

# Create the model object
r <-  recosystem::Reco()

# Tune the parameters
opts <-  r$tune(edx_reco, opts = list(dim = c(10), 
                                      lrate = c(0.1),
                                      costp_l2 = c(0.1), 
                                      costq_l2 = c(0.1),
                                      nthread  = 4,
                                      niter = 10,
                                      verbose = TRUE))

# Train the model
r$train(edx_reco, opts = c(opts$min, nthread = 4, niter = 10))

# Calculate the predicted values (ratings)
y_hat_reco_fin <-  r$predict(validation_reco, out_memory())

# Obtain RMSE by comparing to validation set ratings
rmse_matfac_fin <- RMSE(validation$rating, y_hat_reco_fin)
print(c('Matrix Factorization RMSE obtained from validation$rating:', rmse_matfac_fin))

# Comparing with all the models calculated
rmse_results

##### END