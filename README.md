# Movie Rating Prediction
This project aims to predict movie ratings using a neural network model. The model is built using TensorFlow/Keras and is trained on a dataset containing movie information such as genre, director, actors, year, duration, and votes.

# Dataset
The dataset used in this project is named movie.csv and contains the following columns:

## Name: Name of the movie
## Year: Release year of the movie
## Duration: Duration of the movie in minutes
## Genre: Genre of the movie
## Rating: Movie rating (target variable)
## Votes: Number of votes the movie received
## Director: Director of the movie
## Actor 1: Lead actor
## Actor 2: Supporting actor
## Actor 3: Supporting actor
# Project Structure
The notebook walks through the following steps:

1. Loading and Initial Data Exploration: Load the dataset and perform initial checks on its structure, missing values, and duplicates.
2. Data Cleaning and Preprocessing: Handle missing values, remove duplicates, and convert data types for relevant columns (Year, Duration, Votes).
3. Exploratory Data Analysis (EDA): Visualize the distribution of key features like Rating, Year, Duration, and Votes using histograms.
4. Feature Engineering: Drop irrelevant columns (ID, Name) and encode categorical features (Genre, Director, Actor 1, Actor 2, Actor 3) using Label Encoding.
5. Data Splitting and Scaling: Split the data into training and testing sets and scale the numerical features using StandardScaler.
6. Model Building: Define a sequential neural network model using TensorFlow/Keras with Dense layers and ReLU activation, and a linear output layer for regression.
7. Model Compilation and Training: Compile the model using the Adam optimizer and Mean Squared Error loss, and train it on the preprocessed training data.
8. Model Evaluation: Evaluate the trained model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared score. Visualize the training loss curve and the relationship between actual and predicted ratings.
9. Prediction: Demonstrate how to make predictions on new input data using the trained model.
Dependencies
The following libraries are required to run the notebook:

pandas
numpy
seaborn
matplotlib
sklearn
tensorflow
You can install these dependencies using pip:
