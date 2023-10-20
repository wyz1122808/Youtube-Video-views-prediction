# Youtube-Video-views-prediction
This is a simple application built to predict the view count of MrBeast's YouTube videos based on certain video characteristics.

## Features

- Data Collection: The data is sourced from a CSV file named MrBeast_youtube_stats.csv.
- Data Processing: The titles of the videos are processed with the TF-IDF (Term Frequency-Inverse Document Frequency) technique, and other numerical features are standardized.
- Model: A sequential neural network model is implemented using Keras to predict the view count.


### Data Preparation:

- The video data is loaded from a CSV file.
- The age of each video is computed by taking the difference between the publish time and a given data collection date.
- Only relevant features such as title, view count, duration in seconds, and video age in days are considered.
- Any missing values are dropped and the index is reset.

### Modeling

- The data is split into training and testing sets.
- A neural network model with several dense layers and dropout layers is defined.
- The model is trained on the training set and early stopping is applied based on validation loss.

## Example Usage
To predict the view count for a video:

- Input the video's title.
- Input the video's duration in seconds.
- Input the video's age in days.
  
The model will then provide a prediction for the view count of the video.
