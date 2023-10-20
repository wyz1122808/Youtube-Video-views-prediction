import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('./MrBeast_youtube_stats.csv')
df['publishTime'] = pd.to_datetime(df['publishTime'])



# Calculate age of the videos
data_collection_date = pd.Timestamp("2021-12-30 00:00:00+00:00")
df['video_age_days'] = (data_collection_date - df['publishTime']).dt.days
df = df[df['duration_seconds'] != 0]

df = df[['title', 'viewCount', 'duration_seconds', 'video_age_days']]


df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)



print(df.head())
print(df['viewCount'].describe())


# TF-IDF Vectorization of the titles
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
title_tfidf = tfidf_vectorizer.fit_transform(df['title']).toarray()

# Create a DataFrame from the TF-IDF matrix
title_df = pd.DataFrame(title_tfidf, columns=tfidf_vectorizer.get_feature_names_out())

# Scale numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['duration_seconds', 'video_age_days']])




# Combine features
X = pd.concat([pd.DataFrame(title_tfidf), pd.DataFrame(scaled_features)], axis=1)
y = df['viewCount']


y_scaler = StandardScaler()
y = y_scaler.fit_transform(df[['viewCount']])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))  # The input shape is the number of features
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))





model.add(Dense(1, activation='relu'))  # Output layer: single neuron for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=3, validation_split=0.1, verbose=1, callbacks=[early_stop])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'test'])
plt.show()


mse = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")


def predict_viewCount(title, duration, video_age):
    title_tfidf_pred = tfidf_vectorizer.transform([title]).toarray()
    scaled_features_pred = scaler.transform([[duration, video_age]])
    combined_features = pd.concat([pd.DataFrame(title_tfidf_pred), pd.DataFrame(scaled_features_pred)], axis=1).values
    return y_scaler.inverse_transform(model.predict(combined_features))[0][0]

# Example Usage
title_input = input("Enter the video title: ")
duration_input = float(input("Enter the video duration in seconds: "))
video_age_input = float(input("Enter the video age in days: "))

predicted_views = predict_viewCount(title_input, duration_input, video_age_input)
print(f"Predicted viewCount: {predicted_views}")