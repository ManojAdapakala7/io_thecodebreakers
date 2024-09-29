import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # To control TensorFlow logging output
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib 


data = pd.read_csv('health_data.csv')


if data.isnull().sum().any():
    print("Missing values detected. Filling missing values with the mean...")
    data.fillna(data.mean(), inplace=True)


X = data[['Age', 'Gender', 'BMI', 'BloodPressure', 'CholesterolLevel', 'Smoking', 'Diabetes', 'AlcoholConsumption']].values
y = data[['HeartDisease', 'Diabetes', 'Stroke', 'LungDisease', 'LiverDisease']].values


X[:, 1] = np.where(X[:, 1] == 'Male', 1, 0)  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5)) 
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.5))  
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=5, activation='sigmoid'))  #


model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stopping])


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Example: Age 65, Male, BMI 28, BloodPressure 130, Cholesterol 220, Smokes, No diabetes, Alcohol consumption
new_person = scaler.transform([[65, 1, 28, 130, 220, 1, 0, 1]])
predictions = model.predict(new_person)


diseases = ['Heart Disease', 'Diabetes', 'Stroke', 'Lung Disease', 'Liver Disease']
for i, prob in enumerate(predictions[0]):
    print(f"Risk of {diseases[i]}: {prob * 100:.2f}%")

def predict_risk(age, gender, bmi, bp, cholesterol, cigarette, diabetes, alcohol):
    
    risk = {
        "heart_disease": 20,  
        "diabetes": 15,       
    }
    
    return risk


model.save('your_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")