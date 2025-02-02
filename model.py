import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'age': [40, 50, 60, 35, 45],
    'cholesterol': [210, 250, 180, 200, 220],
    'blood_pressure': [130, 140, 110, 120, 135],
    'heart_disease': [1, 1, 0, 0, 1]
})

class HeartDiseaseModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()
    
    def train(self):
        X = data[['age', 'cholesterol', 'blood_pressure']]
        y = data['heart_disease']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)
