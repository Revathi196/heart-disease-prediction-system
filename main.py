import pandas as pd
from model import HeartDiseaseModel

def main():
    # Load sample data (replace with actual dataset)
    data = pd.DataFrame({
        'age': [45, 50],
        'cholesterol': [230, 180],
        'blood_pressure': [140, 120]
    })
    
    model = HeartDiseaseModel()
    model.train()
    predictions = model.predict(data)
    
    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
