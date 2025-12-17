import numpy as np
import pandas as pd
import pickle

# Load model and scaler once at module level for efficiency
def load_model_and_scaler():
    """Load the trained model and scaler from pickle files."""
    with open('saved_models/rent_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Feature columns used during training (exact order from model.py)
FEATURE_COLUMNS = [
    'BHK', 'Size', 'Bathroom',
    'Area Type_Carpet Area', 'Area Type_Super Area',
    'City_Chennai', 'City_Delhi', 'City_Hyderabad', 'City_Kolkata', 'City_Mumbai',
    'Furnishing Status_Semi-Furnished', 'Furnishing Status_Unfurnished',
    'Tenant Preferred_Bachelors/Family', 'Tenant Preferred_Family',
    'Size_per_BHK', 'Bathroomm_to_BHK_ratio'
]

def create_feature_dataframe(bhk, size, bathroom, area_type, city, furnishing_status, tenant_preferred):
    """
    Create a feature DataFrame with proper one-hot encoding matching model.py.
    
    Parameters:
    - bhk: Number of bedrooms (int)
    - size: Size in square feet (int)
    - bathroom: Number of bathrooms (int)
    - area_type: 'Super Area', 'Carpet Area', or 'Built Area'
    - city: 'Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Delhi', or 'Kolkata'
    - furnishing_status: 'Unfurnished', 'Semi-Furnished', or 'Furnished'
    - tenant_preferred: 'Bachelors', 'Bachelors/Family', or 'Family'
    
    Returns:
    - DataFrame with features matching the training format
    """
    # Start with base features
    data = {
        'BHK': bhk,
        'Size': size,
        'Bathroom': bathroom,
    }
    
    # One-hot encoding for Area Type (drop_first=True, so 'Built Area' is baseline)
    data['Area Type_Carpet Area'] = 1 if area_type == 'Carpet Area' else 0
    data['Area Type_Super Area'] = 1 if area_type == 'Super Area' else 0
    
    # One-hot encoding for City (drop_first=True, so 'Bangalore' is baseline)
    data['City_Chennai'] = 1 if city == 'Chennai' else 0
    data['City_Delhi'] = 1 if city == 'Delhi' else 0
    data['City_Hyderabad'] = 1 if city == 'Hyderabad' else 0
    data['City_Kolkata'] = 1 if city == 'Kolkata' else 0
    data['City_Mumbai'] = 1 if city == 'Mumbai' else 0
    
    # One-hot encoding for Furnishing Status (drop_first=True, so 'Furnished' is baseline)
    data['Furnishing Status_Semi-Furnished'] = 1 if furnishing_status == 'Semi-Furnished' else 0
    data['Furnishing Status_Unfurnished'] = 1 if furnishing_status == 'Unfurnished' else 0
    
    # One-hot encoding for Tenant Preferred (drop_first=True, so 'Bachelors' is baseline)
    data['Tenant Preferred_Bachelors/Family'] = 1 if tenant_preferred == 'Bachelors/Family' else 0
    data['Tenant Preferred_Family'] = 1 if tenant_preferred == 'Family' else 0
    
    # Feature engineering (same as model.py)
    data['Size_per_BHK'] = size / bhk
    data['Bathroomm_to_BHK_ratio'] = bathroom / bhk
    
    # Create DataFrame with columns in the exact order
    df = pd.DataFrame([data])[FEATURE_COLUMNS]
    
    return df

def predict_rent(bhk, size, bathroom, area_type, city, furnishing_status, tenant_preferred):
    """
    Predict house rent based on input features.
    
    Parameters:
    - bhk: Number of bedrooms (int)
    - size: Size in square feet (int)
    - bathroom: Number of bathrooms (int)
    - area_type: 'Super Area', 'Carpet Area', or 'Built Area'
    - city: 'Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Delhi', or 'Kolkata'
    - furnishing_status: 'Unfurnished', 'Semi-Furnished', or 'Furnished'
    - tenant_preferred: 'Bachelors', 'Bachelors/Family', or 'Family'
    
    Returns:
    - Predicted rent (int)
    """
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Create feature DataFrame
    features_df = create_feature_dataframe(
        bhk, size, bathroom, area_type, city, furnishing_status, tenant_preferred
    )
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Predict
    prediction = model.predict(features_scaled)
    
    return int(prediction[0])


if __name__ == "__main__":
    # Test prediction
    predicted = predict_rent(
        bhk=2,
        size=1000,
        bathroom=2,
        area_type='Super Area',
        city='Mumbai',
        furnishing_status='Semi-Furnished',
        tenant_preferred='Family'
    )
    print(f"Predicted Rent: Rs.{predicted:,}")
