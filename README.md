# 🏠 House Rent Predictor

A machine learning-powered web application that predicts house rent prices in major Indian cities based on property features. Built with Streamlit for an interactive user experience.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### 🔮 Rent Prediction
- Predict monthly rent based on property characteristics
- Supports 6 major Indian cities: Mumbai, Chennai, Bangalore, Hyderabad, Delhi, and Kolkata
- Considers BHK configuration, size, furnishing status, and tenant preferences

### 📊 Analysis Dashboard
- Interactive visualizations of rent distribution across cities
- BHK-wise rent analysis with scatter plots and bar charts
- Furnishing status impact on rent prices

### 📈 Advanced Insights
- Rent per square foot analysis
- Tenant preference breakdown
- Feature correlation heatmaps
- Dynamic filtering by city and BHK range

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **ML Framework** | Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Model** | Random Forest / Gradient Boosting Regressor |

## 📁 Project Structure

```
House Rent Predictor/
├── app.py                 # Main Streamlit application
├── model.py               # ML model training script
├── predict.py             # Prediction module
├── data/
│   └── House_Rent_Dataset.csv    # Training dataset
├── saved_models/
│   ├── rent_model.pkl     # Trained ML model
│   └── scaler.pkl         # Feature scaler
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/house-rent-predictor.git
   cd house-rent-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   
   Navigate to `http://localhost:8501`

## 📊 Dataset

The model is trained on a dataset containing rental property information from major Indian cities with the following features:

| Feature | Description |
|---------|-------------|
| BHK | Number of bedrooms |
| Size | Property size in square feet |
| Bathroom | Number of bathrooms |
| Area Type | Super Area / Carpet Area / Built Area |
| City | Mumbai, Chennai, Bangalore, Hyderabad, Delhi, Kolkata |
| Furnishing Status | Furnished / Semi-Furnished / Unfurnished |
| Tenant Preferred | Bachelors / Family / Bachelors/Family |

## 🧠 Model Training

To retrain the model with new data:

```bash
python model.py
```

The script will:
1. Load and preprocess the dataset
2. Train multiple models (Linear Regression, Random Forest, Gradient Boosting)
3. Select the best performing model based on R² score
4. Save the model and scaler to `saved_models/`

### Model Performance Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination

## 🖼️ Screenshots

### Rent Predictor
Enter property details and get instant rent predictions with city comparisons.

### Analysis Dashboard
Explore rent distributions, city-wise comparisons, and furnishing impact visualizations.

### Advanced Insights
Deep dive into rent per sqft analysis, tenant preferences, and feature correlations.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset sourced from public rental listings
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)

---

<p align="center">
  Made with ❤️ using Python & Streamlit
</p>
