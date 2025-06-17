# Heart Disease Prediction Web App (Streamlit)

This is a simple Streamlit web app for predicting heart disease using a trained KNeighborsClassifier model.

## How to run

1. Put your trained `model_knn.pkl` and `scaler.pkl` files in this directory.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the browser at the shown address (usually http://localhost:8501).

## Features

- Input form for single prediction
- CSV upload for batch predictions
- Prediction results with colored bar chart visualization
- Download batch prediction results as CSV