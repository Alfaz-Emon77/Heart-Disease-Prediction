# Heart Disease Prediction Web App (Streamlit)

This is a simple Streamlit web app for predicting heart disease using a trained KNeighborsClassifier model.

## How to run

1. Put your trained `model_knn.pkl` and `scaler.pkl` files in this directory.

import joblib
from google.colab import files

# মডেল এবং স্কেলার সেভ করা
joblib.dump(model_kneighbors, 'model_knn.pkl')
joblib.dump(sc, 'scaler.pkl')

# ডাউনলোড করা
files.download('model_knn.pkl')
files.download('scaler.pkl')

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
