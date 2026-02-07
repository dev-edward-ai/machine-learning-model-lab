# 🚀 Complete Setup Guide - Business Insight Generator

## Overview
The Business Insight Generator consists of:
- **Backend**: FastAPI server running on port 8000
- **Frontend**: Static HTML/CSS/JS running on port 3000
- **Communication**: HTTP JSON API calls

## Prerequisites
- Python 3.8+
- pip (Python package manager)
- A terminal/PowerShell

## Step 1: Install Required Packages

Run this command in PowerShell or your terminal:

```powershell
pip install fastapi uvicorn pandas numpy scikit-learn
```

Verify installation:
```powershell
python -c "import fastapi, uvicorn, pandas, numpy; from sklearn import ensemble; print('✅ All packages installed')"
```

## Step 2: Start the Backend Server

Open **Terminal 1** and run:

```bash
cd c:\Users\User\OneDrive\Desktop\mlmodels-lab\machine-learning-model-lab\backend
python -m uvicorn api.main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started server process [XXXX]
INFO:     Application startup complete.
```

✅ Backend is now running on `http://localhost:8000`

## Step 3: Start the Frontend Server

Open **Terminal 2** and run:

```bash
cd c:\Users\User\OneDrive\Desktop\mlmodels-lab\machine-learning-model-lab\frontend
python -m http.server 3000 --directory .
```

**Expected Output:**
```
Serving HTTP on 0.0.0.0 port 3000 (http://0.0.0.0:3000/) ...
```

✅ Frontend is now running on `http://localhost:3000`

## Step 4: Open in Browser

Open **Terminal 3** or Browser and go to:
```
http://localhost:3000
```

You should see:
- A header with "Business Insight Generator"
- A status badge showing "✅ Backend Connected"
- A drop zone for uploading CSV files

## Step 5: Test with Sample Data

Choose one of these sample CSVs from `samples/` folder:

### 1. **Crypto Trading** - `crypto_signals.csv`
- **Columns**: Open, Close, Volume, RSI, MACD
- **Model**: LogisticRegression
- **Output**: Trading Signal (BUY/SELL)
- **UI**: Trading Terminal with sliders

### 2. **Medical Diagnosis** - `heart_disease.csv`
- **Columns**: Age, BMI, BloodPressure, Glucose, Cholesterol
- **Model**: KNeighborsClassifier
- **Output**: Risk Level (Low/Medium/High)
- **UI**: Health Screener

### 3. **Car Pricing** - `used_car_prices.csv`
- **Columns**: Year, Mileage, Horsepower, Brand
- **Model**: DecisionTreeRegressor
- **Output**: Estimated Price
- **UI**: Price Estimator with sliders

### 4. **Sales Forecasting** - `marketing_roi.csv`
- **Columns**: AdSpend, SocialClicks, Season, Budget
- **Model**: RandomForestRegressor
- **Output**: Predicted Revenue & ROI
- **UI**: ROI Simulator

## Testing Instructions

### Option A: Drag & Drop (Easiest)
1. Locate sample CSV file in file explorer
2. Drag it into the drop zone on the website
3. Wait for model training (2-3 seconds)
4. Use the generated UI to make predictions

### Option B: Click Upload Button
1. Click "Choose File" button
2. Navigate to `samples/` folder
3. Select desired CSV
4. Wait for processing
5. Interact with the UI

## Expected Workflow

```
1. Upload CSV
   ↓
2. Backend detects scenario (auto)
   ↓
3. Backend trains model (2-3 seconds)
   ↓
4. Frontend receives scenario type
   ↓
5. UIFactory renders scenario-specific interface
   ↓
6. User adjusts inputs (sliders, forms)
   ↓
7. Real-time predictions show instantly
   ↓
8. Results update as you change values
```

## API Endpoints Reference

### Health Check
```
GET http://localhost:8000/ping
```
Returns: `{"status": "healthy", "message": "..."}`

### Analyze CSV (Upload)
```
POST http://localhost:8000/analyze
Content-Type: multipart/form-data
Body: file=<csv_file>
```
Returns: `{"scenario": "CRYPTO|MEDICAL|CAR_PRICE|SALES", "accuracy": 0.85, ...}`

### Make Predictions

**Crypto:**
```
POST http://localhost:8000/predict/crypto
Content-Type: application/json
{"open": 65000, "close": 65500, "volume": 1000000, "rsi": 55}
```

**Medical:**
```
POST http://localhost:8000/predict/medical
{"age": 45, "bmi": 28.5, "bloodpressure": 120, "glucose": 125}
```

**Car Price:**
```
POST http://localhost:8000/predict/car_price
{"year": 2015, "mileage": 80000, "horsepower": 200}
```

**Sales:**
```
POST http://localhost:8000/predict/sales
{"adspend": 5000, "socialclicks": 10000, "season": "summer"}
```

## Troubleshooting

### "Backend Connection Failed"
- ❌ Make sure Terminal 1 is running with `python -m uvicorn api.main:app ...`
- ❌ Check that port 8000 is not in use
- ✅ Try stopping and restarting the server

### "404 Not Found" Error
- ❌ Make sure Terminal 2 is running with `python -m http.server 3000`
- ❌ Browser URL should be `http://localhost:3000` (not 8000)
- ✅ Clear browser cache (Ctrl+Shift+Delete)

### "ModuleNotFoundError" on Backend
```powershell
# Reinstall packages
pip install --upgrade fastapi uvicorn pandas numpy scikit-learn

# Verify from backend directory
cd backend
python -c "from api.main import app; print('✅ Ready')"
```

### CSV File Not Processing
- ❌ Check file is valid CSV (not Excel)
- ❌ Check file has proper column names
- ❌ Check file is not empty
- ✅ Try one of the sample CSVs first

### Model Not Training
- Look at backend terminal for error messages
- Check that DataFrame has minimum 5 rows
- Verify CSV is properly formatted

## File Structure

```
mlmodels-lab/
├── backend/
│   ├── logic.py           ← ML models & detection
│   ├── api/
│   │   └── main.py        ← FastAPI routes
│   └── __init__.py
├── frontend/
│   ├── index.html         ← HTML structure
│   ├── script.js          ← JavaScript logic
│   └── styles.css         ← Design/styling
├── samples/
│   ├── crypto_signals.csv
│   ├── heart_disease.csv
│   ├── used_car_prices.csv
│   └── marketing_roi.csv
└── START_SERVER.ps1       ← Setup instructions
```

## Quick Command Reference

```powershell
# Terminal 1: Start Backend
cd c:\Users\User\OneDrive\Desktop\mlmodels-lab\machine-learning-model-lab\backend
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2: Start Frontend
cd c:\Users\User\OneDrive\Desktop\mlmodels-lab\machine-learning-model-lab\frontend
python -m http.server 3000 --directory .

# Terminal 3: Open Browser
start http://localhost:3000
```

## Performance Notes

- **CSV Upload**: <100ms
- **Scenario Detection**: <10ms
- **Model Training**: 1-3 seconds (depends on file size)
- **Prediction**: <50ms
- **UI Rendering**: <100ms

Total end-to-end time: ~2-4 seconds from upload to interactive UI

## Next Steps

1. ✅ Install dependencies
2. ✅ Start both servers
3. ✅ Upload sample CSV
4. ✅ Test predictions
5. ✅ Explore all 4 scenarios

## Support

If you encounter issues:
1. Check backend terminal for error messages
2. Verify all terminals are running
3. Try refreshing browser (Ctrl+F5)
4. Restart both servers
5. Check firewall isn't blocking ports 8000 or 3000

---

**Status**: ✅ Ready to Deploy

**Version**: 1.0

**Created**: February 2026
