# 🤖 AutoML Intelligence Platform

**Professional AutoML platform with Smart Dispatcher, 13 real-world scenarios, and production deployment.**

Transform your CSV data into intelligent insights with zero configuration. Our platform automatically detects the best machine learning model, runs a model tournament showing top 3 performers with accuracy scores, and explains everything using real-world examples.

![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![Python](https://img.shields.io/badge/Python-3.11-green?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-teal?logo=fastapi)
![ML Models](https://img.shields.io/badge/ML%20Models-13+-purple)

---

## ✨ Features

- **🏆 Smart Dispatcher** - Model tournament showing top 3 performers with "85% accuracy" style metrics
- **🎯 13 Real-World Scenarios** - From crypto trading to fraud detection, each with sample data
- **💡 Enhanced Explanations** - Every model has industry-specific examples (crypto, healthcare, finance)
- **🐳 Docker Ready** - One command deployment, works anywhere
- **📊 13+ ML Algorithms** - Logistic/Linear Regression, Decision Tree, Random Forest, XGBoost, SVM, KNN, Naive Bayes, K-Means, PCA, Isolation Forest
- **🎨 Premium UI** - Modern dark theme with glassmorphism and smooth animations
- **⚡ Production Ready** - FastAPI backend, Nginx frontend, fully containerized
- **🔍 Automatic Scenario Detection** - Platform identifies which ML use case fits your data

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- ✅ Check Docker installation
- ✅ Build containers
- ✅ Start all services
- ✅ Open your browser automatically

### Option 2: Manual Setup

```bash
# Build and start containers
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 3: Development Mode

```bash
# Backend
cd backend
pip install -r ../requirements.txt
uvicorn api.main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend
python -m http.server 3000
```

---

## 🎯 13 Real-World Scenarios

The platform comes with **13 complete ML scenarios**, each with sample data ready to test:

### Classification (6 scenarios)
1. **💰 Crypto Buy/Sell Signal** - Trading signals with technical indicators (RSI, MACD)
2. **🏦 Loan Approval Assistant** - Interpretable financial decision-making
3. **📱 SMS Spam Detector** - Text-based filtering with Naive Bayes
4. **💵 Fake Banknote Detector** - Precision boundary detection with SVM
5. **❤️ Heart Disease Predictor** - Medical ensemble classification
6. **📊 Customer Churn Predictor** - Subscription cancellation prediction

### Regression (4 scenarios)
7. **📈 Marketing ROI Calculator** - Linear trend analysis for ad spend
8. **🚗 Used Car Price Estimator** - Non-linear vehicle pricing
9. **🏠 Airbnb Nightly Rate** - Neighborhood-based pricing with KNN
10. **✈️ Flight Delay Prediction** - Complex interaction modeling

### Unsupervised (3 scenarios)
11. **🎨 Color Palette Generator** - K-Means pixel clustering
12. **📉 Stock Market Visualizer** - PCA dimensionality reduction
13. **🔍 Credit Card Fraud Detection** - Isolation Forest anomaly detection

**All scenarios include:**
- ✅ Sample CSV datasets in `samples/` folder
- ✅ Real-world industry examples
- ✅ Optimized model parameters
- ✅ Business insights and explanations

See **[SCENARIOS.md](./SCENARIOS.md)** for detailed documentation.

---

## 📚 Supported ML Models

Our platform includes **13+ machine learning algorithms**:

### Supervised Learning

#### Classification
- **Logistic Regression** - Binary/multi-class classification
- **Decision Tree** - Interpretable rule-based decisions
- **KNN (K-Nearest Neighbors)** - Pattern matching based on similarity
- **SVM (Support Vector Machine)** - Maximum margin classification
- **Random Forest** - Ensemble of decision trees
- **Naive Bayes** - Probabilistic classifier
- **XGBoost** - Gradient boosting champion

#### Regression
- **Linear Regression** - Continuous value prediction
- **Decision Tree Regressor** - Non-linear regression
- **KNN Regressor** - Neighborhood-based prediction
- **Random Forest Regressor** - Ensemble regression
- **XGBoost Regressor** - High-performance boosting

### Unsupervised Learning

- **K-Means Clustering** - Automatic customer/data segmentation
- **PCA (Principal Component Analysis)** - Dimensionality reduction
- **Isolation Forest** - Anomaly/fraud detection

---

## 🎯 Use Cases

| Use Case | Example | Recommended Model |
|----------|---------|-------------------|
| **Customer Churn** | Predict which customers will leave | Logistic Regression, Random Forest, XGBoost |
| **Sales Forecasting** | Predict future revenue | Linear Regression, XGBoost Regressor |
| **Customer Segmentation** | Group similar customers | K-Means Clustering |
| **Fraud Detection** | Identify suspicious transactions | Isolation Forest, SVM |
| **Product Recommendations** | Suggest items based on similarity | KNN |
| **Disease Diagnosis** | Medical classification | Naive Bayes, Random Forest |
| **Price Prediction** | Estimate house/product prices | Random Forest, XGBoost |

---

## 💡 How It Works

1. **Upload Your CSV** - Drag and drop or click to browse
2. **Select Your Goal** - Choose business objective (churn, revenue, segmentation, etc.)
3. **Automatic Analysis** - Platform runs model tournament and selects best algorithm
4. **Get Insights** - View results with business-friendly explanations and real-world analogies

### Example Real-World Explanation (KNN):

> **🏘️ The Neighborhood Analogy**
> 
> KNN is like determining who you are based on your neighbors. If you speak Thai and your closest neighbors are Thai, you're probably Thai! The algorithm looks at the K closest data points and makes a decision based on what's most common among them.
>
> **Real-World Scenario:** Imagine you're living in the USA but speak Thai. KNN checks your 5 closest neighbors - if 4 of them are Thai and 1 is American, it predicts you're Thai! Same applies to product recommendations...

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Docker Compose Network            │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │   Frontend   │      │     Backend     │ │
│  │  (Nginx)     │◄────►│   (FastAPI)     │ │
│  │  Port 3000   │      │   Port 8000     │ │
│  └──────────────┘      └─────────────────┘ │
│        │                       │            │
│        │                       ▼            │
│        │              ┌─────────────────┐  │
│        │              │ ML Engine       │  │
│        │              │ - Auto Detection│  │
│        └──────────────┤ - 10+ Models    │  │
│         HTTP Requests │ - Explanations  │  │
│                       └─────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## 📖 API Documentation

### Analyze Endpoint

**POST** `/analyze`

Automatic model detection and analysis.

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@your_data.csv" \
  -F "business_objective=churn" \
  -F "target_col=churned"
```

**Response:**
```json
{
  "recommended_model": "Random Forest Classifier",
  "task_type": "classification",
  "metric_value": 0.92,
  "reasoning": "Selected because it achieved the highest accuracy of 0.920...",
  "business_insights": {
    "headline": "ALERT: 42 entities flagged as high risk (18.5% of records)",
    "recommended_action": "Prioritize outreach to high-risk customers..."
  },
  "model_explanation": {
    "analogy": "🌲🌲🌲 Wisdom of the Crowd",
    "how_it_works": "Random Forest creates hundreds of decision trees...",
    "real_world_example": "Instead of one banker reviewing your application..."
  }
}
```

### Smart Dispatch Endpoint (NEW!)

**POST** `/smart-dispatch`

Run model tournament and get top 3 performers with performance metrics.

**Request:**
```bash
curl -X POST http://localhost:8000/smart-dispatch \
  -F "file=@samples/crypto_signals.csv" \
  -F "target_col=buy_signal"
```

**Response:**
```json
{
  "scenario": {
    "name": "Crypto Buy/Sell Signal",
    "icon": "💰",
    "confidence": 92.5,
    "industry": "Finance/Trading"
  },
  "top_models": [
    {"name": "Logistic Regression", "score": 85.2, "score_type": "Accuracy"},
    {"name": "Random Forest Classifier", "score": 83.1, "score_type": "Accuracy"},
    {"name": "XGBoost Classifier", "score": 82.8, "score_type": "Accuracy"}
  ],
  "recommended_model": {
    "name": "Logistic Regression",
    "explanation": "Best for binary classification with probability estimates..."
  },
  "dataset_summary": {
    "num_rows": 50,
    "num_cols": 9,
    "num_numeric": 8,
    "num_categorical": 1
  }
}
```

### List Scenarios

**GET** `/scenarios`

Get all 13 available real-world ML scenarios.

```bash
curl http://localhost:8000/scenarios
```

### Interactive API Docs

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.11, FastAPI, scikit-learn, XGBoost, pandas, numpy
- **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
- **Infrastructure:** Docker, Docker Compose, Nginx
- **ML Libraries:** scikit-learn, XGBoost, pandas, numpy

---

## 📁 Project Structure

```
machine-learning-model-lab/
├── backend/
│   ├── api/
│   │   ├── routers/          # API endpoints
│   │   ├── services/         # ML logic & explanations
│   │   ├── schemas/          # Data models
│   │   └── main.py           # FastAPI app
│   └── Dockerfile
├── frontend/
│   ├── index.html            # Main UI
│   ├── app.js                # Frontend logic
│   ├── styles.css            # Premium styling
│   ├── nginx.conf            # Server config
│   └── Dockerfile
├── docker-compose.yml        # Orchestration
├── requirements.txt          # Python dependencies
├── setup.ps1                 # Windows setup script
├── setup.sh                  # Linux/Mac setup script
└── README.md
```

---

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
API_BASE_URL=http://localhost:8000
FRONTEND_PORT=3000
BACKEND_PORT=8000
DEBUG=false
```

### Docker Ports

- Frontend: `3000` (configurable in `docker-compose.yml`)
- Backend: `8000` (configurable in `docker-compose.yml`)

---

## 📊 Sample Datasets

Sample CSV files are available in the `samples/` directory:

- `classification_iris.csv` - Classification example (Iris dataset)
- `regression_housing.csv` - Regression example (House prices)
- `clustering_customers.csv` - Clustering example (Customer segmentation)

---

## 🐞 Troubleshooting

### Docker Issues

**Problem:** "Docker is not running"
```bash
# Windows: Start Docker Desktop
# Linux: sudo systemctl start docker
```

**Problem:** Port already in use
```bash
# Change ports in docker-compose.yml
# Or stop conflicting services
```

### Build Failures

```bash
# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Backend Errors

```bash
# View logs
docker-compose logs backend

# Access container
docker exec -it ml-backend bash
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is open for personal, educational, and experimental use.

---

## 👤 Author

**dev-edward-ai**
- GitHub: [@dev-edward-ai](https://github.com/dev-edward-ai)

---

## 🙏 Acknowledgments

- Built with scikit-learn, XGBoost, FastAPI
- UI inspired by modern design systems
- ML explanations crafted for clarity and accessibility

---

**Ready to transform your data into insights? Get started in 30 seconds!** 🚀

```bash
# Windows
.\setup.ps1

# Linux/Mac
./setup.sh
```

Then visit **http://localhost:3000** and upload your first CSV!
