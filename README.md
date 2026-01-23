# 🤖 AutoML Intelligence Platform

**Professional AutoML platform with automatic model detection, real-world explanations, and Docker deployment.**

Transform your CSV data into intelligent insights with zero configuration. Our platform automatically detects the best machine learning model for your data and explains it using real-world analogies.

![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![Python](https://img.shields.io/badge/Python-3.11-green?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-teal?logo=fastapi)
![ML Models](https://img.shields.io/badge/ML%20Models-10+-purple)

---

## ✨ Features

- **🎯 Automatic Model Detection** - Upload CSV, get instant analysis with the best-fit algorithm
- **💡 Real-World Explanations** - Every model comes with intuitive analogies (e.g., "KNN is like asking your neighbors")
- **🐳 Docker Ready** - One command deployment, works on any laptop
- **📊 10+ ML Algorithms** - Comprehensive model library covering all major use cases
- **🎨 Premium UI** - Modern dark theme with glassmorphism and smooth animations
- **⚡ Production Ready** - FastAPI backend, Nginx frontend, fully containerized

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

## 📚 Supported ML Models

Our platform includes **10+ machine learning algorithms**:

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
