# 🧠 InsightAI - Business Intelligence Platform

<div align="center">

![InsightAI Banner](https://img.shields.io/badge/InsightAI-ML%20Platform-8b5cf6?style=for-the-badge&logo=python&logoColor=white)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-machine--learning--model--lab.onrender.com-10b981?style=for-the-badge)](https://machine-learning-model-lab.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-f7931e?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**An interactive machine learning platform demonstrating real-world business scenarios with specialized ML models.**

[Live Demo](https://machine-learning-model-lab.onrender.com) • [Documentation](#-documentation) • [Getting Started](#-quick-start)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo Scenarios](#-demo-scenarios)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**InsightAI** is a full-stack machine learning web application that demonstrates how different ML algorithms can be applied to solve real-world business problems. The platform features four interactive scenarios, each powered by a specialized machine learning model.

Users can interact with pre-loaded sample data, adjust input parameters, and receive instant AI-powered predictions with confidence scores.

---

## ✨ Features

- 🎨 **Modern Dark UI** - Professional glassmorphism design with smooth animations
- 🤖 **4 ML Models** - Each scenario uses a different, purpose-built algorithm
- 📊 **Auto-loaded Data** - Sample datasets load automatically for instant demos
- ⚡ **Real-time Predictions** - Instant results as you adjust parameters
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile
- 🔗 **Educational Links** - Each model links to detailed explanations
- 🐳 **Docker Ready** - Containerized for easy deployment
- ☁️ **Cloud Deployed** - Live on Render's free tier

---

## 🎪 Demo Scenarios

| Scenario | Model | Use Case | Type |
|----------|-------|----------|------|
| 🚀 **Crypto Trading** | Logistic Regression | Buy/Sell signal prediction using technical indicators | Classification |
| 🏥 **Medical Diagnosis** | K-Nearest Neighbors | Health risk assessment based on patient vitals | Classification |
| 🚗 **Vehicle Pricing** | Decision Tree Regressor | Market valuation based on car specifications | Regression |
| 💰 **Sales Forecasting** | Random Forest Regressor | ROI prediction for marketing campaigns | Regression |

### Model Deep Dives

Each scenario links to detailed blog posts explaining the underlying algorithms:

- [Logic Behind Logistic Regression](https://novaz-edd.hashnode.dev/logic-behind-logistic-regression)
- [Logic Behind KNN](https://novaz-edd.hashnode.dev/logic-behind-knn)
- [Decision Tree Regressor](https://novaz-edd.hashnode.dev/decision-tree-regressor)
- [Random Forest Regressor](https://novaz-edd.hashnode.dev/random-forest-regressor)

---

## 🛠 Tech Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **Uvicorn** - ASGI server for production
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning algorithms

### Frontend
- **HTML5 / CSS3** - Modern semantic markup
- **Vanilla JavaScript** - No framework dependencies
- **Font Awesome** - Icon library
- **Google Fonts (Inter)** - Typography

### DevOps
- **Docker** - Containerization
- **Render** - Cloud deployment (Free tier)
- **GitHub** - Version control

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/Novaz-Edd/machine-learning-model-lab.git
   cd machine-learning-model-lab
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the backend server**
   ```bash
   cd backend
   uvicorn api.main:app --reload --port 8000
   ```

5. **Serve the frontend** (in a new terminal)
   ```bash
   # From project root
   python -m http.server 3000
   ```

6. **Open in browser**
   ```
   http://localhost:3000/frontend/index.html
   ```

### Using Docker

```bash
# Build and run
docker build -t insightai .
docker run -p 8000:8000 insightai

# Open in browser
http://localhost:8000
```

---

## 📁 Project Structure

```
machine-learning-model-lab/
├── 📂 backend/
│   ├── 📂 api/
│   │   ├── main.py              # FastAPI application entry
│   │   ├── routers/             # API route handlers
│   │   ├── schemas/             # Pydantic models
│   │   └── services/            # Business logic
│   ├── logic.py                 # ML model training & prediction
│   └── Dockerfile               # Backend container config
│
├── 📂 frontend/
│   ├── index.html               # Main HTML (all pages)
│   ├── styles.css               # CSS styles (dark theme)
│   ├── script.js                # JavaScript logic
│   └── Dockerfile               # Frontend container config
│
├── 📂 samples/                  # Sample CSV datasets
│   ├── crypto_signals.csv
│   ├── heart_disease.csv
│   ├── used_car_prices.csv
│   └── marketing_roi.csv
│
├── Dockerfile                   # Production Dockerfile
├── docker-compose.yml           # Multi-container setup
├── render.yaml                  # Render deployment config
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📡 API Documentation

### Base URL
- **Local:** `http://localhost:8000`
- **Production:** `https://machine-learning-model-lab.onrender.com`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend |
| `GET` | `/ping` | Health check |
| `POST` | `/analyze` | Upload CSV & train model |
| `POST` | `/predict/crypto` | Crypto trading prediction |
| `POST` | `/predict/medical` | Medical risk prediction |
| `POST` | `/predict/car_price` | Vehicle price estimation |
| `POST` | `/predict/sales` | Sales/ROI forecast |

### Example Request

```bash
curl -X POST "https://machine-learning-model-lab.onrender.com/predict/crypto" \
  -H "Content-Type: application/json" \
  -d '{"open": 65000, "close": 65500, "volume": 1000000, "rsi": 55}'
```

### Example Response

```json
{
  "signal": "BUY",
  "confidence": 0.87,
  "recommendation": "Strong bullish momentum detected"
}
```

---

## ☁️ Deployment

### Render (Recommended)

1. Fork this repository
2. Go to [render.com](https://render.com) and create a new Web Service
3. Connect your GitHub repository
4. Select **Docker** as the runtime
5. Choose the **Free** instance type
6. Deploy!

### Environment Variables

No additional environment variables are required. The app automatically uses Render's `PORT` variable.

---

## 📚 Documentation

### How It Works

1. **User selects a scenario** from the dashboard
2. **Sample CSV auto-loads** and is sent to the backend
3. **Backend detects scenario** based on column names
4. **Model trains** on the sample data
5. **User adjusts parameters** using sliders/inputs
6. **Real-time predictions** are returned with confidence scores

### Scenario Detection

The backend automatically detects which scenario to use based on CSV column patterns:

- **Crypto:** `rsi`, `macd`, `volume`, `buy_signal`
- **Medical:** `age`, `blood_pressure`, `glucose`, `bmi`
- **Car:** `year`, `mileage`, `horsepower`, `price`
- **Sales:** `ad_spend`, `clicks`, `impressions`, `revenue`

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Novaz-Edd**

- GitHub: [@Novaz-Edd](https://github.com/Novaz-Edd)
- Hashnode: [novaz-edd.hashnode.dev](https://novaz-edd.hashnode.dev)

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Made with ❤️ and Python

</div>
