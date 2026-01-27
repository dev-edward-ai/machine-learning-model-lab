"""
Real-World Model Explanations System

Provides context-aware, easy-to-understand explanations for ML models
using real-world analogies and scenarios.
"""

from typing import Dict, Any, Optional


def get_model_explanation(
    model_name: str,
    task_type: str,
    dataset_context: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Get real-world explanation for a given model.
    
    Args:
        model_name: Name of the ML model
        task_type: Type of ML task (classification, regression, clustering, anomaly)
        dataset_context: Optional context about the dataset (columns, target, etc.)
    
    Returns:
        Dictionary with 'analogy', 'how_it_works', and 'real_world_example'
    """
    
    # Normalize model name for matching
    model_key = model_name.lower().replace(" ", "_")
    
    # Model explanation templates
    explanations = {
        # K-Nearest Neighbors (KNN)
        "knn": {
            "analogy": "🏘️ **The Neighborhood Analogy**",
            "how_it_works": (
                "KNN is like determining who you are based on your neighbors. "
                "If you speak Thai and your closest neighbors are Thai, you're probably Thai! "
                "The algorithm looks at the K closest data points and makes a decision based on what's most common among them."
            ),
            "real_world_example": (
                "**Airbnb Nightly Rate Estimator:** How much should I charge for my room? "
                "KNN finds the 5 nearest apartments physically close to you with similar amenities (WiFi, Pool, Bedrooms). "
                "If they charge [$145, $152, $148, $155, $150], your optimal rate = $150/night (average of neighbors). "
                "It's like asking 'What do similar places nearby charge?' Also used for: product recommendations, real estate pricing."
            ),
            "best_for": "Neighbor-based pricing, recommendation systems, pattern recognition, local decision-making"
        },
        
        # Linear Regression
    "linear_regression": {
        "analogy": "📈 **The Straight Line Predictor**",
        "how_it_works": (
            "Linear regression draws the best straight line through your data points. "
            "It assumes there's a linear relationship between inputs and output. "
            "Think of it as finding the trend line in your data."
        ),
        "real_world_example": (
            "**Marketing Ad ROI Calculator:** If we spend $5,000 more on YouTube ads, how many more sales? "
            "Linear regression finds: Every $1,000 spent = $4,200 in sales (straight-line relationship). "
            "So $5K ad spend → $21K sales! Marketing teams use this to estimate ROI: 'Spend $10K → Get $42K back'. "
            "Simple, interpretable, and fast. Also used for: salary prediction, temperature forecasting, stock trend analysis."
        ),
        "best_for": "Continuous predictions with linear relationships, ROI estimation, trend analysis, simple forecasting"
    },
        
        # Logistic Regression
    "logistic_regression": {
        "analogy": "🎯 **The Yes/No Decision Maker**",
        "how_it_works": (
            "Logistic regression is linear regression's cousin for yes/no decisions. "
            "Instead of predicting a number, it predicts the probability of something being true or false. "
            "It draws an S-shaped curve to separate two classes."
        ),
        "real_world_example": (
            "**Crypto Buy/Sell Signal:** Using technical indicators (RSI, MACD, Moving Averages), "
            "it calculates the probability of price going UP in the next hour. Example: RSI=68, MA_7>MA_30, MACD positive → "
            "85% chance of uptrend → BUY signal! Financial analysts use this for quick probability assessments. "
            "Also used for: Email spam detection, medical diagnosis (disease: yes/no), credit approval (approve/reject)."
        ),
        "best_for": "Binary classification, probability estimation, risk assessment, trading signals"
    },
        
        # Decision Tree
    "decision_tree": {
        "analogy": "🌳 **The 20 Questions Game**",
        "how_it_works": (
            "A decision tree asks a series of yes/no questions to reach a conclusion. "
            "Like playing 20 questions, it splits your data based on the most informative features first, "
            "creating a tree of decisions that leads to a prediction."
        ),
        "real_world_example": (
            "**Loan Approval Assistant:** Banks need to explain rejections. The tree shows: "
            "'Income < $30K?' → Yes → 'Credit Score < 600?' → Yes → REJECTED. Customer sees exactly WHY they were declined! "
            "For cars: 'Mileage > 100K?' → Yes → 'Warranty expired?' → Yes → Price drops 40%. "
            "Decision trees are interpretable - you can visualize the exact rules, making them perfect for regulated industries."
        ),
        "best_for": "Interpretable decisions, rule-based classification, regulatory compliance, explaining rejections"
    },
        
        # Random Forest
    "random_forest": {
        "analogy": "🌲🌲🌲 **Wisdom of the Crowd**",
        "how_it_works": (
            "Random Forest creates hundreds of decision trees, each trained on slightly different data. "
            "Each tree 'votes' on the answer, and the majority wins. "
            "It's like asking 100 experts instead of 1 - more reliable and less prone to overfitting."
        ),
        "real_world_example": (
            "**Disease Risk Predictor:** Does this patient have heart disease? Instead of one doctor's opinion, "
            "imagine 100 doctors reviewing age, cholesterol, BP, ECG - each with slightly different specialties. "
            "If 85 say 'HIGH RISK' and 15 say 'LOW RISK', diagnosis = HIGH RISK with 85% confidence! "
            "Medical fields love Random Forest because it handles messy data well and is more accurate than single trees. "
            "Also used for: loan approval, fraud detection, flight delays."
        ),
        "best_for": "High accuracy, robust predictions, medical diagnosis, handling complex non-linear relationships"
    },
        
        # Support Vector Machine (SVM)
    "svm": {
        "analogy": "✂️ **The Optimal Separator**",
        "how_it_works": (
            "SVM finds the best boundary to separate different classes. "
            "Imagine drawing a line between red and blue marbles - SVM finds the line that maximizes "
            "the distance to the nearest marbles on both sides, creating the safest separation zone."
        ),
        "real_world_example": (
            "**Fake Banknote Detector:** Using sensor data (variance, skewness, curtosis, entropy) from scanned bills, "
            "SVM draws the maximum-margin boundary between REAL and FAKE. It's not just 'close to the line' - "
            "SVM creates a safety zone! Perfect for high-stakes binary decisions where you need confidence. "
            "Banks use this in ATMs to reject counterfeits instantly. Also great for: image recognition, cancer detection, text classification."
        ),
        "best_for": "High-dimensional data, clear class separation, robust to outliers, precision boundary detection"
    },
        
        # K-Means Clustering
        "kmeans": {
            "analogy": "📦 **The Automatic Organizer**",
            "how_it_works": (
                "K-Means groups similar items together without being told what the groups should be. "
                "It's like organizing a messy closet - items naturally group by type (shirts, pants, shoes) "
                "based on their similarities, creating K distinct clusters."
            ),
            "real_world_example": (
                "**Image Color Palette Generator:** Upload a sunset photo with millions of pixels → "
                "K-Means groups pixels into 5 dominant colors. RGB(255,102,102)='Coral Red', RGB(52,152,219)='Sky Blue', etc. "
                "Designers use this to extract color schemes! Input: complex image. Output: 5 hex codes for your brand palette. "
                "Also used for: customer segmentation, document clustering, image compression."
            ),
            "best_for": "Color extraction, customer segmentation, pattern discovery, data compression, unsupervised grouping"
        },
        
        # Naive Bayes
    "naive_bayes": {
        "analogy": "🎲 **The Probability Detective**",
        "how_it_works": (
            "Naive Bayes uses probability to make predictions. It calculates how likely something belongs "
            "to a class based on its features, assuming features are independent (hence 'naive'). "
            "It's based on Bayes' theorem - updating beliefs based on new evidence."
        ),
        "real_world_example": (
            "**Spam vs Ham SMS Detector:** Message contains 'FREE', 'WINNER', '!!!', and URL → "
            "P(Spam | these words) = 95%! This is literally how early email spam filters worked. "
            "Word 'FREE' appears in 80% of spam aber only 5% of ham. Combines all word probabilities to classify. "
            "Super fast and works great with text! Also used for: sentiment analysis, medical diagnosis, document classification."
        ),
        "best_for": "Text classification, spam filtering, fast predictions with limited data, NLP tasks"
    },
        
        # XGBoost (Gradient Boosting)
    "xgboost": {
        "analogy": "🎯 **The Error-Correcting Champion**",
        "how_it_works": (
            "XGBoost builds models sequentially, where each new model focuses on correcting the mistakes "
            "of previous models. Like a student learning from exam mistakes, each iteration gets smarter "
            "by targeting what it got wrong before. It's one of the most powerful ML algorithms."
        ),
        "real_world_example": (
            "**Customer Churn Predictor:** Predicting who will cancel Netflix/Spotify subscription. "
            "First model: 'Low tenure = churn' (70% accuracy). Second model finds 'Wait, users with high support calls ALSO churn!' "
            "Third model: 'Actually, paperless billing + monthly contract combo is risky!' After 100 iterations → 95% accuracy! "
            "This is the industry standard for tabular data. Companies use it to identify at-risk customers and send coupons BEFORE they quit. "
            "Why it dominates: Kaggle competitions, fraud detection, click prediction, recommendation systems."
        ),
        "best_for": "Winning competitions, customer churn, maximum accuracy on tabular data, handling complex patterns"
    },
        
        # PCA (Dimensionality Reduction / Gradient Boosting)
        "pca": {
            "analogy": "🗜️ **The Information Compressor**",
            "how_it_works": (
                "PCA reduces complex data to its essential components, like creating a highlight reel from a full movie. "
                "It finds the most important 'directions' in your data and keeps those while discarding noise. "
                "Useful for visualization and preprocessing before other algorithms."
            ),
            "real_world_example": (
                "**Stock Market Sector Visualizer:** 500 stocks with 50+ features (PE ratio, volatility, beta, growth, etc.). "
                "Impossible to visualize! PCA compresses to 2D: X-axis='Value vs Growth', Y-axis='Risk Level'. "
                "Now you can plot all 500 stocks on one chart! Tech stocks (NVIDIA, AMD) cluster together, consumer stocks (Coca-Cola) far away. "
                "Perfect for identifying sector trends and correlations. Also used for: image compression, genetics, noise reduction."
            ),
            "best_for": "Stock visualization, data visualization, noise reduction, preprocessing, handling high-dimensional data"
        },
        
        # Gradient Boosting (general)
        "gradient_boosting": {
            "analogy": "📚 **The Incremental Learner**",
            "how_it_works": (
                "Gradient Boosting is like studying for an exam by focusing on your weak areas. "
                "Each new model focuses on the errors made by previous models, gradually improving performance. "
                "It combines weak learners into a strong ensemble."
            ),
            "real_world_example": (
                "**Real-World Scenario:** Credit scoring. First model uses basic features (income, age) and makes rough predictions. "
                "Second model focuses on cases where the first model was uncertain, using payment history. "
                "Third model tackles remaining difficult cases using employment stability. "
                "Each step refines the score, resulting in highly accurate credit risk assessment used by banks worldwide."
            ),
            "best_for": "High-stakes predictions, complex non-linear patterns, tabular data"
        },
        
        # Isolation Forest (Anomaly Detection)
        "isolation_forest": {
            "analogy": "🔍 **The Outlier Hunter**",
            "how_it_works": (
                "Isolation Forest identifies anomalies by how 'easy' they are to isolate from normal data. "
                "Outliers are 'few and different' - they get isolated quickly with fewer random splits. "
                "Think of it as finding the one person wearing a tuxedo at a beach party - they stand out!"
            ),
            "real_world_example": (
                "**Credit Card Fraud Detection:** You usually buy groceries ($50-$100) locally. "
                "Suddenly: $3,500 electronics purchase from Thailand at 3 AM! Isolation Forest flags this: "
                "'This transaction looks WEIRD compared to your history.' Banks freeze the card instantly. "
                "It detects: unusual amounts, strange locations, odd times, suspicious merchant categories. "
                "Also used for: network intrusion detection, quality control, outlier removal."
            ),
            "best_for": "Fraud detection, anomaly detection, outlier removal, network security, quality control"
        },
    }
    
    # Get base explanation or default
    base_key = None
    for key in ["knn", "linear_regression", "logistic_regression", "decision_tree", 
                "random_forest", "svm", "kmeans", "naive_bayes", "xgboost", "pca", "gradient_boosting", "isolation_forest"]:
        if key in model_key:
            base_key = key
            break
    
    if not base_key:
        # Default explanation for unknown models
        return {
            "analogy": "🤖 **Advanced Machine Learning**",
            "how_it_works": f"The {model_name} algorithm analyzes patterns in your data to make predictions.",
            "real_world_example": f"This model is being used to analyze your dataset and provide insights based on detected patterns.",
            "best_for": "Data analysis and pattern recognition"
        }
    
    explanation = explanations[base_key].copy()
    
    # Add task-specific context
    if task_type == "classification":
        explanation["task_context"] = "🏷️ Classification: Predicting which category each item belongs to."
    elif task_type == "regression":
        explanation["task_context"] = "📊 Regression: Predicting a numerical value for each item."
    elif task_type == "clustering":
        explanation["task_context"] = "🎨 Clustering: Discovering natural groups in your data without predefined labels."
    elif task_type == "anomaly":
        explanation["task_context"] = "⚠️ Anomaly Detection: Identifying unusual or outlier data points."
    
    return explanation


def get_all_models_overview() -> Dict[str, Dict[str, str]]:
    """
    Get overview of all available models.
    
    Returns:
        Dictionary mapping model categories to their descriptions
    """
    return {
        "Supervised - Classification": {
            "models": ["Logistic Regression", "Decision Tree", "KNN", "SVM", "Random Forest", "Naive Bayes", "XGBoost"],
            "use_case": "When you know the categories and want to predict which category new data belongs to",
            "examples": "Spam detection, disease diagnosis, customer churn, sentiment analysis"
        },
        "Supervised - Regression": {
            "models": ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
            "use_case": "When you want to predict a continuous number",
            "examples": "House prices, sales forecasting, temperature prediction, stock prices"
        },
        "Unsupervised - Clustering": {
            "models": ["K-Means"],
            "use_case": "When you want to find natural groups in your data without predefined labels",
            "examples": "Customer segmentation, document clustering, gene sequencing, image compression"
        },
        "Unsupervised - Dimensionality Reduction": {
            "models": ["PCA"],
            "use_case": "When you want to reduce data complexity while preserving important information",
            "examples": "Data visualization, noise reduction, feature extraction, preprocessing"
        },
        "Unsupervised - Anomaly Detection": {
            "models": ["Isolation Forest"],
            "use_case": "When you want to identify unusual or suspicious data points",
            "examples": "Fraud detection, network intrusion, quality control, outlier detection"
        }
    }


def format_explanation_for_ui(explanation: Dict[str, str]) -> str:
    """
    Format explanation dictionary as HTML-friendly string for frontend display.
    
    Args:
        explanation: Dictionary from get_model_explanation
    
    Returns:
        HTML-formatted string
    """
    parts = []
    
    if "task_context" in explanation:
        parts.append(f"<div class='task-context'>{explanation['task_context']}</div>")
    
    if "analogy" in explanation:
        parts.append(f"<div class='analogy'>{explanation['analogy']}</div>")
    
    if "how_it_works" in explanation:
        parts.append(f"<div class='how-it-works'><strong>How it works:</strong> {explanation['how_it_works']}</div>")
    
    if "real_world_example" in explanation:
        parts.append(f"<div class='real-world-example'>{explanation['real_world_example']}</div>")
    
    if "best_for" in explanation:
        parts.append(f"<div class='best-for'><strong>Best for:</strong> {explanation['best_for']}</div>")
    
    return "\n".join(parts)
