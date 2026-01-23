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
                "**Real-World Scenario:** Imagine you're living in the USA but speak Thai. "
                "KNN checks your 5 closest neighbors - if 4 of them are Thai and 1 is American, "
                "it predicts you're Thai! Same applies to product recommendations: if your 5 nearest "
                "customers (by behavior) all bought product X, you'll likely buy it too."
            ),
            "best_for": "Recommendation systems, pattern recognition, classification when similar items behave similarly"
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
                "**Real-World Scenario:** Predicting house prices based on square footage. "
                "If houses generally increase by $100 per square foot, linear regression finds that pattern. "
                "A 2,000 sq ft house might be $200K, a 3,000 sq ft house $300K - a straight line relationship. "
                "Also used for sales forecasting, temperature predictions, or any scenario where one thing consistently affects another."
            ),
            "best_for": "Continuous predictions with linear relationships, trend analysis, simple forecasting"
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
                "**Real-World Scenario:** Email spam detection. Based on features like 'number of exclamation marks', "
                "'contains word FREE', and 'sender reputation', it calculates the probability an email is spam. "
                "If probability > 50%, mark as spam. Also perfect for medical diagnosis (disease: yes/no), "
                "credit approval (approve/reject), or customer churn (will leave/will stay)."
            ),
            "best_for": "Binary classification, probability estimation, risk assessment"
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
                "**Real-World Scenario:** Diagnosing why your car won't start. "
                "First question: 'Does it make any sound?' → No → 'Is the battery dead?' → Yes → Replace battery! "
                "For business: 'Is customer tenure > 2 years?' → No → 'Did they call support 3+ times?' → Yes → High churn risk! "
                "The tree makes decisions just like a human troubleshooting guide, easy to understand and explain."
            ),
            "best_for": "Interpretable decisions, rule-based classification, handling mixed data types"
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
                "**Real-World Scenario:** Loan approval decision. Instead of one banker reviewing your application, "
                "imagine 100 bankers each reviewing it independently with slightly different information. "
                "If 85 approve and 15 reject, you get approved! This prevents one bad tree from making a wrong decision. "
                "Used in fraud detection, disease diagnosis, or any high-stakes decision where accuracy matters."
            ),
            "best_for": "High accuracy, robust predictions, handling complex non-linear relationships"
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
                "**Real-World Scenario:** Security clearance - separating authorized from unauthorized personnel. "
                "You want the clearest possible boundary with maximum margin of safety. "
                "SVM would find the decision boundary (combination of credentials, badges, access history) "
                "that best separates the two groups with maximum confidence. "
                "Perfect for image recognition (cat vs dog), text classification, or any scenario needing clear separation."
            ),
            "best_for": "High-dimensional data, clear class separation, robust to outliers"
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
                "**Real-World Scenario:** Customer segmentation for marketing. "
                "Without knowing customer types in advance, K-Means discovers natural groups: "
                "Group 1 = 'Budget shoppers' (low spending, frequent discount use), "
                "Group 2 = 'Premium buyers' (high spending, brand loyal), "
                "Group 3 = 'Occasional purchasers' (medium spending, seasonal). "
                "Now you can target each group with tailored campaigns! Also used for gene sequencing, image compression."
            ),
            "best_for": "Customer segmentation, pattern discovery, data compression, unsupervised grouping"
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
                "**Real-World Scenario:** Medical diagnosis. If you have symptoms = [fever, cough, fatigue], "
                "Naive Bayes calculates: P(Flu | symptoms) vs P(Cold | symptoms) vs P(COVID | symptoms). "
                "It knows 'fever' is 80% common in flu, 40% in cold, 90% in COVID, combines all symptom probabilities, "
                "and predicts the most likely disease. Fast and effective! Also powers spam filters and sentiment analysis."
            ),
            "best_for": "Text classification, spam filtering, fast predictions with limited data"
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
                "**Real-World Scenario:** Predicting insurance claims. First model predicts claim amounts but makes errors. "
                "Second model learns 'where did we overpredict? underpredict?' and corrects those patterns. "
                "Third model corrects what model 2 missed. After 100+ iterations, you have a super-accurate predictor! "
                "This is why XGBoost dominates Kaggle competitions and powers fraud detection, click prediction, and ranking systems."
            ),
            "best_for": "Winning competitions, complex patterns, maximum accuracy, handling messy data"
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
                "**Real-World Scenario:** You have customer data with 100 features (age, income, purchases, clicks, etc.). "
                "PCA might discover that 3 'super-features' capture 95% of the variation: "
                "'Purchasing Power' (combination of income, spending), 'Engagement' (clicks, time on site), "
                "and 'Loyalty' (repeat visits, reviews). Now you can visualize customers in 3D instead of 100D! "
                "Also used in image compression, genetics, and removing data redundancy."
            ),
            "best_for": "Data visualization, noise reduction, preprocessing, handling high-dimensional data"
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
    }
    
    # Get base explanation or default
    base_key = None
    for key in ["knn", "linear_regression", "logistic_regression", "decision_tree", 
                "random_forest", "svm", "kmeans", "naive_bayes", "xgboost", "pca", "gradient_boosting"]:
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
