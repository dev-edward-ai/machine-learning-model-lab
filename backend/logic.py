"""
Business Insight Generator - Core Logic
4 Hardcoded Scenarios with Forced Models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
from typing import Tuple, Dict, Any


# ============================================================================
# SCENARIO DETECTION
# ============================================================================

def detect_scenario(df: pd.DataFrame) -> str:
    """
    Detect which of the 4 scenarios the CSV belongs to.
    Returns: 'CRYPTO', 'MEDICAL', 'CAR_PRICE', or 'SALES'
    """
    columns_lower = [col.lower() for col in df.columns]
    
    # CRYPTO: Volume, Open, Close, RSI, MACD, etc.
    crypto_indicators = {'open', 'close', 'volume', 'rsi', 'macd', 'high', 'low'}
    if crypto_indicators.intersection(set(columns_lower)):
        return 'CRYPTO'
    
    # MEDICAL: Age, BMI, BloodPressure, Glucose, Cholesterol
    medical_indicators = {'age', 'bmi', 'bloodpressure', 'glucose', 'cholesterol', 'insulin', 'pregnancies'}
    if medical_indicators.intersection(set(columns_lower)):
        return 'MEDICAL'
    
    # CAR_PRICE: Year, Mileage, Brand, Horsepower, Price
    car_indicators = {'year', 'mileage', 'brand', 'horsepower', 'price', 'miles', 'engine'}
    if car_indicators.intersection(set(columns_lower)):
        return 'CAR_PRICE'
    
    # SALES: AdSpend, SocialClicks, Season, CampaignType, Budget, Revenue
    sales_indicators = {'adspend', 'socialclicks', 'season', 'campaigntype', 'budget', 'revenue', 'clicks', 'spend'}
    if sales_indicators.intersection(set(columns_lower)):
        return 'SALES'
    
    # Default to SALES if ambiguous
    return 'SALES'


# ============================================================================
# TRACK 1: CRYPTO TRADING (LogisticRegression)
# ============================================================================

def prepare_crypto_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Prepare crypto data for LogisticRegression."""
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Select features - expanded to match actual CSV columns
    feature_keys = ['price', 'open', 'close', 'volume', 'rsi', 'macd', 'high', 'low', 
                    'moving_average_7', 'moving_average_30', 'signal_strength']
    feature_cols = [columns_lower.get(key) for key in feature_keys if key in columns_lower]
    feature_cols = [col for col in feature_cols if col is not None and col in df.columns][:6]  # Max 6 features
    
    # Target (assume last numeric column or 'buy_signal')
    target_col = None
    for col in ['buy_signal', 'signal', 'target', 'label']:
        if col.lower() in columns_lower:
            target_col = columns_lower[col.lower()]
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    y = df[target_col].values if target_col in df.columns else np.random.randint(0, 2, len(df))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols


def train_crypto_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Train LogisticRegression for crypto trading."""
    X, y, scaler, feature_cols = prepare_crypto_data(df)
    
    # Ensure binary classification
    unique_vals = np.unique(y)
    if len(unique_vals) > 2:
        y = (y > np.median(y)).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'feature_cols': feature_cols,
        'classes': ['SELL (0)', 'BUY (1)']
    }


# ============================================================================
# TRACK 2: DISEASE DIAGNOSIS (K-Nearest Neighbors)
# ============================================================================

def prepare_medical_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Prepare medical data for KNN."""
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Select features - expanded to match actual CSV columns (heart_disease.csv)
    feature_keys = ['age', 'bmi', 'bloodpressure', 'glucose', 'cholesterol', 'insulin', 'pregnancies',
                    'resting_bp', 'fasting_blood_sugar', 'max_heart_rate', 'chest_pain_type', 'sex']
    feature_cols = [columns_lower.get(key) for key in feature_keys if key in columns_lower]
    feature_cols = [col for col in feature_cols if col is not None and col in df.columns][:6]
    
    # Target
    target_col = None
    for col in ['outcome', 'target', 'diagnosis', 'disease', 'has_disease']:
        if col.lower() in columns_lower:
            target_col = columns_lower[col.lower()]
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    y = df[target_col].values if target_col in df.columns else np.random.randint(0, 2, len(df))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols


def train_medical_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Train KNN for disease diagnosis."""
    X, y, scaler, feature_cols = prepare_medical_data(df)
    
    # Ensure binary classification
    unique_vals = np.unique(y)
    if len(unique_vals) > 2:
        y = (y > np.median(y)).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'feature_cols': feature_cols,
        'classes': ['Low Risk', 'High Risk']
    }


# ============================================================================
# TRACK 3: USED CAR PRICING (DecisionTreeRegressor)
# ============================================================================

def prepare_car_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Prepare car data for DecisionTreeRegressor."""
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Select features - expanded to match actual CSV columns (used_car_prices.csv)
    feature_keys = ['year', 'mileage', 'horsepower', 'engine', 'miles', 'engine_size', 
                    'previous_owners', 'warranty_months', 'condition_score']
    feature_cols = [columns_lower.get(key) for key in feature_keys if key in columns_lower]
    feature_cols = [col for col in feature_cols if col is not None and col in df.columns][:5]
    
    # Target (price)
    target_col = None
    for col in ['price', 'value', 'cost']:
        if col.lower() in columns_lower:
            target_col = columns_lower[col.lower()]
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    y = df[target_col].values if target_col in df.columns else np.random.rand(len(df)) * 100000
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols


def train_car_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Train DecisionTreeRegressor for car pricing."""
    X, y, scaler, feature_cols = prepare_car_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    r2_score = model.score(X_test, y_test)
    
    return {
        'model': model,
        'scaler': scaler,
        'r2_score': r2_score,
        'feature_cols': feature_cols,
        'y_min': y.min(),
        'y_max': y.max()
    }


# ============================================================================
# TRACK 4: MARKETING/SALES PREDICTION (RandomForestRegressor)
# ============================================================================

def prepare_sales_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler, dict]:
    """Prepare sales data for RandomForestRegressor."""
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Prioritize specific numeric features from marketing_roi.csv
    priority_keys = ['ad_spend', 'adspend', 'clicks', 'impressions', 'audience_size', 'ctr']
    feature_cols = []
    for key in priority_keys:
        if key in columns_lower and columns_lower[key] in df.columns:
            col = columns_lower[key]
            if df[col].dtype in ['int64', 'float64']:
                feature_cols.append(col)
    
    # Fill remaining slots with other numeric columns (exclude target)
    for col in df.columns:
        if len(feature_cols) >= 6:
            break
        if df[col].dtype in ['int64', 'float64'] and col.lower() not in ['revenue', 'return', 'roi', 'profit', 'sales_generated', 'campaign_id']:
            if col not in feature_cols:
                feature_cols.append(col)
    
    feature_cols = feature_cols[:6]  # Max 6 features
    
    # Prepare X
    X = df[feature_cols].fillna(df[feature_cols].mean()).values
    
    # Target (revenue)
    target_col = None
    for col in ['sales_generated', 'revenue', 'return', 'roi', 'profit', 'sales']:
        if col.lower() in columns_lower:
            target_col = columns_lower[col.lower()]
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    y = df[target_col].values if target_col in df.columns else np.random.rand(len(df)) * 100000
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols


def train_sales_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Train RandomForestRegressor for sales/marketing prediction."""
    X, y, scaler, feature_cols = prepare_sales_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    r2_score = model.score(X_test, y_test)
    
    return {
        'model': model,
        'scaler': scaler,
        'r2_score': r2_score,
        'feature_cols': feature_cols,
        'y_min': y.min(),
        'y_max': y.max()
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_specific_model(df: pd.DataFrame, scenario: str) -> Dict[str, Any]:
    """
    Train the hardcoded model for the detected scenario.
    """
    if scenario == 'CRYPTO':
        result = train_crypto_model(df)
        result['scenario'] = 'CRYPTO'
    elif scenario == 'MEDICAL':
        result = train_medical_model(df)
        result['scenario'] = 'MEDICAL'
    elif scenario == 'CAR_PRICE':
        result = train_car_model(df)
        result['scenario'] = 'CAR_PRICE'
    elif scenario == 'SALES':
        result = train_sales_model(df)
        result['scenario'] = 'SALES'
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return result
