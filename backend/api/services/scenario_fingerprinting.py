"""
Scenario Fingerprinting Engine

Implements domain logic to detect scenarios BEFORE training models,
ensuring proper model selection based on business context rather than raw accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re


class ScenarioFingerprinting:
    """
    Advanced scenario detection using domain-specific heuristics and column pattern matching.
    Prevents lazy KNN selection by enforcing business logic.
    """
    
    def __init__(self):
        self.scenario_fingerprints = {
            "crypto_signals": {
                "column_patterns": [
                    r"(open|close|high|low|volume)",
                    r"(price|rsi|macd|moving.*average)",
                    r"(signal|buy|sell|trade)"
                ],
                "preferred_models": ["Logistic Regression", "Random Forest Classifier"],
                "data_characteristics": {
                    "target_type": "binary",
                    "feature_types": ["numeric"],
                    "domain_indicators": ["financial", "trading", "crypto"]
                },
                "confidence_boost": 0.15,  # 15% confidence boost for correct models
                "ui_component": "TradingTerminal"
            },
            
            "loan_applications": {
                "column_patterns": [
                    r"(income|salary|credit.*score|employment)",
                    r"(loan|debt|approved|amount)",
                    r"(years.*employed|tenure|experience)"
                ],
                "preferred_models": ["Decision Tree Classifier", "Random Forest Classifier"],
                "data_characteristics": {
                    "target_type": "binary",
                    "feature_types": ["numeric", "categorical"],
                    "domain_indicators": ["financial", "banking", "loan"]
                },
                "confidence_boost": 0.20,  # Decision trees preferred for interpretability
                "ui_component": "LoanOfficerDashboard"
            },
            
            "sms_spam": {
                "column_patterns": [
                    r"(word.*count|text|message|content)",
                    r"(spam|ham|special.*char|url)",
                    r"(caps|uppercase|urgency)"
                ],
                "preferred_models": ["Naive Bayes", "Logistic Regression"],
                "data_characteristics": {
                    "target_type": "binary",
                    "feature_types": ["numeric", "text_derived"],
                    "domain_indicators": ["text", "nlp", "communication"]
                },
                "confidence_boost": 0.25,  # Naive Bayes excels at text classification
                "ui_component": "SpamDetector"
            },
            
            "heart_disease": {
                "column_patterns": [
                    r"(age|cholesterol|blood.*pressure)",
                    r"(heart|chest.*pain|ecg|cardiac)",
                    r"(disease|condition|diagnosis)"
                ],
                "preferred_models": ["Random Forest Classifier", "XGBoost Classifier"],
                "data_characteristics": {
                    "target_type": "binary",
                    "feature_types": ["numeric"],
                    "domain_indicators": ["medical", "healthcare", "diagnosis"]
                },
                "confidence_boost": 0.18,
                "ui_component": "MedicalDashboard"
            },
            
            "customer_segments": {
                "column_patterns": [
                    r"(customer|user|client)",
                    r"(segment|group|cluster|category)",
                    r"(behavior|purchase|activity)"
                ],
                "preferred_models": ["KMeans", "PCA"],
                "data_characteristics": {
                    "target_type": None,  # Unsupervised
                    "feature_types": ["numeric"],
                    "domain_indicators": ["customer", "segmentation", "marketing"]
                },
                "confidence_boost": 0.30,  # Strong preference for clustering
                "ui_component": "ClusteringVisualization"
            },
            
            "price_prediction": {
                "column_patterns": [
                    r"(price|cost|value|amount)",
                    r"(year|age|mileage|size)",
                    r"(brand|model|type|category)"
                ],
                "preferred_models": ["Random Forest Regressor", "XGBoost Regressor"],
                "data_characteristics": {
                    "target_type": "continuous",
                    "feature_types": ["numeric", "categorical"],
                    "domain_indicators": ["pricing", "valuation", "prediction"]
                },
                "confidence_boost": 0.15,
                "ui_component": "PricingCalculator"
            }
        }
    
    def detect_scenario(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect the most likely scenario based on column patterns and data characteristics.
        Returns scenario with confidence score and recommended models.
        """
        column_names = [col.lower() for col in df.columns]
        
        scenario_scores = {}
        
        for scenario_id, fingerprint in self.scenario_fingerprints.items():
            score = self._calculate_scenario_score(df, column_names, target_col, fingerprint)
            scenario_scores[scenario_id] = score
        
        # Find best matching scenario
        best_scenario_id = max(scenario_scores, key=scenario_scores.get)
        best_score = scenario_scores[best_scenario_id]
        
        if best_score < 0.3:  # Low confidence, use general approach
            return {
                "scenario_id": "general",
                "scenario_name": "General Analysis",
                "confidence": best_score,
                "preferred_models": [],
                "ui_component": "GeneralResults",
                "reasoning": "No strong scenario match detected"
            }
        
        best_fingerprint = self.scenario_fingerprints[best_scenario_id]
        
        return {
            "scenario_id": best_scenario_id,
            "scenario_name": best_fingerprint.get("name", best_scenario_id.replace("_", " ").title()),
            "confidence": best_score,
            "preferred_models": best_fingerprint["preferred_models"],
            "confidence_boost": best_fingerprint["confidence_boost"],
            "ui_component": best_fingerprint["ui_component"],
            "reasoning": self._generate_reasoning(column_names, best_fingerprint)
        }
    
    def _calculate_scenario_score(self, df: pd.DataFrame, column_names: List[str], 
                                 target_col: Optional[str], fingerprint: Dict) -> float:
        """Calculate confidence score for a specific scenario based on multiple factors."""
        
        score = 0.0
        max_score = 1.0
        
        # 1. Column pattern matching (40% weight)
        pattern_score = self._score_column_patterns(column_names, fingerprint["column_patterns"])
        score += pattern_score * 0.4
        
        # 2. Target variable characteristics (30% weight)
        if target_col and target_col in df.columns:
            target_score = self._score_target_characteristics(df[target_col], fingerprint["data_characteristics"])
            score += target_score * 0.3
        else:
            # Unsupervised learning scenarios
            if fingerprint["data_characteristics"]["target_type"] is None:
                score += 0.3  # Boost for clustering scenarios
        
        # 3. Feature type distribution (20% weight)
        feature_score = self._score_feature_types(df, fingerprint["data_characteristics"]["feature_types"])
        score += feature_score * 0.2
        
        # 4. Data size and complexity (10% weight)
        complexity_score = self._score_data_complexity(df)
        score += complexity_score * 0.1
        
        return min(score, max_score)
    
    def _score_column_patterns(self, column_names: List[str], patterns: List[str]) -> float:
        """Score based on how well column names match expected patterns."""
        if not patterns:
            return 0.0
            
        matches = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            if any(regex.search(col) for col in column_names):
                matches += 1
        
        return matches / total_patterns
    
    def _score_target_characteristics(self, target_series: pd.Series, characteristics: Dict) -> float:
        """Score based on target variable characteristics."""
        expected_type = characteristics.get("target_type")
        
        if expected_type == "binary":
            unique_values = target_series.nunique()
            if unique_values == 2:
                return 1.0
            elif unique_values <= 5:
                return 0.6
            else:
                return 0.2
                
        elif expected_type == "continuous":
            if target_series.dtype in ['float64', 'int64'] and target_series.nunique() > 10:
                return 1.0
            else:
                return 0.3
                
        elif expected_type is None:  # Unsupervised
            return 1.0  # No target needed
            
        return 0.0
    
    def _score_feature_types(self, df: pd.DataFrame, expected_types: List[str]) -> float:
        """Score based on feature type distribution."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        numeric_ratio = len(numeric_cols) / len(df.columns)
        categorical_ratio = len(categorical_cols) / len(df.columns)
        
        score = 0.0
        
        if "numeric" in expected_types:
            score += numeric_ratio * 0.5
        if "categorical" in expected_types:
            score += categorical_ratio * 0.5
        if "text_derived" in expected_types:
            # Look for text-like numeric features (word counts, etc.)
            text_indicators = ["count", "length", "chars", "words"]
            text_like_cols = [col for col in df.columns 
                             if any(indicator in col.lower() for indicator in text_indicators)]
            if text_like_cols:
                score += 0.3
        
        return min(score, 1.0)
    
    def _score_data_complexity(self, df: pd.DataFrame) -> float:
        """Score based on data size and complexity."""
        n_rows, n_cols = df.shape
        
        # Prefer certain models for certain data sizes
        complexity_score = 0.5  # Base score
        
        if n_rows > 10000:
            complexity_score += 0.2  # Favor ensemble methods
        if n_cols > 20:
            complexity_score += 0.2  # Favor feature-robust models
        if df.isnull().sum().sum() == 0:
            complexity_score += 0.1  # Clean data bonus
            
        return min(complexity_score, 1.0)
    
    def _generate_reasoning(self, column_names: List[str], fingerprint: Dict) -> str:
        """Generate human-readable reasoning for scenario detection."""
        
        matched_patterns = []
        for pattern in fingerprint["column_patterns"]:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = [col for col in column_names if regex.search(col)]
            if matches:
                matched_patterns.append(f"Found {pattern} patterns in: {matches[:3]}")
        
        reasoning = f"Detected based on: {'; '.join(matched_patterns[:2])}"
        preferred = ", ".join(fingerprint["preferred_models"][:2])
        reasoning += f". Recommending {preferred} for this domain."
        
        return reasoning


class HeuristicScorer:
    """
    Applies domain-specific scoring adjustments to raw model performance,
    ensuring business logic takes precedence over pure accuracy metrics.
    """
    
    def __init__(self, fingerprinting_engine: ScenarioFingerprinting):
        self.fingerprinting = fingerprinting_engine
    
    def adjust_model_scores(self, raw_scores: List[Dict], scenario_info: Dict) -> List[Dict]:
        """
        Apply heuristic adjustments to model scores based on detected scenario.
        
        Args:
            raw_scores: List of {"name": str, "score": float, "model": object}
            scenario_info: Scenario detection results from fingerprinting
            
        Returns:
            Adjusted scores with domain logic applied
        """
        
        if scenario_info["scenario_id"] == "general":
            return raw_scores  # No adjustments for general scenarios
        
        preferred_models = scenario_info.get("preferred_models", [])
        confidence_boost = scenario_info.get("confidence_boost", 0.0)
        
        adjusted_scores = []
        
        for model_result in raw_scores:
            model_name = model_result["name"]
            raw_score = model_result["score"]
            
            # Apply boost to preferred models
            if model_name in preferred_models:
                # Add confidence boost (but cap at 100%)
                adjusted_score = min(raw_score + (confidence_boost * 100), 100.0)
                boost_applied = True
            else:
                # Slight penalty for non-preferred models in strong scenario matches
                if scenario_info["confidence"] > 0.7:
                    adjusted_score = max(raw_score - 5.0, 0.0)  # 5% penalty
                else:
                    adjusted_score = raw_score
                boost_applied = False
            
            adjusted_result = model_result.copy()
            adjusted_result["score"] = adjusted_score
            adjusted_result["raw_score"] = raw_score
            adjusted_result["domain_boost_applied"] = boost_applied
            
            adjusted_scores.append(adjusted_result)
        
        # Re-sort by adjusted scores
        return sorted(adjusted_scores, key=lambda x: x["score"], reverse=True)
    
    def explain_adjustments(self, model_name: str, scenario_info: Dict, 
                          raw_score: float, adjusted_score: float) -> str:
        """Generate explanation for score adjustments."""
        
        if abs(raw_score - adjusted_score) < 0.1:
            return "No domain adjustments applied."
        
        if adjusted_score > raw_score:
            boost = adjusted_score - raw_score
            return (f"Domain expertise boost: +{boost:.1f}% for {model_name} "
                   f"in {scenario_info['scenario_name']} scenarios. "
                   f"Reason: {scenario_info.get('reasoning', 'Domain best practice')}")
        else:
            penalty = raw_score - adjusted_score
            return (f"Domain penalty: -{penalty:.1f}% for {model_name}. "
                   f"Model less suitable for {scenario_info['scenario_name']} context.")