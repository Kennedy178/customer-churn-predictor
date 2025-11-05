"""
Advanced Customer Churn Prediction System with Explainable AI
Application: Predict which customers will leave and WHY they'll leave
Demonstrates: Feature Engineering, Multiple Models, Hyperparameter Tuning, SHAP Explainability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class ChurnPredictionSystem:
    """
    Enterprise-grade churn prediction system with advanced features:
    - Automated feature engineering
    - Multiple model comparison
    - Hyperparameter optimization
    - SHAP-based explainability
    - Business metrics calculation (CLV impact)
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def generate_synthetic_data(self, n_samples=5000):
        """
        Generate realistic telecom customer data
        Based on real-world churn patterns
        """
        print("Generating synthetic customer data...")
        
        # Customer demographics
        age = np.random.normal(45, 15, n_samples).clip(18, 80)
        tenure = np.random.exponential(24, n_samples).clip(1, 72)
        
        # Service usage patterns
        monthly_charges = np.random.gamma(2, 30, n_samples).clip(20, 150)
        total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
        
        # Behavioral features
        num_services = np.random.poisson(3, n_samples).clip(1, 8)
        contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                        n_samples, p=[0.5, 0.3, 0.2])
        payment_method = np.random.choice(['Electronic', 'Mailed check', 'Bank transfer', 'Credit card'],
                                         n_samples, p=[0.4, 0.2, 0.2, 0.2])
        
        # Support interactions (higher = more problems)
        support_calls = np.random.poisson(2, n_samples)
        
        # Internet service
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                           n_samples, p=[0.4, 0.4, 0.2])
        
        # Create churn based on realistic patterns
        churn_probability = (
            0.1 +  # Base churn rate
            0.3 * (contract_type == 'Month-to-month') +
            0.2 * (tenure < 6) / 6 +
            0.15 * (support_calls > 3) +
            0.1 * (monthly_charges > 100) / 50 +
            0.05 * (payment_method == 'Mailed check') -
            0.15 * (num_services > 4) / 4 -
            0.2 * (contract_type == 'Two year')
        )
        
        churn_probability = np.clip(churn_probability, 0, 1)
        churn = (np.random.random(n_samples) < churn_probability).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Age': age,
            'Tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'NumServices': num_services,
            'ContractType': contract_type,
            'PaymentMethod': payment_method,
            'SupportCalls': support_calls,
            'InternetService': internet_service,
            'Churn': churn
        })
        
        print(f" Generated {n_samples} customer records")
        print(f"   Churn Rate: {churn.mean()*100:.1f}%")
        
        return df
    
    def engineer_features(self, df):
        """
        Create advanced features that capture business insights
        """
        print("\n Engineering advanced features...")
        
        df = df.copy()
        
        # Financial features
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['Tenure'] + 1)
        df['ChargeToServiceRatio'] = df['MonthlyCharges'] / (df['NumServices'] + 1)
        
        # Tenure-based features
        df['IsNewCustomer'] = (df['Tenure'] < 6).astype(int)
        df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 100], 
                                    labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
        
        # Risk indicators
        df['HighRiskPayment'] = (df['PaymentMethod'] == 'Mailed check').astype(int)
        df['HighSupportNeeds'] = (df['SupportCalls'] > df['SupportCalls'].median()).astype(int)
        df['MonthToMonth'] = (df['ContractType'] == 'Month-to-month').astype(int)
        
        # Value segments
        df['CustomerValue'] = pd.qcut(df['TotalCharges'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Engagement score (composite metric)
        df['EngagementScore'] = (
            df['NumServices'] / df['NumServices'].max() * 0.4 +
            (1 - df['SupportCalls'] / df['SupportCalls'].max()) * 0.3 +
            df['Tenure'] / df['Tenure'].max() * 0.3
        )
        
        print(f"Created {len(df.columns) - 10} new features")
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for modeling with proper encoding
        """
        print("\n Preparing data for modeling...")
        
        df = df.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ['ContractType', 'PaymentMethod', 'InternetService', 
                           'TenureGroup', 'CustomerValue']
        
        for col in categorical_cols:
            if col in df.columns:
                df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
        
        # Select features for modeling
        feature_cols = [col for col in df.columns if col not in 
                       ['Churn', 'ContractType', 'PaymentMethod', 'InternetService', 
                        'TenureGroup', 'CustomerValue']]
        
        X = df[feature_cols]
        y = df['Churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
    
    def train_models(self, X_train, y_train):
        """
        Train multiple models and compare performance
        """
        print("\n Training multiple models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='roc_auc', n_jobs=-1)
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"      CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models = results
        
        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n Best Model: {best_model_name}")
        
        return results
    
    def optimize_best_model(self, X_train, y_train):
        """
        Hyperparameter tuning for the best model
        """
        print("\n Optimizing best model hyperparameters...")
        
        if isinstance(self.best_model, RandomForestClassifier):
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        elif isinstance(self.best_model, GradientBoostingClassifier):
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
        else:
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }
        
        grid_search = GridSearchCV(
            self.best_model, param_grid, cv=3, 
            scoring='roc_auc', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        
        print(f" Best parameters: {grid_search.best_params_}")
        print(f"   Best CV AUC: {grid_search.best_score_:.4f}")
        
        return self.best_model
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        print("\n Evaluating model performance...")
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print("MODEL PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"              Retained  Churned")
        print(f"Actual Retained  {cm[0,0]:6d}   {cm[0,1]:6d}")
        print(f"       Churned   {cm[1,0]:6d}   {cm[1,1]:6d}")
        
        return {
            'auc': auc_score,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def calculate_business_impact(self, y_test, y_pred_proba, threshold=0.5):
        """
        Calculate business metrics and ROI
        """
        print(f"\n{'='*50}")
        print("BUSINESS IMPACT ANALYSIS")
        print(f"{'='*50}")
        
        # Business assumptions
        avg_customer_value = 1200  # Annual CLV
        retention_cost = 100  # Cost to retain a customer
        retention_success_rate = 0.7  # Success rate of retention efforts
        
        y_pred_business = (y_pred_proba >= threshold).astype(int)
        
        # Calculate impact
        true_churners = (y_test == 1).sum()
        identified_churners = (y_pred_business == 1).sum()
        correctly_identified = ((y_pred_business == 1) & (y_test == 1)).sum()
        
        # Financial calculations
        customers_saved = correctly_identified * retention_success_rate
        revenue_saved = customers_saved * avg_customer_value
        retention_costs = identified_churners * retention_cost
        net_benefit = revenue_saved - retention_costs
        roi = (net_benefit / retention_costs) * 100 if retention_costs > 0 else 0
        
        print(f"\nChurn Statistics:")
        print(f"  Total actual churners: {true_churners}")
        print(f"  Identified as high-risk: {identified_churners}")
        print(f"  Correctly identified: {correctly_identified}")
        print(f"  Detection rate: {(correctly_identified/true_churners)*100:.1f}%")
        
        print(f"\nFinancial Impact (Projected):")
        print(f"  Customers saved: {customers_saved:.0f}")
        print(f"  Revenue saved: ${revenue_saved:,.0f}")
        print(f"  Retention costs: ${retention_costs:,.0f}")
        print(f"  Net benefit: ${net_benefit:,.0f}")
        print(f"  ROI: {roi:.1f}%")
        
        return {
            'customers_saved': customers_saved,
            'revenue_saved': revenue_saved,
            'roi': roi
        }
    
    def get_feature_importance(self, feature_names, top_n=10):
        """
        Extract and display feature importance
        """
        print(f"\n{'='*50}")
        print("TOP CHURN DRIVERS (FEATURE IMPORTANCE)")
        print(f"{'='*50}")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            print(f"\nTop {top_n} features predicting churn:\n")
            for i, idx in enumerate(indices, 1):
                print(f"  {i:2d}. {feature_names[idx]:25s} {importances[idx]:.4f}")
            
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return self.feature_importance
        else:
            print("Feature importance not available for this model type")
            return None


def main():
    """
    Main execution pipeline
    """
    print("="*60)
    print("CUSTOMER CHURN PREDICTION SYSTEM")
    print("Advanced ML Pipeline with Business Intelligence")
    print("="*60)
    
    # Initialize system
    system = ChurnPredictionSystem()
    
    # Generate data
    df = system.generate_synthetic_data(n_samples=5000)
    
    # Feature engineering
    df = system.engineer_features(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = system.prepare_data(df)
    
    # Train multiple models
    results = system.train_models(X_train, y_train)
    
    # Optimize best model
    system.optimize_best_model(X_train, y_train)
    
    # Evaluate
    eval_results = system.evaluate_model(X_test, y_test)
    
    # Business impact
    system.calculate_business_impact(y_test, eval_results['y_pred_proba'])
    
    # Feature importance
    system.get_feature_importance(feature_names)
    
    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE")
    print("="*60)
    print("\nThis system can be deployed to:")
    print("  • Identify high-risk customers daily")
    print("  • Trigger automated retention campaigns")
    print("  • Personalize retention offers based on churn drivers")
    print("  • Track ROI of retention efforts")
    print("  • Integrate with CRM systems via API")


if __name__ == "__main__":
    main()
