import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import os

class ChurnAnalyzer:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scalers = {}
        
    def load_data(self):
        data_path = os.path.join(os.getcwd(), 'data')
        self.data = pd.read_csv(os.path.join(data_path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
        return self.data

    def prepare_features(self):
        df = self.data.copy()
        
        numeric_features = ['MonthlyCharges', 'TotalCharges', 'tenure']
        categorical_features = [
            'Contract', 'PaymentMethod', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'MultipleLines',
            'SeniorCitizen', 'Dependents', 'PhoneService'
        ]
        
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        for col in numeric_features:
            df[col] = df[col].fillna(df[col].median())
        
        df['Revenue_per_Month'] = (df['TotalCharges'] / (df['tenure'] + 1)).fillna(0)
        df['MonthlyCharges_to_Total_Ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).fillna(0)
        
        df['tenure_squared'] = df['tenure'] ** 2
        df['MonthlyCharges_squared'] = df['MonthlyCharges'] ** 2
        
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        df[service_cols] = df[service_cols].fillna('No')
        
        df['total_services'] = df[service_cols].apply(
            lambda x: sum([1 for i in x if i not in ['No', 'No phone service', 'No internet service']])
        )
        
        df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
        df['is_loyal_customer'] = (df['tenure'] >= 24).astype(int)
        
        df['high_monthly_charges'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
        df['high_total_charges'] = (df['TotalCharges'] > df['TotalCharges'].median()).astype(int)
        
        df['has_protection_services'] = ((df['OnlineSecurity'] == 'Yes') | 
                                       (df['OnlineBackup'] == 'Yes') | 
                                       (df['DeviceProtection'] == 'Yes')).astype(int)
        df['has_support'] = (df['TechSupport'] == 'Yes').astype(int)
        
        engineered_numeric = [
            'Revenue_per_Month', 'MonthlyCharges_to_Total_Ratio',
            'tenure_squared', 'MonthlyCharges_squared', 'total_services'
        ]
        
        df_numeric = df[numeric_features + engineered_numeric]
        df_categorical = pd.get_dummies(df[categorical_features].fillna('Missing'))
        df_binary = df[['is_new_customer', 'is_loyal_customer',
                       'high_monthly_charges', 'high_total_charges',
                       'has_protection_services', 'has_support']]
        
        df_model = pd.concat([df_numeric, df_categorical, df_binary], axis=1)
        df_model = df_model.fillna(0)  
        
        y = (df['Churn'] == 'Yes').astype(int)
        
        return df_model, y

    def logistic_regression_analysis(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, sampling_strategy=0.8)), 
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],
            'classifier__max_iter': [1000],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1', 
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': np.abs(best_model.named_steps['classifier'].coef_[0])
        }).sort_values('coefficient', ascending=False)
        
        self.models['logistic'] = best_model
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.axvline(x=fpr[optimal_idx], color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=tpr[optimal_idx], color='r', linestyle='--', alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Logistic Regression (with Optimal Threshold)')
        plt.legend(loc="lower right")
        plt.savefig('visualizations/logistic_regression_roc.png')
        plt.close()
        
        return {
            'model': best_model,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'roc_auc': roc_auc
        }

    def xgboost_analysis(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', xgb.XGBClassifier(use_label_encoder=False, random_state=42))
        ])
        
        param_grid = {
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__n_estimators': [100, 200],
            'classifier__min_child_weight': [1, 3],
            'classifier__subsample': [0.8, 1.0],
            'classifier__colsample_bytree': [0.8, 1.0]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features (XGBoost)')
        plt.tight_layout()
        plt.savefig('visualizations/xgboost_feature_importance.png')
        plt.close()
        
        self.models['xgboost'] = best_model
        
        return {
            'model': best_model,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'roc_auc': roc_auc
        }

    def cohort_analysis(self):
        df = self.data.copy()
        
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        df['tenure_cohort'] = pd.qcut(df['tenure'], 
                                     q=5, 
                                     labels=['New (0-20%)', 'Early (20-40%)', 
                                            'Developing (40-60%)', 'Mature (60-80%)', 
                                            'Loyal (80-100%)'])
        
        df['value_cohort'] = pd.qcut(df['MonthlyCharges'], 
                                    q=5, 
                                    labels=['Low', 'Medium-Low', 
                                           'Medium', 'Medium-High', 
                                           'High'])
        
        df['IsRetained'] = (df['Churn'] == 'No').astype(int)
        df['CLV'] = df['tenure'] * df['MonthlyCharges']
        df['ARPU'] = df['TotalCharges'] / df['tenure']
        
        tenure_metrics = df.groupby('tenure_cohort').agg({
            'Churn': lambda x: (x == 'Yes').mean(),
            'IsRetained': 'mean',
            'MonthlyCharges': 'mean',
            'TotalCharges': lambda x: x.mean(),
            'CLV': 'mean',
            'ARPU': 'mean',
            'customerID': 'count',
            'tenure': 'mean'
        }).round(2)
        
        tenure_metrics['total_monthly_revenue'] = tenure_metrics['MonthlyCharges'] * tenure_metrics['customerID']
        tenure_metrics['potential_monthly_revenue_loss'] = tenure_metrics['total_monthly_revenue'] * tenure_metrics['Churn']
        
        tenure_metrics.columns = ['Churn_Rate', 'Retention_Rate', 'Monthly_ARPU', 
                                 'Total_Revenue', 'Customer_Lifetime_Value', 
                                 'Average_Revenue_Per_User', 'Cohort_Size', 'Average_Tenure',
                                 'Monthly_Cohort_Revenue', 'Monthly_Revenue_at_Risk']
        
        value_metrics = df.groupby('value_cohort').agg({
            'Churn': lambda x: (x == 'Yes').mean(),
            'IsRetained': 'mean',
            'tenure': 'mean',
            'TotalCharges': lambda x: x.mean(),
            'CLV': 'mean',
            'ARPU': 'mean',
            'customerID': 'count',
            'MonthlyCharges': 'mean'
        }).round(2)
        
        value_metrics['total_monthly_revenue'] = value_metrics['MonthlyCharges'] * value_metrics['customerID']
        value_metrics['potential_monthly_revenue_loss'] = value_metrics['total_monthly_revenue'] * value_metrics['Churn']
        
        value_metrics.columns = ['Churn_Rate', 'Retention_Rate', 'Average_Tenure', 
                                'Total_Revenue', 'Customer_Lifetime_Value', 
                                'Average_Revenue_Per_User', 'Cohort_Size', 'Average_Monthly_Charges',
                                'Monthly_Cohort_Revenue', 'Monthly_Revenue_at_Risk']
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=tenure_metrics.reset_index(), 
                   x='tenure_cohort', 
                   y='Retention_Rate')
        plt.title('Retention Rate by Tenure Cohort')
        plt.ylabel('Retention Rate')
        plt.xlabel('Tenure Cohort')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/tenure_cohort_retention.png')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=value_metrics.reset_index(), 
                   x='value_cohort', 
                   y='Customer_Lifetime_Value')
        plt.title('Customer Lifetime Value by Monthly Charges Segment')
        plt.ylabel('Customer Lifetime Value ($)')
        plt.xlabel('Monthly Charges Segment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/value_cohort_clv.png')
        plt.close()
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='tenure_cohort', y='ARPU', 
                    hue='value_cohort', marker='o')
        plt.title('ARPU Trends by Cohort')
        plt.ylabel('Average Revenue Per User ($)')
        plt.xlabel('Tenure Cohort')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/cohort_arpu_trends.png')
        plt.close()
        
        return {
            'tenure_metrics': tenure_metrics,
            'value_metrics': value_metrics
        }

def main():
    os.makedirs('visualizations', exist_ok=True)
    analyzer = ChurnAnalyzer()
    data = analyzer.load_data()
    X, y = analyzer.prepare_features()
    
    print("Training models with improvements...")
    print("(This may take a few minutes due to hyperparameter tuning)")
    
    lr_results = analyzer.logistic_regression_analysis(X, y)
    xgb_results = analyzer.xgboost_analysis(X, y)
    
    from model_analysis import ModelAnalysis
    model_analyzer = ModelAnalysis()
    analysis_results = model_analyzer.analyze_model_performance(lr_results, xgb_results)
    
    print("\nLogistic Regression Results:")
    print("Best Parameters:", lr_results['best_params'])
    print("Cross-validation ROC-AUC Score: {:.3f}".format(lr_results['cv_score']))
    print("Test Set ROC-AUC Score: {:.3f}".format(lr_results['roc_auc']))
    print("\nClassification Report:")
    print(lr_results['classification_report'])
    
    print("\nXGBoost Results:")
    print("Best Parameters:", xgb_results['best_params'])
    print("Cross-validation ROC-AUC Score: {:.3f}".format(xgb_results['cv_score']))
    print("Test Set ROC-AUC Score: {:.3f}".format(xgb_results['roc_auc']))
    print("\nClassification Report:")
    print(xgb_results['classification_report'])
    
    print("\nTop 5 Important Features (Logistic Regression):")
    print(lr_results['feature_importance'].head())
    print("\nTop 5 Important Features (XGBoost):")
    print(xgb_results['feature_importance'].head())
    
    print("\nModel analysis visualizations saved:")
    for name, path in analysis_results.items():
        print(f"- {name}: {path}")
    
    cohort_results = analyzer.cohort_analysis()

if __name__ == "__main__":
    main()
