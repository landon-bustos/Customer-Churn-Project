import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class ModelAnalysis:
    def __init__(self):
        pass
    
    def analyze_model_performance(self, logistic_results, xgboost_results):
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        top_lr_features = logistic_results['feature_importance'].head(10)
        sns.barplot(data=top_lr_features, x='coefficient', y='feature')
        plt.title('Top 10 Features (Logistic Regression)')
        
        plt.subplot(1, 2, 2)
        top_xgb_features = xgboost_results['feature_importance'].head(10)
        sns.barplot(data=top_xgb_features, x='importance', y='feature')
        plt.title('Top 10 Features (XGBoost)')
        
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance_comparison.png')
        plt.close()
        
        metrics = pd.DataFrame({
            'Metric': ['ROC-AUC', 'Cross-val Score'],
            'Logistic Regression': [logistic_results['roc_auc'], logistic_results['cv_score']],
            'XGBoost': [xgboost_results['roc_auc'], xgboost_results['cv_score']]
        })
        
        plt.figure(figsize=(10, 6))
        metrics.set_index('Metric').plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=0)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig('visualizations/model_performance_comparison.png')
        plt.close()
        
        print("\nLogistic Regression Classification Report:")
        print(logistic_results['classification_report'])
        print("\nXGBoost Classification Report:")
        print(xgboost_results['classification_report'])
        
        return {
            'feature_importance_comparison': 'visualizations/feature_importance_comparison.png',
            'model_performance_comparison': 'visualizations/model_performance_comparison.png'
        }

if __name__ == "__main__":
    from focused_churn_analysis import ChurnAnalyzer
    
    churn_analyzer = ChurnAnalyzer()
    model_analyzer = ModelAnalysis()
    
    churn_analyzer.load_data()
    X, y = churn_analyzer.prepare_features()
    
    print("Training models...\n")
    
    lr_results = churn_analyzer.logistic_regression_analysis(X, y)
    xgb_results = churn_analyzer.xgboost_analysis(X, y)
    
    analysis_results = model_analyzer.analyze_model_performance(lr_results, xgb_results)
    
    print("Analysis completed. Visualizations saved:")
    for name, path in analysis_results.items():
        print(f"- {name}: {path}")
