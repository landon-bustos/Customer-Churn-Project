import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ChurnEDA:
    def __init__(self):
        self.data = None
        self.numeric_cols = None
        self.categorical_cols = None
    
    def load_data(self):
        self.data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
        self.numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_cols = [col for col in self.data.columns if col not in self.numeric_cols 
                               and col != 'customerID']
        return self.data
    
    def basic_analysis(self):
        churn_dist = self.data['Churn'].value_counts(normalize=True)
        plt.figure(figsize=(8, 6))
        plt.pie(churn_dist, labels=['No', 'Yes'], autopct='%1.1f%%')
        plt.title('Customer Churn Distribution')
        plt.savefig('visualizations/churn_distribution.png')
        plt.close()
    
    def numeric_analysis(self):
        
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(self.numeric_cols, 1):
            plt.subplot(1, 3, i)
            sns.histplot(data=self.data, x=col, hue='Churn', multiple="stack")
            plt.title(f'{col} Distribution by Churn')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/numeric_distributions.png')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data[self.numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Numeric Variables')
        plt.savefig('visualizations/correlation_matrix.png')
        plt.close()
    
    def categorical_analysis(self):
        n_cols = 3
        n_rows = (len(self.categorical_cols) - 1) // n_cols + 1
        plt.figure(figsize=(20, 5*n_rows))
        
        for i, col in enumerate(self.categorical_cols, 1):
            if col != 'Churn':
                plt.subplot(n_rows, n_cols, i)
                churn_pcts = (self.data.groupby(col)['Churn']
                             .value_counts(normalize=True)
                             .unstack())
                churn_pcts.plot(kind='bar', stacked=True)
                plt.title(f'Churn Rate by {col}')
                plt.xlabel(col)
                plt.ylabel('Percentage')
                plt.xticks(rotation=45)
                
                for c in churn_pcts.columns:
                    for j, v in enumerate(churn_pcts[c]):
                        plt.text(j, v/2, f'{v:.1%}', ha='center')
        
        plt.tight_layout()
        plt.savefig('visualizations/categorical_analysis.png')
        plt.close()
    
    def service_analysis(self):
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        service_adoption = pd.DataFrame()
        for col in service_cols:
            service_adoption[col] = self.data[col].value_counts(normalize=True)
        
        plt.figure(figsize=(15, 6))
        service_adoption.iloc[0].plot(kind='bar')
        plt.title('Service Adoption Rates')
        plt.xlabel('Services')
        plt.ylabel('Adoption Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/service_adoption.png')
        plt.close()
        
        self.data['num_services'] = self.data[service_cols].apply(
            lambda x: sum(x != 'No') - sum(x == 'No phone service' ) - sum(x == 'No internet service'),
            axis=1
        )
        
        plt.figure(figsize=(10, 6))
        churn_by_services = self.data.groupby('num_services')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        sns.barplot(data=churn_by_services, x='num_services', y='Churn')
        plt.title('Churn Rate by Number of Services')
        plt.xlabel('Number of Services')
        plt.ylabel('Churn Rate (%)')
        for i, v in enumerate(churn_by_services['Churn']):
            plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        plt.savefig('visualizations/churn_by_services.png')
        plt.close()
    
    def contract_payment_analysis(self):
        contract_cols = ['Contract', 'PaperlessBilling', 'PaymentMethod']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for i, col in enumerate(contract_cols):
            churn_rates = (self.data.groupby(col)['Churn']
                          .apply(lambda x: (x == 'Yes').mean() * 100))
            
            churn_rates.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Churn Rate by {col}')
            axes[i].set_ylabel('Churn Rate (%)')
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=45)
            
            for j, v in enumerate(churn_rates):
                axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('visualizations/contract_payment_analysis.png')
        plt.close()
    
    def tenure_analysis(self):
        self.data['tenure_group'] = pd.qcut(self.data['tenure'], 
                                          q=4, 
                                          labels=['0-25%', '25-50%', '50-75%', '75-100%'])
        
        plt.figure(figsize=(10, 6))
        churn_by_tenure = self.data.groupby('tenure_group')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        sns.barplot(data=churn_by_tenure, x='tenure_group', y='Churn')
        plt.title('Churn Rate by Tenure Group')
        plt.xlabel('Tenure Percentile')
        plt.ylabel('Churn Rate (%)')
        for i, v in enumerate(churn_by_tenure['Churn']):
            plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        plt.savefig('visualizations/churn_by_tenure.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='tenure', y='MonthlyCharges', 
                       hue='Churn', alpha=0.5)
        plt.title('Monthly Charges vs Tenure')
        plt.savefig('visualizations/charges_tenure_scatter.png')
        plt.close()

def main():
    import os
    os.makedirs('visualizations', exist_ok=True)
    eda = ChurnEDA()
    eda.load_data()
    eda.basic_analysis()
    eda.numeric_analysis()
    eda.categorical_analysis()
    eda.service_analysis()
    eda.contract_payment_analysis()
    eda.tenure_analysis()

if __name__ == "__main__":
    main()
