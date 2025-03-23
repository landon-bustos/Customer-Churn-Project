# Customer Churn Analysis Project

## Overview
Predictive analysis of customer churn in telecommunications using machine learning, achieving 81% accuracy in identifying at-risk customers.

## Table of Contents
1. [Key Findings](#key-findings)
2. [Model Performance](#model-performance)
3. [Business Impact](#business-impact)
4. [Technical Details](#technical-details)
5. [Setup and Usage](#setup-and-usage)

## Key Findings

### Customer Behavior
- Month-to-month contracts have 3x higher churn rate
- Long-term customers (24+ months) show 70% lower churn risk
- Technical support users have 45% lower churn probability

### Service Impact
- Fiber optic service users show 15% higher churn rate
- Multiple service subscribers are 2x more likely to stay
- Protection services reduce churn risk by 40%

### Financial Patterns
- High monthly charges (>$80) correlate with 2x churn risk
- Early tenure (<6 months) + high charges = highest risk segment
- Service upgrades in first 3 months reduce churn by 30%

## Model Performance

### XGBoost (Primary Model)
- Accuracy: 81%
- ROC-AUC: 0.856
- Precision: 88% (No Churn), 62% (Churn)
- Recall: 85% (No Churn), 67% (Churn)

### Key Predictors
1. Contract Type (Month-to-month)
2. Technical Support Status
3. Online Security
4. Protection Services
5. Internet Service Type

## Business Impact

### 1. Early Engagement Strategy
- Implement structured onboarding program
- Focus on first 90-day customer experience
- Provide early-stage technical support

### 2. Service Quality Improvements
- Enhance fiber optic service reliability
- Expand technical support accessibility
- Develop service quality guarantees

### 3. Contract Optimization
- Design compelling long-term contracts
- Create smooth transition paths
- Implement loyalty rewards program

### 4. Risk Management
- Monitor early warning indicators
- Deploy proactive support measures
- Regular satisfaction assessments

### 5. Value Enhancement
- Segment-based pricing strategy
- Optimize service bundles
- Design targeted retention packages

## Technical Details

### Implementation
- Python with scikit-learn and XGBoost
- SMOTE for handling class imbalance
- Grid search optimization
- Feature engineering and selection

### Repository Structure
```
├── focused_churn_analysis.py   # Main model
├── eda_analysis.py            # Data exploration
├── model_analysis.py         # Performance evaluation
├── requirements.txt          # Dependencies
└── visualizations/           # Output charts
```

## Setup and Usage

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python focused_churn_analysis.py
```

### Output
- Model performance metrics
- Feature importance analysis
- Visualization charts
- Segment-wise predictions
