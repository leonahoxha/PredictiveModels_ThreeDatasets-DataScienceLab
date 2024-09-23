# Predictive Models on Three Datasets - Data Science Lab

## Project Overview
This project analyzes three distinct datasets sourced from **Kaggle**, focusing on evaluating the performance of **supervised learning algorithms** in predicting binary target variables. The datasets span across **health**, **environment**, and **business** sectors:
- **Cardiovascular Dataset** (Health)
- **Weather in Australia** (Environment)
- **Hotel Reservation Dataset** (Business)

The project explores the impact of sophisticated modeling techniques, model transferability across datasets, and the effects of standardization. Additionally, we tackle challenges such as **imbalanced datasets**, feature selection, hyperparameter tuning, and mitigating overfitting.

## Key Objectives
- Evaluate the performance of different **supervised learning algorithms** on multiple datasets.
- Investigate the **transferability** of models across datasets.
- Analyze the effect of **standardization** and **hyperparameter optimization** on model performance.
- Mitigate issues of **imbalanced datasets** and assess the trade-offs between feature selection and model performance.

## Datasets
- **Cardiovascular Dataset**: Focuses on predicting cardiovascular events using health-related features.
- **Weather in Australia Dataset**: Predicts rainfall in Australia based on weather conditions.
- **Hotel Reservation Dataset**: Aims to predict whether a reservation will be canceled or confirmed.

All datasets contain a binary target variable and a variety of features for model training and evaluation.

- **Source for all datasets**: [Kaggle Datasets](https://www.kaggle.com/datasets)

## Methods and Techniques
### Supervised Learning Algorithms:
- **Random Forest Classifier**
- **Logistic Regression**
- **Decision Tree**
- **Support Vector Machine (SVM)**

### Key Analysis Techniques:
- **Preprocessing**: Handling missing data, standardization, and imbalanced data correction.
- **Feature Selection**: Employed techniques like **SelectKBest** for choosing influential features.
- **Cross-Validation**: Ensured model generalizability and avoided overfitting.
- **Learning Curve Analysis**: Examined model performance as the size of training data increased.
- **Hyperparameter Optimization**: Manually adjusted key hyperparameters for each model to achieve optimal performance.

## Tools & Technologies
- **Python** for scripting and data analysis.
- **Pandas**, **NumPy** for data manipulation.
- **Scikit-learn** for machine learning models and preprocessing techniques.
- **Matplotlib**, **Seaborn** for visualizations.

## Key Findings
1. **Sophisticated Modeling**: Advanced models, like **Random Forest** and **SVM**, performed better than simpler models such as **Logistic Regression**, especially when handling complex patterns in the data.
   
2. **Model Transferability**: Models that performed well in one dataset did not consistently yield the same performance across different datasets. Dataset-specific characteristics significantly influenced model success.

3. **Standardization**: Standardization had varying effects—improving **Random Forest** in the **Weather in Australia** dataset but not showing significant improvement in others.

4. **Imbalanced Data**: The **Cardiovascular Dataset** exhibited imbalanced classes, which we addressed using undersampling techniques. Balancing the data improved model precision and recall.

5. **Feature Selection**: Feature selection resulted in slightly lower performance but was computationally more efficient.

6. **Hyperparameter Optimization**: Manual tuning of hyperparameters demonstrated significant improvements in **model generalization** and reduced overfitting risks.

7. **Overfitting**: Risk of overfitting was present in complex models like **Random Forest** when model depth and number of estimators were not carefully monitored. Cross-validation and learning curve analysis helped mitigate this issue.

## Conclusion
This project demonstrated the **importance of selecting the right model** for each dataset and the trade-offs involved in feature selection, standardization, and hyperparameter tuning. While **Random Forest** and **SVM** provided the highest performance, they required careful tuning to avoid overfitting.

The findings also highlighted that **model performance is highly dataset-dependent**, emphasizing the need for dataset-specific preprocessing and model selection strategies.

## Future Enhancements
- Explore **ensemble techniques** such as stacking models for better performance.
- Experiment with **deep learning models** for more complex datasets.
- Investigate further **feature engineering** techniques to improve model accuracy.
