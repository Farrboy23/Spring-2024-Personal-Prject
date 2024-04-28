import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform

# Load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = ['ExistingAccountStatus', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount',
                'Savings', 'EmploymentSince', 'InstallmentRate', 'PersonalStatusSex', 'OtherDebtors',
                'PresentResidence', 'Property', 'Age', 'OtherInstallmentPlans', 'Housing',
                'ExistingCredits', 'Job', 'PeopleLiable', 'Telephone', 'ForeignWorker', 'CreditRisk']
data = pd.read_csv(url, names=column_names, delimiter=' ')

# Preprocessing data
# Encode categorical data
categorical_columns = [col for col in data.columns if data[col].dtype == object]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Identify numeric columns (excluding the target variable 'CreditRisk')
numeric_columns = [col for col in data.columns if col not in categorical_columns and col != 'CreditRisk']

# Scale numeric data
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Split the dataset into training and test sets
X = data.drop('CreditRisk', axis=1)
y = data['CreditRisk'].apply(lambda x: 1 if x == 2 else 0)  # Risk is '1' if CreditRisk is '2' else '0'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize SMOTE
smote = SMOTE(random_state=42)

# Initialize the XGBClassifier
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Create a pipeline with SMOTE and XGBClassifier
pipeline = ImbPipeline(steps=[('smote', smote), ('xgb_classifier', xgb_classifier)])

# Define hyperparameter space for RandomizedSearchCV
param_dist = {
    'xgb_classifier__n_estimators': randint(100, 500),
    'xgb_classifier__learning_rate': uniform(0.01, 0.3),
    'xgb_classifier__max_depth': randint(3, 10),
    'xgb_classifier__min_child_weight': randint(1, 6),
    'xgb_classifier__subsample': uniform(0.5, 0.5),
    'xgb_classifier__colsample_bytree': uniform(0.5, 0.5)
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=100, cv=3, 
                                   scoring='roc_auc', n_jobs=-1, verbose=3, random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_model = random_search.best_estimator_

# Predictions and Evaluations using the best model
y_pred = best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Good Credit', 'Bad Credit']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Feature Importance
best_xgb_classifier = best_model.named_steps['xgb_classifier']
feature_importances = pd.DataFrame(best_xgb_classifier.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
sns.barplot(x='importance', y=feature_importances.index, data=feature_importances)
plt.title('Feature Importances')
plt.show()

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
y_pred_prob = best_xgb_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
average_precision = average_precision_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Average precision = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Probability Distribution Plot
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_prob, bins=20, kde=True, color='skyblue')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability of Risk')
plt.ylabel('Frequency')
plt.show()

# Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='.')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.title('Calibration Curve')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.show()
