import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score,classification_report,ConfusionMatrixDisplay

from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from app.featureengineering import FeatureEngineering

import joblib as jbl

#Import the dataset
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
sample_submission = pd.read_csv('Data/sample_submission.csv')

# Define feature matrix X and target vector y
X = train.drop(['is_promoted','employee_id'], axis=1)
y = train['is_promoted']

def evaluate_model(y_true, y_pred, y_proba):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, y_proba))
    
# Segregate the categorical columns & numerical columns
categorical_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel']
numerical_cols = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',
       'KPIs_met >80%', 'awards_won?', 'avg_training_score', 'new_joinee']

# Split the data into training and validation sets
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=42)

categorical_pipeline = Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]
)

# Preprocessing pipeline for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat_processing', categorical_pipeline, categorical_cols),
        ('scaler', StandardScaler(), numerical_cols)
    ]
)

# Model Builder
model_pipeline = Pipeline(
    steps= [
        ('feature_engineering', FeatureEngineering()),
        ('preprocessor', preprocessor),
        ('undersampler', RandomOverSampler()),
        ('model', XGBClassifier(subsample=0.9,reg_lambda=1,reg_alpha=0.8,n_estimators=550,min_child_weight=8,max_depth=7,learning_rate=0.2,gamma=0.2,colsample_bytree=0.7,scale_pos_weight=1))
    ]
)

model_pipeline.fit(train_x, train_y)
best_threshold = 0.77

y_train_proba = model_pipeline.predict_proba(train_x)[:, 1]
y_train_pred = (y_train_proba >= best_threshold).astype(int)

print("Train Set Evaluation:")
evaluate_model(train_y, y_train_pred, y_train_proba)

y_test_proba = model_pipeline.predict_proba(test_x)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

print("Test Set Evaluation:")
evaluate_model(test_y, y_test_pred, y_test_proba)

# Save the model
jbl.dump({'model':model_pipeline, 'threshold':best_threshold}, 'Models/hr_analytics_thrsh77_model.pkl')