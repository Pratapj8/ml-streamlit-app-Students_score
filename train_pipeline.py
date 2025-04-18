import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

# --- Load and Clean Dataset ---
df = pd.read_csv('/Users/apple/Downloads/Data_science_file/Practice_project/stud.csv')
df = df.drop_duplicates().reset_index(drop=True)

# --- Target and Features ---
target = 'math_score'
X = df.drop(columns=[target])
y = df[target]

cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
function_transform_cols = ['reading_score']
standard_scale_cols = ['writing_score']

# --- Pipelines ---
function_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p, validate=False))
])

scaler_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# --- Combine Preprocessing ---
preprocessor = ColumnTransformer(transformers=[
    ('function', function_transformer, function_transform_cols),
    ('scaler', scaler_transformer, standard_scale_cols),
    ('cat', cat_transformer, cat_features)
])

# --- Full Pipeline (model to be added later) ---
base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),
    ('model', RandomForestRegressor(random_state=42))  # Placeholder for GridSearch
])

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Hyperparameter Tuning ---
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(base_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# --- Best Model ---
best_model = grid_search.best_estimator_
print("✅ Best Parameters:", grid_search.best_params_)

# --- Evaluation ---
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Model Evaluation on Test Set:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# --- Save Final Model ---
joblib.dump(best_model, 'best_model_pipeline.pkl')
print("💾 Model saved as 'best_model_pipeline.pkl'")
