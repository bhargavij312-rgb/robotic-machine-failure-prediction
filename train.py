import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from features import add_features, LEAKAGE_COLS, TARGET

df = pd.read_csv('data/ai4i2020.csv')
df = add_features(df)
X = df.drop(columns=LEAKAGE_COLS + [TARGET, 'UDI', 'Product ID'])
y = df[TARGET]
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
preprocess = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
model = Pipeline([('preprocess', preprocess), ('clf', DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'models/failure_model.joblib')
print(classification_report(y_test, model.predict(X_test)))
