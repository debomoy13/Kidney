# %%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier

from xgboost import XGBClassifier

import joblib


# %%
df = pd.read_csv(
    "C:\\Users\\Debomoy Patra\\Downloads\\archive\\kidney_disease_dataset.csv"
)

df.head()
df.columns
df.dtypes


# %%
categorical_columns = [
    'Red blood cells in urine',
    'Pus cells in urine',
    'Pus cell clumps in urine',
    'Bacteria in urine',
    'Hypertension (yes/no)',
    'Diabetes mellitus (yes/no)',
    'Coronary artery disease (yes/no)',
    'Appetite (good/poor)',
    'Pedal edema (yes/no)',
    'Anemia (yes/no)',
    'Family history of chronic kidney disease',
    'Smoking status',
    'Physical activity level',
    'Urinary sediment microscopy results'
]

binary_columns = [
    'Hypertension (yes/no)',
    'Diabetes mellitus (yes/no)',
    'Coronary artery disease (yes/no)',
    'Pedal edema (yes/no)',
    'Anemia (yes/no)',
    'Family history of chronic kidney disease',
    'Smoking status'
]

for i in binary_columns:
    df[i] = df[i].map({
        'yes': 1,
        'no': 0
    })


df['Appetite'] = df['Appetite (good/poor)'].map({
    'good': 1,
    'poor': 0
})


df['Physical activity level'] = df['Physical activity level'].map({
    'low': 0,
    'moderate': 1,
    'high': 2
})


label_encoders = {}

for col in [
    'Red blood cells in urine',
    'Pus cells in urine',
    'Pus cell clumps in urine',
    'Bacteria in urine',
    'Urinary sediment microscopy results'
]:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

    label_encoders[col] = encoder


# %%
from sklearn.model_selection import train_test_split

X = df.drop('Target', axis=1)


target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    df['Target'].astype(str)
)


columns_to_drop = [
    'Appetite (good/poor)',
    'Physical activity level'
] + categorical_columns


X = X.drop(
    [col for col in columns_to_drop if col in X.columns],
    axis=1
)


# splitting the data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)


# %%
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# %%

model = {

    "Xgboost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )

}


# %%
accuracy_res = []

for i, m in model.items():

    if i in [
        'Logistic Regression',
        'SVM',
        'neural_network'
    ]:

        m.fit(
            X_train_scaled,
            y_train
        )

        y_pred = m.predict(
            X_test_scaled
        )

    else:

        m.fit(
            X_train,
            y_train
        )

        y_pred = m.predict(
            X_test
        )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    accuracy_res.append(
        (i, accuracy)
    )

    print("Classification Report")
    print(i)

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# %%
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


for name, m in model.items():

    if name in [
        "Logistic Regression",
        "SVM",
        "Neural Network"
    ]:

        X_data = X_train_scaled

    else:

        X_data = X_train


    scores = cross_val_score(
        m,
        X_data,
        y_train,
        cv=cv,
        scoring="accuracy"
    )


    print(name)

    print(
        f"CV Accuracy: {scores.mean() * 100:.2f}%"
    )

    print(
        f"Std: ±{scores.std() * 100:.2f}%"
    )


# %%
from sklearn.metrics import confusion_matrix

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# %%
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)


# %%
# Saving the model and preprocessing objects

joblib.dump(
    model["Xgboost"],
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    target_encoder,
    "target_encoder.pkl"
)

joblib.dump(
    label_encoders,
    "label_encoders.pkl"
)

joblib.dump(
    list(X.columns),
    "feature_columns.pkl"
)

print("Model saved successfully.")