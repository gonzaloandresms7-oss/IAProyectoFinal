
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, precision_score, recall_score

def preprocessing(X_train, X_test):

    numerical = X_train.select_dtypes(
        include=['number']
    ).columns

    categorical = X_train.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns

    X_train_num = X_train[numerical].copy()
    X_test_num = X_test[numerical].copy()

    X_train_cat = X_train[categorical].copy()
    X_test_cat = X_test[categorical].copy()

    if len(numerical) > 0:

        scaler = StandardScaler()

        X_train_num = scaler.fit_transform(X_train_num)
        X_test_num = scaler.transform(X_test_num)

    else:
        X_train_num = X_train_num.to_numpy()
        X_test_num = X_test_num.to_numpy()

    if len(categorical) > 0:

        encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )

        X_train_cat = encoder.fit_transform(X_train_cat)
        X_test_cat = encoder.transform(X_test_cat)

    else:
        X_train_cat = X_train_cat.to_numpy()
        X_test_cat = X_test_cat.to_numpy()

    X_train = pd.DataFrame(
        X_train_num
    )

    X_test = pd.DataFrame(
        X_test_num
    )

    X_train = pd.concat([
        X_train,
        pd.DataFrame(X_train_cat)
    ], axis=1)

    X_test = pd.concat([
        X_test,
        pd.DataFrame(X_test_cat)
    ], axis=1)

    return X_train.to_numpy(), X_test.to_numpy()


def select_model(model_type, n_neighbors=5):

    if model_type == 'KNN':

        model = KNeighborsClassifier(
            n_neighbors=n_neighbors
        )

    elif model_type == 'Logistic Regression':

        model = LogisticRegression(
            max_iter=1000
        )

    elif model_type == 'Random Forest':

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    else:
        raise ValueError('Invalid model selection')

    return model


def train_model(df, features, target, model_type,
                feature_mode, n_components=2, n_neighbors=5):

    data = df[features + [target]].dropna()

    X = data[features]
    y = data[target]

    if y.nunique() < 2:
        raise ValueError(
            'The target must have at least 2 classes'
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            'Each target class must have at least 2 records'
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_test = preprocessing(
        X_train,
        X_test
    )

    explained_variance = None

    if feature_mode == 'PCA Reduction':

        if n_components < 2:
            raise ValueError(
                'PCA requires at least 2 components'
            )

        if n_components > min(X_train.shape):
            raise ValueError(
                'Number of components exceeds available dimensions'
            )

        pca = PCA(
            n_components=n_components
        )

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        explained_variance = sum(
            pca.explained_variance_ratio_
        )

    model = select_model(
        model_type,
        min(n_neighbors, len(X_train))
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average='weighted',
        zero_division=0
    )

    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Predictions': predictions,
        'Actual': y_test,
        'Explained Variance': explained_variance
    }

    return results
