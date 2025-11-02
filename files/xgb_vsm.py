import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.svm import SVC


def load_data():
    # load data
    x = np.random.uniform(low=0.0, high=1.0, size=(5000, 2))
    r = np.linalg.norm(x, axis=-1)
    y = (r < 1).astype(np.float32)
    return x, y


def main():
    # Load data
    X, y = load_data()

    # Randomly split the data in to training set and testing test;
    # Let testing set contain 20% of total dataset
    # You can check the train_test_split function in sklearn package
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # TO DO：
    # Using the XgboostClassifier, report the training and testing accuracy
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    train_pred = xgb_model.predict(X_train)
    train_accuracy = (train_pred == y_train).mean()
    print(train_accuracy)
    test_pred = xgb_model.predict(X_test)
    test_accuracy = (test_pred == y_test).mean()
    print(test_accuracy)

    # TO DO：
    # Using SVM, report the training and testing accuracy
    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    train_pred = svc_model.predict(X_train)
    train_accuracy = (train_pred == y_train).mean()
    print(train_accuracy)
    test_pred = svc_model.predict(X_test)
    test_accuracy = (test_pred == y_test).mean()
    print(test_accuracy)


if __name__ == "__main__":

    # API usage
    main()
