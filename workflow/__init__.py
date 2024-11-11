from workflow import etl, feature_selection, train, validate


def train_model():
    X_train, X_test, y_train, y_test = etl.process_data()
    X_train = feature_selection.select_features(X_train)
    train.train_model(X_train, y_train)
    metrics = validate.validate_model(X_test, y_test)
    return metrics


def validate_model():
    metrics = validate.validate_model()
    return metrics
