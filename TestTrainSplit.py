import sklearn as sk


def train_test_split(data, labels, ratio):

    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(data, labels, test_size=ratio, random_state=42)
    split_dict = {"Train": {'Data': X_train, 'Labels': y_train}, "Test": {'Data': X_test, 'Labels': y_test}}
    return split_dict
