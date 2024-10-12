import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

DEBUG = False

# debug print
def dprint(text: str):
    if DEBUG:
        print(text)

# returns all possible combinations of the list passed as argument
def list_features(features) -> list:
    dprint("Starting making features combinations")
    n_combinations = 2 ** len(features)
    all_combinations = []
    i = 1
    while i < n_combinations:
        bin_n = bin(i)[2:]
        all_combinations.append([])
        for idx, ch in enumerate(bin_n):
            if(ch == '1'):
                all_combinations[i-1].append(features[idx])
        i += 1
    dprint("finished making combinations")
    return all_combinations


def encodeDF(preprocessor, input):
    return preprocessor.transform(input)

def learn(features):
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    test_y = pd.read_csv('gender_submission.csv')

    train_y = train_data.Survived

    train_X = train_data[features]
    test_X = test_data[features]

    numeric_features = train_X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = train_X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])

    preprocessor.fit(train_X)

    train_X_encoded = encodeDF(preprocessor, train_X)
    test_X_encoded = encodeDF(preprocessor, test_X)

    model = RandomForestRegressor()
    model.fit(train_X_encoded, train_y)

    test_predictions = model.predict(test_X_encoded)
    test_data['Survived'] = test_predictions
    final = test_data[['PassengerId', 'Survived']]
    final.to_csv("prediction.csv", index=False)
    
    return mean_absolute_error(test_y['Survived'], test_predictions)

def learn_avg10(features):
    dprint("starting learning")
    min_mae = -1
    min_mae_features = []
    for idx, f_list in enumerate(features):
        dprint(f"run {idx + 1} out of {len(features)}")
        mae = 0
        for i in range(10):
            mae += learn(f_list)
        avg_mae = mae / 10
        if avg_mae < min_mae:
            min_mae = avg_mae
            min_mae_features = f_list

    return min_mae, min_mae_features

if __name__ == "__main__":
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--debug":
            DEBUG = True
    all_features = ['Pclass','Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    fs = list_features(all_features)
    for i, f in enumerate(fs):
        if len(f) == 0:
            print(i, f)

    mae, fs = learn_avg10(list_features(all_features))
    print("mae: {mae}, features: {fs}")
