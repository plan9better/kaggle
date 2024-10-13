import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DEBUG = False

def dprint(text: str):
    if DEBUG:
        print(text)

def extract_title(name: str) -> int:
    title_map = {
        "Mr": 1, "Mrs": 1, "Miss": 1, "Ms": 1, "Master": 1, "Mlle": 1, "Mme": 1,
        "Dr": 2, "Rev": 2,
        "Major": 3, "Capt": 3, "Col": 3,
        "Lady": 4, "Sir": 4, "Jonkheer": 4, "Don": 4, "Dona": 4, "Countess": 4
    }
    left = name.split('.')[0]
    title = left.split(' ')[-1]
    return title_map.get(title, 0)  # Return 0 for unknown titles

def transform_embarks(input: str) -> int:
    em_map = {"S": 0, "C": 1, "Q": 2}
    return em_map.get(input, -1)  # Return -1 for unknown ports

def transform_sex(input: str) -> int:
    sex_map = {"male": 0, "female": 1}
    return sex_map.get(input, -1)  # Return -1 for unknown sex

def transform_data(data, is_training: bool):
    transformed = data.copy()
    transformed['Title'] = transformed['Name'].apply(extract_title)
    transformed['EmbarkedN'] = transformed['Embarked'].apply(transform_embarks)
    transformed['SexN'] = transformed['Sex'].apply(transform_sex)
    
    columns_to_drop = ['Name', 'Embarked', 'Sex', 'Ticket', 'Cabin']
    
    transformed.drop(columns=columns_to_drop, inplace=True)
    if is_training:
        transformed.dropna(inplace=True)
    return transformed

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg == "--debug":
            DEBUG = True

    train_data = pd.read_csv('train.csv')
    train_X = transform_data(train_data, True)
    train_y = train_X['Survived']
    train_X.drop(columns=['Survived'], inplace=True)

    dprint(f"Shape of train_X: {train_X.shape}")
    dprint(f"Shape of train_y: {train_y.shape}")

    model = RandomForestRegressor(random_state=42)
    model.fit(train_X, train_y)

    test_data = pd.read_csv('test.csv')
    test_X = transform_data(test_data, False)

    dprint(f"Shape of test_X: {test_X.shape}")

    y_pred = model.predict(test_X)
    
    print("Predictions:")
    print(len(y_pred))

    # Optionally, you can save the predictions to a CSV file
    results = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred.round().astype(int)})
    results.to_csv('submission.csv', index=False)
    print("Predictions saved to 'submission.csv'")
