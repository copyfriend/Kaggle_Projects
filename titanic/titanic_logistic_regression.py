import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
dataset = pd.read_csv(SCRIPT_PATH + "/train.csv")

#features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
training_data = dataset[features]

training_label = dataset['Survived']

#from above plt, it showed that Sex/Age/Fare matters in the importance
features = ['Sex', 'Age', 'Fare']
training_data = dataset[features]
training_data = training_data.fillna(0)
training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'])#only sex needs to label encode
model = LogisticRegression(penalty='l2', class_weight='balanced', solver='newton-cg')
model.fit(training_data, training_label)

test_data = pd.read_csv(SCRIPT_PATH + "/test.csv")
preidction_data = test_data[features]
preidction_data = preidction_data.fillna(0)
preidction_data['Sex'] = LabelEncoder().fit_transform(preidction_data['Sex'])#only sex needs to label encode
result_labels = model.predict(preidction_data)
results = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : result_labels
})
results.to_csv(SCRIPT_PATH + "/submission5.csv", index=False)

cv_scores = cross_val_score(model, training_data, training_label, scoring = 'roc_auc', cv=5)
print(cv_scores)