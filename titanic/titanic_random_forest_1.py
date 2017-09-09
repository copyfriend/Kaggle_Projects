import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
dataset = pd.read_csv(SCRIPT_PATH + "/train.csv")

#features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
training_data = dataset[features]

training_label = dataset['Survived']
training_data = training_data.fillna(0)
row_index = training_data['Cabin']==0
training_data.loc[row_index, 'Cabin'] = 'U'
row_index = training_data['Embarked']==0
training_data.loc[row_index, 'Embarked'] = 'S'
#training_data['Name'] = LabelEncoder().fit_transform(training_data['Name'])

training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'])
#training_data['Ticket'] = LabelEncoder().fit_transform(training_data['Ticket'])

training_data['Cabin'] = LabelEncoder().fit_transform(training_data['Cabin'])
training_data['Embarked'] = LabelEncoder().fit_transform(training_data['Embarked'])

model = RandomForestClassifier()
model.fit(training_data, training_label)

y_pos = np.arange(len(features))
plt.barh(y_pos, model.feature_importances_, align='center', alpha=0.4)
plt.yticks(y_pos, features)
plt.xlabel('features')
plt.title('feature_importances')
plt.show()
#from above plt, it showed that Sex/Age/Fare matters in the importance
features = ['Sex', 'Age', 'Fare', 'Cabin', 'Embarked']
training_data = dataset[features]
training_data = training_data.fillna(0)
row_index = training_data['Sex']==0
training_data.loc[row_index, 'Sex'] = 1 if np.random.rand(1,1)>0.5 else 0
row_index = training_data['Age']==0
training_data.loc[row_index, 'Age'] = training_data['Age'].mean()
row_index = training_data['Fare']==0
training_data.loc[row_index, 'Fare'] = training_data['Fare'].mean()
row_index = training_data['Cabin']==0
training_data.loc[row_index, 'Cabin'] = 'U'
row_index = training_data['Embarked']==0
training_data.loc[row_index, 'Embarked'] = 'S'
training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'])#sex needs to label encode
training_data['Cabin'] = LabelEncoder().fit_transform(training_data['Cabin'])#Cabin needs to label encode
training_data['Embarked'] = LabelEncoder().fit_transform(training_data['Embarked'])#Embarked needs to label encode
model = RandomForestClassifier()
model.fit(training_data, training_label)

test_data = pd.read_csv(SCRIPT_PATH + "/test.csv")
preidction_data = test_data[features]
preidction_data = preidction_data.fillna(0)
row_index = preidction_data['Sex']==0
preidction_data.loc[row_index, 'Sex'] = 1 if np.random.rand(1,1)>0.5 else 0
row_index = preidction_data['Cabin']==0
preidction_data.loc[row_index, 'Cabin'] = 'U'
row_index = preidction_data['Embarked']==0
preidction_data.loc[row_index, 'Embarked'] = 'S'
preidction_data['Sex'] = LabelEncoder().fit_transform(preidction_data['Sex'])#only sex needs to label encode
preidction_data['Cabin'] = LabelEncoder().fit_transform(preidction_data['Cabin'])#Cabin needs to label encode
preidction_data['Embarked'] = LabelEncoder().fit_transform(preidction_data['Embarked'])#Embarked needs to label encode
result_labels = model.predict(preidction_data)
results = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : result_labels
})
results.to_csv(SCRIPT_PATH + "/submission7.csv", index=False)

cv_scores = cross_val_score(model, training_data, training_label, scoring = 'roc_auc', cv=5)
print(cv_scores)