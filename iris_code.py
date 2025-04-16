from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

iris = sklearn_to_df(datasets.load_iris())
iris.rename(columns={'target': 'species'}, inplace=True)

x = iris.drop('species', axis=1)
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
iris_test_data = {'sepal length (cm)': 5.1, 'sepal width (cm)': 3.5, 'petal length (cm)': 1.4, 'petal width (cm)': 0.1
}

feature_order = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
prediction_input_df = pd.DataFrame([iris_test_data])
prediction = model.predict(prediction_input_df[feature_order])
print("Prediksi untuk data uji coba manual:", prediction)

fig_tree, ax_tree = plt.subplots(figsize=(25, 20))
tree.plot_tree(model, feature_names=features, filled=True)
plt.show()

fig_heatmap, ax_heatmap = plt.subplots(figsize=(7, 7))
sns.set(font_scale=1.4)
sns.heatmap(cm, ax=ax_heatmap, annot=True, annot_kws={"size": 16})

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nData uji (x_test):")
print(x_test)

sns.pairplot(iris, hue='species', palette='Set1')
plt.show()

print("\nJumlah data latih:", len(x_train))
print("\nDeskripsi dataset:")
print(iris.describe().T)
print("\n10 Data teratas:")
print(iris.head(10))
print("\nDataset iris keseluruhan:")
print(iris)
