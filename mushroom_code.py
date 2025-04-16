import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.decomposition import PCA

column_names = ['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
mushroom = pd.read_csv('D:/My-Projects/Python/PrakMesin/H6/mushroom/agaricus-lepiota.data', header=None, names=column_names)
X = mushroom.drop('class', axis=1)  # Fitur
y = mushroom['class']  # Target

X_encoded = pd.get_dummies(X)
x_train, x_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=10)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

fig_tree, ax_tree = plt.subplots(figsize=(25, 20))
tree.plot_tree(model, feature_names=X_encoded.columns, filled=True)
plt.show()

fig_heatmap, ax_heatmap = plt.subplots(figsize=(7, 7))
sns.set(font_scale=1.4)
sns.heatmap(cm, ax=ax_heatmap, annot=True, annot_kws={"size": 16}, cmap="YlGnBu")
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_encoded)
y_label = y.map({'p': 'beracun', 'e': 'layak'})

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_label, palette={'beracun': 'red', 'layak': 'blue'})
plt.title('Visualisasi PCA Dataset Mushroom')
plt.xlabel('Kombinasi 1')
plt.ylabel('Kombinasi 2')
plt.legend(title='Kelas')
plt.show()

print("\nJumlah data latih:", len(x_train))
print("\nDeskripsi dataset:")
print(mushroom.describe(include='all').T)
print("\n10 Data teratas:")
print(mushroom.head(10))
print("\nDataset mushroom keseluruhan:")
print(mushroom)
