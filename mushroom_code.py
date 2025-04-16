import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

column_names = [
    'class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment','gill_spacing', 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root','stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring','stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type','spore_print_color', 'population', 'habitat']

mushroom = pd.read_csv(
    'D:/My-Projects/Python/PrakMesin/H6/mushroom/agaricus-lepiota.data',
    header=None, names=column_names
)

mushroom['class'] = mushroom['class'].map({'p': 'beracun', 'e': 'layak'})
X = mushroom.drop('class', axis=1)
y = mushroom['class']

X_encoded = pd.get_dummies(X)
x_train, x_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=10)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(25, 20))
tree.plot_tree(model, feature_names=X_encoded.columns, filled=True, class_names=model.classes_)
plt.title("Visualisasi Decision Tree")
plt.show()
plt.figure(figsize=(7, 7))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Prediksi', fontsize=16)
plt.ylabel('Aktual', fontsize=16)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nJumlah data latih:", len(x_train))
print("\nDeskripsi dataset:")
print(mushroom.describe(include='all').T)

selected_cols = ['cap_shape', 'cap_color', 'odor', 'gill_size', 'class']
mushroom_subset = mushroom[selected_cols].copy()
le = LabelEncoder()
for col in selected_cols:
    mushroom_subset[col] = le.fit_transform(mushroom_subset[col])

sns.pairplot(mushroom_subset, hue='class', palette='Set1')
plt.suptitle('Pairplot Fitur Terpilih (Encoded)', y=1.02)
plt.show()
