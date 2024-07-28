import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv('./creditcard.csv')

print(df[['Time', 'Amount']].describe())
print(f'Missing values in any column: {df.isnull().sum().max()}')

print(f'No Frauds: {round(df["Class"].value_counts()[0]/len(df) * 100,2)}% of the dataset')
print(f'Frauds: {round(df["Class"].value_counts()[1]/len(df) * 100,2)}% of the dataset')

sns.histplot(df['Amount'], kde=False, bins=50)
plt.title('Distribution of Transaction Amounts')
plt.show()

sns.histplot(df['Time'], kde=False, bins=50)
plt.title('Distribution of Transaction Times')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18, 4))
sns.histplot(df['Amount'], ax=ax[0], color='r', kde=True)
ax[0].set_title('Distribution of Transaction Amount')
sns.histplot(df['Time'], ax=ax[1], color='b', kde=True)
ax[1].set_title('Distribution of Transaction Time')
plt.show()

rob_scaler = RobustScaler()
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', df.pop('scaled_amount'))
df.insert(1, 'scaled_time', df.pop('scaled_time'))

print(df.head())

fraud_df = df[df['Class'] == 1]
non_fraud_df = df[df['Class'] == 0].sample(len(fraud_df), random_state=42)
balanced_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

print('Balanced dataset distribution:', dict(Counter(balanced_df['Class'])))

fig, ax = plt.subplots(1, 2, figsize=(18, 4))
sns.countplot(x='Class', data=df, ax=ax[0])
ax[0].set_title('Original Dataset Class Distribution')
sns.countplot(x='Class', data=balanced_df, ax=ax[1])
ax[1].set_title('Balanced Dataset Class Distribution')
plt.show()

corr_matrix = balanced_df.corr()

corr_with_class = corr_matrix["Class"].abs().sort_values(ascending=False)

top_features = corr_with_class.index[1:5]  
print(f'Top correlated features: {top_features.tolist()}')

fig, axes = plt.subplots(1, len(top_features), figsize=(20, 5))
for i, feature in enumerate(top_features):
    sns.boxplot(x='Class', y=feature, data=balanced_df, ax=axes[i])
    axes[i].set_title(f'{feature} vs Class')
plt.show()

X_balanced = balanced_df.drop('Class', axis=1)
y_balanced = balanced_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

ann_model = Sequential()
ann_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
ann_model.add(Dropout(0.5))
ann_model.add(Dense(32, activation='relu'))
ann_model.add(Dropout(0.5))
ann_model.add(Dense(1, activation='sigmoid'))

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = ann_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

loss, accuracy = ann_model.evaluate(X_test, y_test, verbose=0)
print(f'ANN Test Accuracy: {accuracy:.4f}')

rf_accuracy = rf_model.score(X_test, y_test)
loss, ann_accuracy = ann_model.evaluate(X_test, y_test, verbose=0)

accuracies = [rf_accuracy, ann_accuracy]
models = ['RandomForest', 'ANN']

plt.figure(figsize=(8, 6))
plt.scatter(models, accuracies)
plt.plot(models, accuracies, linestyle='--', color='r')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.005, f"{acc:.4f}", ha='center')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: RandomForest vs ANN')
plt.ylim(min(accuracies) - 0.01, max(accuracies) + 0.01)
plt.show()


plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

y_pred_prob_ann = ann_model.predict(X_test).ravel()

fpr_ann, tpr_ann, thresholds_ann = roc_curve(y_test, y_pred_prob_ann)
roc_auc_ann = auc(fpr_ann, tpr_ann)
plt.figure()
plt.plot(fpr_ann, tpr_ann, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_ann:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

precision_ann, recall_ann, _ann = precision_recall_curve(y_test, y_pred_prob_ann)
plt.figure()
plt.plot(recall_ann, precision_ann, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

rf_accuracy = rf_model.score(X_test, y_test)
print(f"RandomForest Accuracy: {rf_accuracy:.4f}")

ann_accuracy = ann_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"ANN Accuracy: {ann_accuracy:.4f}")

rf_report = classification_report(y_test, y_pred)
print("RandomForest Classification Report:")
print(rf_report)

ann_report = classification_report(y_test, (y_pred_prob_ann > 0.5).astype(int))
print("\nANN Classification Report:")
print(ann_report)

rf_conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('RandomForest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

ann_conf_matrix = confusion_matrix(y_test, (y_pred_prob_ann > 0.5).astype(int))
plt.figure(figsize=(8, 6))
sns.heatmap(ann_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('ANN Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

rf_fpr, rf_tpr, _ = roc_curve(y_test, y_pred_prob)
rf_roc_auc = auc(rf_fpr, rf_tpr)

ann_fpr, ann_tpr, _ = roc_curve(y_test, y_pred_prob_ann)
ann_roc_auc = auc(ann_fpr, ann_tpr)

plt.figure(figsize=(10, 8))
plt.plot(rf_fpr, rf_tpr, color='blue', lw=2, label=f'RandomForest ROC curve (AUC = {rf_roc_auc:.2f})')
plt.plot(ann_fpr, ann_tpr, color='red', lw=2, label=f'ANN ROC curve (AUC = {ann_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

rf_precision, rf_recall, _ = precision_recall_curve(y_test, y_pred_prob)
print("\nRandomForest Precision-Recall Curve:")
plt.figure()
plt.plot(rf_recall, rf_precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('RandomForest Precision-Recall Curve')
plt.show()

ann_precision, ann_recall, _ = precision_recall_curve(y_test, y_pred_prob_ann)
print("\nANN Precision-Recall Curve:")
plt.figure()
plt.plot(ann_recall, ann_precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ANN Precision-Recall Curve')
plt.show()