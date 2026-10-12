import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

#Load Dataset....
df = pd.read_excel('OLA_Dataset.xlsx', sheet_name='July')

#  Data Preparation & Cleaning....
df_model = df[
    df['Booking_Status'].isin(
        ['Success', 'Canceled by Driver', 'Canceled by Customer']
    )
].copy()
df_model['Is_Canceled'] = df_model['Booking_Status'].apply(
    lambda x: 0 if x == 'Success' else 1
)

# Select predictive features (including wait times and distance)
features = ['Vehicle_Type', 'Payment_Method', 'Ride_Distance', 'V_TAT', 'C_TAT']
X = df_model[features].copy()
y = df_model['Is_Canceled']

# Handle missing values Data cleAning part...
X['V_TAT'] = X['V_TAT'].fillna(X['V_TAT'].median())
X['C_TAT'] = X['C_TAT'].fillna(X['C_TAT'].median())

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['Vehicle_Type', 'Payment_Method'], drop_first=True)

#Train-Test Split.....
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and Train the Decision Tree Model
# Increase max_depth to 4 so the tree has multiple splits to display
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# Evaluating the Model...
y_pred = dt_model.predict(X_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n')

# 6. Plot and Save 
plt.figure(figsize=(25, 12), dpi=300)
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['Success', 'Canceled'],
    filled=True,
    rounded=True,
    fontsize=9,
)

plt.title(
    'Decision Tree for OLA Ride Cancellation Prediction',
    fontsize=16,
    fontweight='bold',
    pad=20,
)
plt.tight_layout()

# Save img...
plt.savefig('Decision_Tree_OP.png', bbox_inches='tight')
plt.show()