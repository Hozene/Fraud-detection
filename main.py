import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class FraudDetection:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.model = DecisionTreeClassifier()

    def preprocess_data(self):
        self.original_types = self.data["type"].copy()  # Keeping a copy of the original 'type' column for plotting
        self.data.fillna(self.data.median(), inplace=True)
        self.data['balanceDiff'] = self.data['newbalanceOrig'] - self.data['oldbalanceOrg']

        self.data["type"] = self.data["type"].map({
            "CASH_OUT": 1,
            "PAYMENT": 2,
            "CASH_IN": 3,
            "TRANSFER": 4,
            "DEBIT": 5
        })
        self.data["isFraud"] = self.data["isFraud"].map({
            0: "Not Fraud",
            1: "Fraud"
        })

    def analyze_feature_importance(self):
        feature_importances = self.model.feature_importances_
        features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "balanceDiff"]
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(importance_df['Feature'], importance_df['Importance'])
        plt.title('Feature Importance')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.show()


    def plot_transaction_types(self):
        type_counts = self.original_types.value_counts()
        transactions = type_counts.index
        quantity = type_counts.values

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            quantity, labels=transactions, autopct='%1.1f%%', wedgeprops=dict(width=0.3))
        '''
        autopct formats the percentage display.
        wedgeprops = dict() creates the hole in the middle.
        '''
        plt.setp(autotexts, size=10, weight="bold")
        ax.set_title("Transaction Types Distribution")
        plt.show()

    def train_model(self):
        x = np.array(self.data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
        y = np.array(self.data[["isFraud"]])

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=42)
        self.model.fit(xtrain, ytrain)

        score = self.model.score(xtest, ytest)
        print(f"Model Accuracy: {score}")

    def predict(self, features):
        prediction = self.model.predict(np.array(features).reshape(1, -1))
        print(f"Prediction: {prediction[0]}")
        return prediction[0]


if __name__ == '__main__':
    fraud_detection = FraudDetection("csv/onlinefraud.csv")
    fraud_detection.preprocess_data()
    fraud_detection.plot_transaction_types()
    fraud_detection.train_model()
    features = [[4, 9000.60, 9000.60, 0.0]]
    fraud_detection.predict(features)