import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import ttk, messagebox
import graphviz
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace "4" with the number of cores you have


class FraudDetection:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.model = DecisionTreeClassifier()
        self.type_mapping = {
            "CASH_OUT": 1,
            "PAYMENT": 2,
            "CASH_IN": 3,
            "TRANSFER": 4,
            "DEBIT": 5
        }

    def preprocess_data(self):
        self.original_types = self.data["type"].copy()  # Keeping a copy of the original 'type' column for plotting
        self.data["type"] = self.data["type"].map(self.type_mapping)
        self.data["isFraud"] = self.data["isFraud"].map({
            0: "Not Fraud",
            1: "Fraud"
        })

    def analyze_feature_importance(self):
        feature_importances = self.model.feature_importances_
        features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig"]
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

    def plot_clusters(self):
        transaction_types = self.original_types.unique()
        for transaction_type in transaction_types:
            type_data = self.data[self.original_types == transaction_type]
            features = type_data[["amount", "oldbalanceOrg"]].values
            if len(features) > 1000:  # Limit the number of data points for clustering
                features = features[:1000]
            kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
            clusters = kmeans.predict(features)

            plt.figure(figsize=(8, 6))
            plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap='viridis')
            plt.xlabel('Amount')
            plt.ylabel('Old Balance')
            plt.title(f'Clustering of Transaction Amounts for Type {transaction_type}')
            plt.show()

    def plot_decision_tree(self):
        dot_data = export_graphviz(self.model, out_file=None,
                                   feature_names=["type", "amount", "oldbalanceOrg", "newbalanceOrig"],
                                   class_names=["Not Fraud", "Fraud"],
                                   filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data, format='svg')
        graph.render('decision_tree', format='svg', view=True)

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

    def create_ui(self):
        def predict_transaction():
            try:
                transaction_type = transaction_type_var.get()
                amount = float(amount_var.get())
                old_balance = float(old_balance_var.get())

                if transaction_type not in self.type_mapping:
                    raise ValueError("Unknown transaction type selected.")

                if transaction_type in ["CASH_OUT", "TRANSFER", "DEBIT"]:
                    new_balance = old_balance - amount
                elif transaction_type in ["CASH_IN", "PAYMENT"]:
                    new_balance = old_balance + amount

                features = [[self.type_mapping[transaction_type], amount, old_balance, new_balance]]
                result = self.predict(features)
                messagebox.showinfo("Prediction Result", f"The transaction is {result}.")
            except ValueError as e:
                messagebox.showerror("Error", str(e))

        def plot_decision_tree_wrapper():
            try:
                self.plot_decision_tree()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to plot Decision Tree: {str(e)}")

        root = tk.Tk()
        root.title("Fraud Detection")

        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 12))
        style.configure('TButton', font=('Arial', 12))

        frame = ttk.Frame(root, padding=(20, 10))
        frame.grid(row=0, column=0, sticky='nsew')

        ttk.Label(frame, text="Transaction Type:", style='TLabel').grid(row=0, column=0, padx=10, pady=5, sticky='w')
        transaction_type_var = tk.StringVar()
        transaction_type_menu = ttk.OptionMenu(frame, transaction_type_var, "", *self.type_mapping.keys())
        transaction_type_menu.grid(row=0, column=1, padx=10, pady=5, sticky='w')

        ttk.Label(frame, text="Amount:", style='TLabel').grid(row=1, column=0, padx=10, pady=5, sticky='w')
        amount_var = tk.StringVar()
        ttk.Entry(frame, textvariable=amount_var).grid(row=1, column=1, padx=10, pady=5, sticky='w')

        ttk.Label(frame, text="Account Balance:", style='TLabel').grid(row=2, column=0, padx=10, pady=5, sticky='w')
        old_balance_var = tk.StringVar()
        ttk.Entry(frame, textvariable=old_balance_var).grid(row=2, column=1, padx=10, pady=5, sticky='w')

        ttk.Button(root, text="Predict", command=predict_transaction, style='TButton').grid(row=1, column=0, padx=20, pady=10)
        ttk.Button(root, text="Plot Decision Tree", command=plot_decision_tree_wrapper, style='TButton').grid(row=2, column=0, padx=20, pady=10)

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        root.mainloop()


if __name__ == '__main__':
    fraud_detection = FraudDetection("csv/onlinefraud.csv")
    fraud_detection.preprocess_data()
    fraud_detection.plot_transaction_types()
    fraud_detection.plot_clusters()
    fraud_detection.train_model()
    features = [[4, 9000.60, 9000.60, 0.0]]
    fraud_detection.predict(features)
    fraud_detection.analyze_feature_importance()
    # fraud_detection.plot_decision_tree()
    # fraud_detection.create_ui()
