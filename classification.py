import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

os.makedirs('outputs', exist_ok=True)


class RuleBasedHARClassifier:
    def fit(self, X, y):
        self.class_means = {}
        for cls in np.unique(y):
            self.class_means[cls] = X[y == cls].mean(axis=0).mean()
        return self

    def predict(self, X):
        preds = []
        sample_means = X.mean(axis=1)
        for m in sample_means:
            closest_class = min(
                self.class_means.keys(),
                key=lambda c: abs(self.class_means[c] - m)
            )
            preds.append(closest_class)
        return np.array(preds)


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Decision Tree":        DecisionTreeClassifier(criterion='gini', random_state=42),
        "Rule-Based Model":     RuleBasedHARClassifier(),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64),
                                               activation='relu',
                                               max_iter=300,
                                               random_state=42),
        "Naive Bayes":          GaussianNB(),
        "SVM (RBF Kernel)":     SVC(kernel='rbf', C=2, gamma='scale', random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred, digits=3))

        results[name] = acc

        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred),
                    annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(f'outputs/confusion_matrix_{name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Bar chart comparison
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), [v * 100 for v in results.values()], color='steelblue')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 100)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('outputs/model_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results