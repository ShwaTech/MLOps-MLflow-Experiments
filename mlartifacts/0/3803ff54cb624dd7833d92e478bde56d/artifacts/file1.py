import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

## So IMPORTANT !!
mlflow.set_tracking_uri("http://localhost:5000")

## Load Wine Dataset
wine = load_wine()
X = wine.data
y = wine.target

## Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

## Define the Parameters for RandomForest Classifier Model
n_estimators = 60
max_depth = 10


## Start MLflow Experiment
with mlflow.start_run():
    ## Initialize the Model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    ## Train the Model
    rf.fit(X_train, y_train)
    
    ## Predict on Test Data
    y_pred = rf.predict(X_test)
    
    ## Evaluate the Model
    accuracy = accuracy_score(y_test, y_pred)
    
    ## Log Parameters, Metrics and Model to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    ## Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    ## Save the Confusion Matrix
    plt.savefig("confusion_matrix.png")
    
    ## Log the Confusion Matrix to MLflow
    mlflow.log_artifact("confusion_matrix.png")

    ## Log Our File to MLflow
    mlflow.log_artifact(__file__)
    
    ## Print the Evaluation Report
    print("Accuracy:", accuracy)
    


