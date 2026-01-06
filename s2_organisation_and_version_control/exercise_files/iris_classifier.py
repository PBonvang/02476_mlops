from pathlib import Path
from typing import Annotated

import joblib
import typer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = typer.Typer()
@app.command()
def train(output: Annotated[Path, typer.Option("--output", "-o")] = Path("models/iris_model.pkl")):
    """Train and evaluate the model."""
    # Load the dataset
    data = load_breast_cancer()
    x = data.data
    y = data.target

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel="linear", random_state=42)
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, output)
    print(f"Artifacts saved to {output}")

    return accuracy, report


# this "if"-block is added to enable the script to be run from the command line
if __name__ == "__main__":
    app()