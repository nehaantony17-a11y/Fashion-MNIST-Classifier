import mlflow
import pandas as pd

# Set experiment name
mlflow.set_experiment("Fashion_MNIST_SVM_Comparison")

# Read comparison results
results = pd.read_csv("results/model_comparison.csv")

# Track each model
for _, row in results.iterrows():

    with mlflow.start_run(run_name=row["Model"]):

        mlflow.log_param("model", row["Model"])
        mlflow.log_param("test_samples", int(row["Test_Samples"]))

        mlflow.log_metric("accuracy", row["Accuracy"])
        mlflow.log_metric("precision", row["Precision"])
        mlflow.log_metric("recall", row["Recall"])
        mlflow.log_metric("f1_score", row["F1_Score"])

        mlflow.log_artifact("results/model_comparison.csv")

        print(f"Logged {row['Model']} successfully!")

print("All experiments logged to MLflow!")