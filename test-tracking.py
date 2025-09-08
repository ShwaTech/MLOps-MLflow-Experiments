import mlflow

print("Printing tracking URI scheme below")
print(mlflow.get_tracking_uri())
print("\n------------------------------\n")


mlflow.set_tracking_uri("http://localhost:5000")

print("Printing tracking URI scheme below")
print(mlflow.get_tracking_uri())
print("\n------------------------------\n")

