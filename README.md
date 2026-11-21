# home-category-api
Home Stack Classification API using ML model

## Build gRPC Stubs
```bash
python -m grpc_tools.protoc -Iproto --python_out=interface/grpc/servicer/generated --pyi_out=interface/grpc/servicer/generated --grpc_python_out=interface/grpc/servicer/generated proto/expenseCategorizer.proto
```

## Train Model
```bash
python train_model.py
```

## Run Locally
```bash
# 1. Update ~/Downloads/Expense Sheet - Categorized.csv (or append to corrections.csv)
# 2. Run training and overwrite model/expense-pipeline.joblib
python train_model.py
```

## Build Container
```bash
docker build -t alokkusingh/home-category-api:latest -t alokkusingh/home-category-api:1.0.0 .
```
```bash
docker push alokkusingh/home-category-api:latest
```
```bash
docker push alokkusingh/home-category-api:1.0.0
```
### Run Locally
```bash
docker run -p 50051:50051 --rm --name home-category-api alokkusingh/home-category-api
```

```bash
pip install neptune
```
```bash
pip install neptune-notebooks
jupyter nbextension enable --py neptune-notebooks
```

```bash
brew install grpcurl
```
```bash
grpcurl --plaintext localhost:50051 list
```
```bash
grpcurl -plaintext localhost:50051 describe
```
```bash
grpcurl -plaintext localhost:50051 describe interface.grpc.servicer.ExpenseCategorizer
```
```bash
grpcurl -plaintext localhost:5051 describe grpc.health.v1.Health
```
```bash
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```
```bash
# Expected - Grocery
grpcurl -plaintext -d '{"head": "Chicken"}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
# Expected - Education
grpcurl -plaintext -d '{"head": "Drawing class", "comment": "Something", "amount": 500}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
# Expected - Celebration
grpcurl -plaintext -d '{"head": "birthday", "comment": "", "amount": 34000}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
grpcurl -plaintext -d '{"head": "fitness", "comment": "", "amount": 1090500}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
# Expected - Education
grpcurl -plaintext -d '{"head": "Stationary", "comment": "", "amount": 100}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
# Expected - Medical
grpcurl -plaintext -d '{"head": "Physiotherepist", "comment": "", "amount": 100}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
# Expected - Travel
grpcurl -plaintext -d '{"head": "Auto", "comment": "", "amount": 100}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
# Expected - Grocery
grpcurl -plaintext -d '{"head": "Dishwasher", "comment": "", "amount": 459}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
# Expected - Grocery
grpcurl -plaintext -d '{"head": "Dishwasher", "comment": "Rs 2556.00 at FLIPKART PAYMENTS on 10-10-2024 19%3A39%3A23,2556.0", "amount": 2556}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
# Expected - Appliances
grpcurl -plaintext -d '{"head": "Dishwasher", "comment": "INR 32472 at Flipkart In on 10-10-2024 17%3A59%3A32", "amount": 32472}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
# Expected - Fitness
grpcurl -plaintext -d '{"head": "Protein Powder", "comment": "", "amount": 5000}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```bash
kubectl replace --force -f /Users/aloksingh/git/home-stack/yaml/home-analytics-service.yaml --namespace=home-stack
```
```bash
kubectl replace --force -f /Users/aloksingh/git/home-stack/yaml/dashboard-service.yaml
```
