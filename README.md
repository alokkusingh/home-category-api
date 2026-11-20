# home-category-api
Home Stack Classification API using ML model

## Build gRPC Stubs
```shell
python -m grpc_tools.protoc -Iproto --python_out=interface/grpc/servicer/generated --pyi_out=interface/grpc/servicer/generated --grpc_python_out=interface/grpc/servicer/generated proto/expenseCategorizer.proto
```

## Train Model
```shell
python train_model.py
```

## Run Locally
```shell
python main.py 
```

## Build Container
```shell
docker build -t alokkusingh/home-category-api:latest -t alokkusingh/home-category-api:1.0.0 .
```
```shell
docker push alokkusingh/home-category-api:latest
```
```shell
docker push alokkusingh/home-category-api:1.0.0
```
### Run Locally
```shell
docker run -p 50051:50051 --rm --name home-category-api alokkusingh/home-category-api
```

```shell
pip install neptune
```
```shell
pip install neptune-notebooks
jupyter nbextension enable --py neptune-notebooks
```

```shell
brew install grpcurl
```
```shell
grpcurl --plaintext localhost:50051 list
```
```shell
grpcurl -plaintext localhost:50051 describe
```
```shell
grpcurl -plaintext localhost:50051 describe interface.grpc.servicer.ExpenseCategorizer
```
```shell
grpcurl -plaintext localhost:5051 describe grpc.health.v1.Health
```
```shell
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```
```shell
grpcurl -plaintext -d '{"head": "Chicken"}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```shell
grpcurl -plaintext -d '{"head": "Drawing class", "comment": "Something", "amount": 500}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```shell
grpcurl -plaintext -d '{"head": "birthday", "comment": "party", "amount": 1090500}' 127.0.0.1:50051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
```shell
kubectl replace --force -f /Users/aloksingh/git/home-stack/yaml/home-analytics-service.yaml --namespace=home-stack
```
```shell
kubectl replace --force -f /Users/aloksingh/git/home-stack/yaml/dashboard-service.yaml
```
