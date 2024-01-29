# home-category-api
Home Stack Classification API using ML model

```shell
python -m grpc_tools.protoc -Iproto --python_out=interface/grpc/servicer/generated --pyi_out=interface/grpc/servicer/generated --grpc_python_out=interface/grpc/servicer/generated proto/expenseCategorizer.proto
```

```shell
pip install neptune
```
```shell
pip install neptune-notebooks
jupyter nbextension enable --py neptune-notebooks
```

```shell
docker build -t alokkusingh/home-category-api:latest -t alokkusingh/home-category-api:1.0.0 .
```
```shell
docker push alokkusingh/home-category-api:latest
```
```shell
docker push alokkusingh/home-category-api:1.0.0
```
```shell
docker run -p 5051:50051 --rm --name home-category-api alokkusingh/home-category-api
```
```shell
brew install grpcurl
```
```shell
grpcurl --plaintext localhost:5051 list
```
```shell
grpcurl -plaintext localhost:5051 describe
```
```shell
grpcurl -plaintext localhost:5051 describe interface.grpc.servicer.ExpenseCategorizer
```
```shell
grpcurl -plaintext localhost:5051 describe grpc.health.v1.Health
```
```shell
grpcurl -plaintext localhost:5051 grpc.health.v1.Health/Check
```
```shell
grpcurl -plaintext -d '{"head": "steamer"}' localhost:5051 interface.grpc.servicer.ExpenseCategorizer/getExpenseCategoryUnary
```
