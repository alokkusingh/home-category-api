# home-category-api
Home Stack Categorisation API using ML model

```shell
pip install grpcio-tools
```

```shell
python -m grpc_tools.protoc -Iproto --python_out=stream/grpc/stub --pyi_out=stream/grpc/stub --grpc_python_out=stream/grpc/stub proto/expenseCategorizer.proto
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
grpcurl --plaintext localhost:50051 list
```
```shell
grpcurl -plaintext localhost:50051 describe
```
```shell
grpcurl -plaintext localhost:50051 describe ExpenseCategorizer
```
```shell
grpcurl -plaintext localhost:50051 describe grpc.health.v1.Health
```
```shell
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```
```shell
grpcurl -plaintext -d '{"head": "Netflix recharge"}' localhost:50051 ExpenseCategorizer/getExpenseCategoryUnary
```
