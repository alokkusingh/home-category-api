# home-category-api
Home Stack Categorisation API using ML model

```shell
pip install grpcio-tools
```

```shell
python -m grpc_tools.protoc -Iproto --python_out=stream/grpc/stub --pyi_out=stream/grpc/stub --grpc_python_out=stream/grpc/stub proto/expenseCategorizer.proto
```
