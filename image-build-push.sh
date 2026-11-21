#!/usr/bin/env bash
# Build and Publish
docker build -t alokkusingh/home-category-api:latest -t alokkusingh/home-category-api:1.0.0 .
docker push alokkusingh/home-category-api:latest
docker push alokkusingh/home-category-api:1.0.0