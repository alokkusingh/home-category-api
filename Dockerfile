#FROM arm64v8/python:alpine3.9
FROM python:3.9-slim

WORKDIR /Users/aloksingh/git/home-category-api

COPY requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt \
  # Clean up
  && rm requirements.txt

COPY . .

CMD [ "python3", "./main.py" ]