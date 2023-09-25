# Audio-Genre-Classification

## Workflows


1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

ECR Repo: 925304758738.dkr.ecr.eu-north-1.amazonaws.com/audio-genre-classification


# AZURE-CICD-Deployment-with-Github-Actions

## Save pass:

jkb9HFDzOf6ZhMKIzUJTH1pcLayqQrpyd9HhrkUXoq+ACRAJcqcW


## Run from terminal:

docker build -t agcapp.azurecr.io/agc:latest .

docker login agcapp.azurecr.io

docker push agcapp.azurecr.io/agc:latest
