# This project is a part of the "Computer Vision" course at the Chulalongkorn University.

# Overview

This project provides API gateways for accessing the Detection Model which responsible for detecting pomelo from images and classifying them into 3 classes: "young", "old" and "pomelo", which equal to "don't know", All the module is containerized using Docker and can be deployed on any server that supports Docker. The project is designed to be easily extensible and maintainable, allowing for future improvements and additions.

# Project Structure

```plaintext
.
├── configs
│   ├── ...all model's configs...
├── data
│   └── ...data folder, which loaded by `download_image.ipynb`
├── docker
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── models
│   ├── ...all model's weights...
├── notebooks
│   ├── download_image.ipynb
├── src
│   ├── ...all source codes...
├── .env
├── pyproject.toml
├── requirements.txt
```

# Installation
1. Clone the repository:
    ```bash
    git clone
    cd pomelo-detection
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Create a `.env` file in the root directory and add the following environment variables like shown in the `.env.example` file:
4. Install project
    ```bash
    pip install -e .
    ```
    
or install with docker

```bash
cd docker
docker-compose up --build
```
