<div align="center">
    <h1 align="center">Pneumonia Detection with BentoML</h1>
    <br>
    <strong> Healthcare AI 🫁🔍- Made Easy with BentoML<br></strong>
    <i>Powered by BentoML 🍱 + HuggingFace 🤗</i>
    <br>
</div>
<br>

## 📖 Introduction 📖
In this project, we showcase the seamless integration of an image detection model into a service using BentoML. Leveraging the power of the pretrained `nickmuchi/vit-finetuned-chest-xray-pneumonia model` from HuggingFace, users can submit their lung X-ray images for analysis. The model will then determine, with precision, whether the individual has pneumonia or not.


📝 **Disclaimer: Please note that this project is not intended to replace professional medical advice. It is designed purely for demonstration and testing purposes. Always consult with a qualified healthcare professional for a proper diagnosis.**

| Normal | Pneumonia                               	|
|------- |-----------------------------------------	|
| ![Normal](samples/NORMAL2-IM-1427-0001.jpeg)| ![Pneumonia](samples/person1950_bacteria_4881.jpeg) |

## 🏃‍♂️ Running the Service 🏃‍♂️
### BentoML CLI
Clone the repository and install the dependencies:
```bash
git clone https://github.com/bentoml/Pneumonia-Detection-demo.git && cd Pneumonia-Detection-demo

pip install -r requirements/pypi.txt
```

To serve the model with BentoML:
```
bentoml serve
```

You can then open your browser at http://127.0.0.1:3000 and interact with the service through Swagger UI.

### Containers
 We provide two pre-built containers optimized for CPU and GPU usage, respectively. 

To run the service, you'll need a container engine such as Docker, Podman, etc. Quickly test the service by running the appropriate container:

```bash
# cpu
docker run -p 3000:3000 ghcr.io/bentoml/pneumonia-detection-demo:cpu

# gpu
docker run --gpus all -p 3000:3000 ghcr.io/bentoml/pneumonia-detection-demo:gpu
```

## 🌐 Interacting with the Service 🌐
BentoML's default model serving method is through an HTTP server. In this section, we demonstrate various ways to interact with the service:
### cURL
```bash
curl -X 'POST' \
  'http://localhost:3000/v1/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: image/mpo' \
  --data-binary '@path-to-image'
```
> Replace `path-to-image` with the file path of the image you want to send to the service.

The response look like:
```json
{"class_name":"NORMAL"}
```
### Via BentoClient 🐍
To send requests in Python, one can use ``bentoml.client.Client`` to send requests to the service. Check out `client.py` for the example code.

### Swagger UI
You can use Swagger UI to quickly explore the available endpoints of any BentoML service.

## 🚀 Bringing it to Production 🚀
BentoML offers a number of options for deploying and hosting online ML services into production, learn more at [Deploying Bento Docs](https://docs.bentoml.org/en/latest/concepts/deploy.html).

## 👥 Community 👥
BentoML has a thriving open source community where thousands of ML/AI practitioners are 
contributing to the project, helping other users and discussing the future of AI. 👉 [Pop into our Slack community!](https://l.bentoml.com/join-slack)





