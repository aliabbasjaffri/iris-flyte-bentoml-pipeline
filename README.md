# iris-flyte-bentoml-pipeline
Pipeline built on flyte for training Iris dataset using Pytorch and Weights and Biases for experiment tracking.
For some crazy reason, flyte starts to complain when using WandB for experiment tracking.
Everything runs fine if you're deploying the application without WandB.

```shell
# setup project and environment
cd iris_classifier
python3 -m venv venv
source venv/bin/activate

# update your pip while you're at it
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# to run flyte workflows locally..
python3 flyte/workflows/flyte_pipeline.py

# this automatically generates and deploys
# a model using bentoml

# to run this model to make predictions,
# you need to run the following
bentoml serve iris_classifier:latest
```
You'll see a nice workflow locally on your terminal with wandb experiement URL automatically generated at the end.
You can also view and test the api using swagger page after serving the bentoml model, at http://127.0.0.1:5000/.

For testing purpose, you'd need to pass in a [1,4] matrix to the `/classify` endpoint, which is in the order
```shell
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```
This will return a value after classifer has inferred the input.

Made with love with help from [pytorch](https://colab.research.google.com/github/RPI-DATA/course-intro-ml-app/blob/master/content/notebooks/20-deep-learning1/03-pytorch-iris.ipynb), 
[janakiev](https://janakiev.com/blog/pytorch-iris/) and [bentoml](https://github.com/bentoml/gallery/blob/main/quickstart/iris_classifier.ipynb). 
