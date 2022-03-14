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
```
You'll see a nice workflow locally on your terminal with wandb experiement URL automatically generated at the end.


Made with love with help from [pytorch](https://colab.research.google.com/github/RPI-DATA/course-intro-ml-app/blob/master/content/notebooks/20-deep-learning1/03-pytorch-iris.ipynb), 
[janakiev](https://janakiev.com/blog/pytorch-iris/) and [bentoml](https://github.com/bentoml/gallery/blob/main/quickstart/iris_classifier.ipynb). 
