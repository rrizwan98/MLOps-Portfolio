from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    description="its a IRIS Flower Dataset", 
    author="Raza Rizwan", 
    packages=find_packages()
)


# # local package install
# -e .

# # third party packages
# python
# pandas==1.4.2
# numpy==1.22.3
# scikit-learn==1.0.2
# dvc==2.10.1
# dvc[gdrive]
# pytest==7.1.1
# tox==1.2.0
# flask8==4.0.1
# # flask==2.1.1
# # gunicorn==20.1.0
# # mlflow==1.25.1