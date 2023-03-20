from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:

    '''
    this function will return the list of requirements
    '''

    reqs = []
    with open(file_path) as file_obj:
        get_requirements = file_obj.readline()
        reqs = [req.replace("\n", " ") for req in reqs]

        if HYPEN_E_DOT in reqs:
            reqs.remove(HYPEN_E_DOT)

    return reqs

setup(
    name = "MLOps project00",
    version = '0.0.1',
    auther = "Raza",
    author_email='rrizwan1998@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)