from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements = [] 
    try:
        with open(file_path) as file_obj:
            requirements=file_obj.readlines()
            requirements=[req.replace("\n"," ") for req in requirements and not requirements.startswith('#')]

            if '-e .' in requirements:
                requirements.remove('-e .')
        return requirements
    except:
        print(f"Error: The file '{file_path}' was not found.")
        return requirements

setup(
    name="ML project",
    version="0.0.1",
    authon="shivam",
    author_email="shivamsingh8496@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)