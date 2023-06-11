from setuptools import find_packages, setup
from typing import List


hyphen_e_dot = "-e."
def get_requirements(file_path:str)->List[str]:
    """
    This function returns list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n", "") for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
        


setup(
    name = "ML_Project",
    version = "0.0.1",
    author = "Rahmon",
    author_email = "rahmonadeniji317@gmai.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")

    
)