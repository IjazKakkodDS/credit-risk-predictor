from setuptools import find_packages,setup
from typing import List

hyphen_e_dot ='-e .'

def get_requirements(file_path:str)->List[str]:
    # Function to return a list of requirements.
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
    
    return requirements

# Metadata Information
setup( 
    name = 'score_predictor',
    version = '0.0.1',
    author = 'Ijaz Kakkod',
    author_email = 'ijazkakkod@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    )