#! /bin/bash

# Create a .pypirc file in the home directory 
echo "[distutils]
index-servers =
	pypi

[pypi]" > ~/.pypirc

echo "username: " "$USERNAME" >> ~/.pypirc
echo "password: " "$PASSWORD" >> ~/.pypirc

python dev/tools/newrelease/register_pypi.py

# Delete the file again, we do not want to have the password lying around
rm -f ~/.pypirc
