install_env:
	pyenv virtualenv 3.10.6 RTL
	pyenv local RTL
	pip install --upgrade pip
	pip install -r https://gist.githubusercontent.com/krokrob/53ab953bbec16c96b9938fcaebf2b199/raw/9035bbf12922840905ef1fbbabc459dc565b79a3/minimal_requirements.txt
	pip install gym
	pip install swig
	pip install pyproject
	pip install pyproject.toml
	pip install 'gym[box2d]'

delete_env:
	pyenv virtualenv-delete RTL -y
