echomessage:
	echo hello

installintelpython:
    conda update conda
    conda config --add channels intel
    conda create -n uiexp intelpython3_full python=3.8 -y
    conda init bash

installconda:
	wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
	chmod +x Miniconda3-py38_4.12.0-Linux-x86_64.sh
	./Miniconda3-py38_4.12.0-Linux-x86_64.sh

removecondabase:
	conda config --set auto_activate_base false
	rm -rf Miniconda3-py38_4.12.0-Linux-x86_64.sh
	conda init bash

installenv:
#Downloading the latest Miniconda installer for Linux. Your architecture may vary.
	# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
	# bash ~/miniconda.sh -b -p $HOME/miniconda
	conda create -n uiexp python=3.8 -y && conda activate uiexp && conda install -y -c conda-forge mlflow-pipelines
	python -m pip install faiss-cpu sentence-transformers
	python -m pip install pandas numpy matplotlib
	python -m pip install git+https://github.com/neuml/codequestion

installp311:
	sudo apt install software-properties-common
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt update
	sudo apt install python3.11
	python3.11

configp311:
	sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
	sudo apt install python3.11-dev python3.11-venv python3.11-distutils python3.11-gdbm python3.11-tk python3.11-lib2to3