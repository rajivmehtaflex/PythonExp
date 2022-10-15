installenv:
#Downloading the latest Miniconda installer for Linux. Your architecture may vary.
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
	bash ~/miniconda.sh -b -p $HOME/miniconda
	conda create -n uiexp python=3.8 -y && conda activate uiexp && conda install -c conda-forge mlflow-pipelines