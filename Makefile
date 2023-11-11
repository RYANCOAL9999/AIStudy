.PHONY: help
help:
	@echo "Usage:"
	@echo "		virtualenv       Set up virtual environment"
	@echo "		activate         Activate virtual environment"
	@echo "		source           Source environment variable"
	@echo "		install          Install requirements.txt"
	@echo "     drive            Operate a drivingCar.py file"
	@echo "     poc            	 Operate a poc.py file"
	@echo "     sklearn          Operate a sklearn.py file"
	@echo "		min          	 Operate a textMining.py file"
	@echo "	    act 			 Operate a activities.py file"

.PHONY: virtualenv
virtualenv:
	virtualenv venv -p python3

.PHONY: activate
activate:
	. venv/bin/activate

.PHONY: source
source:
	source .env

.PHONY: install
install:
	pip3 install --trusted-host pypi.python.org -r requirements.txt

.PHONY: drive
drive:
	python3 model/drivingCar.py

.PHONY: poc
poc:
	python3 model/poc.py

.PHONY: sklearn
sklearn:
	python3 model/sklearn.py

.PHONY: min
min:
	python3 model/textMining.py

.PHONY: act
act:
	python3 assessment/activate.py