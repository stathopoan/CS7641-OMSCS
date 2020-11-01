CS7641 MACHINE LEARNING
ASSIGNMENT 3

Description
------------
This code is for assignment 3 of machine learning course. It creates plots and results for the report.


Get the code
------------

Fetch the code from my public github repo: https://github.com/stathopoan/CS7641-OMSCS.git

Prerequisites
-------------
Please use Anaconda for easy creation of environment.

1. Navigate to the parent folder assignment3 and locate environment.yml 
2. Type: conda env create -f environment.yml

This will create the exact env required for the code to run. This code is based on common frameworks like sciki-learn

Run the code
-------------
To run the code just navigate to the parent folder assignment3.

For the whole assignment just run: run_experiment.py script



Directories
--------
All plots are saved in the plots folder.
plots
	heart(first dataset)
		YesDR (Everything related to Dimensionality Reduction algorithms)
		NoDR (Clustering without Dimensionality Reduction algorithms)
	wine(second dataset)
		YesDR (Everything related to Dimensionality Reduction algorithms)
		NoDR (Clustering without Dimensionality Reduction algorithms)

The data files are located in the data folder.

Notes
------

The script run_experiment.py will call all routines necessary. Please note that it takes about 15 minutes to run the code.

The code was developed in Anaconda env on Windows 10 pro system wih

Anaconda v. 4.8.2
platform: win64
python   v. 3.7.6

Thank you