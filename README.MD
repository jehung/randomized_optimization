# Comparison of Randomized Optimization Methods

## Before You Run the Files
1) This project uses a modified version of ABAGAIL, located in the ABAGAIL sub-folder
2) The folders NNOUTPUT, CONTPEAKS, FLIPFLOP and TSP must be created in the same folder as the Jython code before running it.
3) The files sample_orig_2016.txt, sample_svcg_2016.txt must be in the same folder as the .py files
4) To run the JAVA files, you must have jpype installed. Do this before running the code in terminal / command line:
- `git clone https://github.com/originell/jpype.git`
- `cd jpype`
- `python setup.py install'`

Additional reference can be found here: https://stackoverflow.com/questions/35736763/practical-use-of-java-class-jar-in-python


## Explanation of Files

The zip folder includes the following files:

#### Python code files
- `NN0.py`: Code for Backpropagation training of neural network
- `NN1.py`: Code for Randomised Hill Climbing training of neural network
- `NN2.py`: Code for Simulated Annleaing training of neural network
- `NN3.py`: Code for Genetic Algorithm training of neural network
- `continuouspeaks.py`: Code to use Randomised Optimisation to solve the Continuous Peaks problem
- `tsp.py`: Code to use Randomised Optimisation to solve the Traveling Salesman Problem
- `flipflop.py`: Code to use Randomised Optimisation to solve the Flip Flop Problem
- `optimization_analysis.py`: Code to do plotting and computation of summary statistics

#### Data files
- `sample_orig_2016.txt`
- `sample_svcg_2016.txt`


#### Resutls folders
- `NNOUTPUT`: Output folder for the Neural Network experiments
- `CONTPEAKS`: Output folder for the Continuous Peaks experiments
- `FLIPFLOP`: Output folder for the Flip Flop experiments
- `TSP`: Output folder for the Traveling Salesman Problem experiments
- `ABAGAIL`: folder with source, ant build file, and jar for ABAGAIL

#### Report
- `jhung34-analysis.pdf`

## License

The content of this repository is licensed under a
[Creative Commons Attribution License](http://creativecommons.org/licenses/by/3.0/us/)
