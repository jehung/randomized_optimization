"""
RHC NN training on Madelon data (Feature selection complete)

"""

"""
git clone https://github.com/originell/jpype.git
cd jpype
python setup.py install'

in terminal / command line
"""

"""https://stackoverflow.com/questions/35736763/practical-use-of-java-class-jar-in-python"""

import os
import csv
import time
import sys
sys.path.append('/Users/jennyhung/MathfreakData/School/OMSCS_ML/Assign2/abagail_py/ABAGAIL/ABAGAIL.jar')
print(sys.path)
import jpype as jp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#jp.startJVM(jp.getDefaultJVMPath(), "-ea")

jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=/Users/jennyhung/MathfreakData/School/OMSCS_ML/Assign2/abagail_py/ABAGAIL/ABAGAIL.jar')
jp.java.lang.System.out.println("hello world")
jp.java.func.nn.backprop.BackPropagationNetworkFactory
jp.java.func.nn.backprop.RPROPUpdateRule
jp.java.func.nn.backprop.BatchBackPropagationTrainer
jp.java.shared.SumOfSquaresError
jp.java.shared.DataSet
jp.java.shared.Instance
jp.java.opt.SimulatedAnnealing
jp.java.opt.example.NeuralNetworkOptimizationProblem
jp.java.opt.RandomizedHillClimbing
jp.java.ga.StandardGeneticAlgorithm
jp.java.func.nn.activation.RELU


BackPropagationNetworkFactory = jp.JPackage('func').nn.backprop.BackPropagationNetworkFactory
DataSet = jp.JPackage('shared').DataSet
SumOfSquaresError = jp.JPackage('shared').SumOfSquaresError
NeuralNetworkOptimizationProblem = jp.JPackage('opt').example.NeuralNetworkOptimizationProblem
RandomizedHillClimbing = jp.JPackage('opt').RandomizedHillClimbing
Instance = jp.JPackage('shared').Instance
RELU = jp.JPackage('func').nn.activation.RELU


INPUT_LAYER = 109
HIDDEN_LAYER1 = 100
HIDDEN_LAYER2 = 100
HIDDEN_LAYER3 = 100
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5001
OUTFILE = 'LOG.txt'


def initialize_instances(infile):
    """Read the train.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        next(dat)
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(float(row[-1])))
            instances.append(instance)

    return instances



def errorOnDataSet(network,ds,measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted,1),0)
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    return MSE,acc


def train(oa, network, oaName, training_ints,validation_ints,testing_ints, measure):
    """Train a given network on a set of instances.
    """
    print("\nError results for %s\n---------------------------" % (oaName,))
    times = [0]
    for iteration in range(TRAINING_ITERATIONS):
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
        times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
            MSE_trg, acc_trg = errorOnDataSet(network,training_ints,measure)
            MSE_val, acc_val = errorOnDataSet(network,validation_ints,measure)
            MSE_tst, acc_tst = errorOnDataSet(network,testing_ints,measure)
            txt = '{},{},{},{},{},{},{},{}\n'.format(iteration,MSE_trg,MSE_val,MSE_tst,acc_trg,acc_val,acc_tst,times[-1]);print(txt)
            with open(OUTFILE,'a+') as f:
                f.write(txt)

def main():
    """Run this experiment"""
    training_ints = initialize_instances('train.csv')
    testing_ints = initialize_instances('test.csv')
    validation_ints = initialize_instances('train.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    #rule = RPROPUpdateRule()
    oa_names = ["RHC"]
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1,HIDDEN_LAYER2,HIDDEN_LAYER3, OUTPUT_LAYER],relu)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = RandomizedHillClimbing(nnop)
    train(oa, classification_network, 'RHC', training_ints,validation_ints,testing_ints, measure)



if __name__ == "__main__":
    with open(OUTFILE,'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_val','MSE_tst','acc_trg','acc_val','acc_tst','elapsed'))
    main()
    jp.shutdownJVM()
