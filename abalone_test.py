"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""
import os
import csv
import time
import sys
sys.path.append('/Users/jennyhung/MathfreakData/School/OMSCS_ML/Assign2/abagail_py/ABAGAIL/ABAGAIL.jar')
print(sys.path)
import jpype as jp
import get_all_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
SimulatedAnnealing = jp.JPackage('opt').SimulatedAnnealing
StandardGeneticAlgorithm = jp.JPackage('opt').ga.StandardGeneticAlgorithm
Instance = jp.JPackage('shared').Instance
RELU = jp.JPackage('func').nn.activation.RELU


INPUT_LAYER = 109
HIDDEN_LAYER = 5
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 1000



def get_all_data():
    #dir = 'D:\\Backups\\StemData\\'
    files = ['sample_orig_2016.txt', 'sample_orig_2015.txt', 'sample_orig_2014.txt', 'sample_orig_2013.txt',
             'sample_orig_2012.txt', 'sample_orig_2011.txt',
             'sample_orig_2010.txt', 'sample_orig_2009.txt', 'sample_orig_2008.txt', 'sample_orig_2007.txt']

    files1 = ['sample_svcg_2016.txt', 'sample_svcg_2015.txt', 'sample_svcg_2014.txt', 'sample_svcg_2013.txt',
              'sample_svcg_2012.txt', 'sample_svcg_2011.txt',
              'sample_svcg_2010.txt', 'sample_svcg_2009.txt', 'sample_svcg_2008.txt', 'sample_svcg_2008.txt']

    merged_data = pd.DataFrame()
    for i in [0]:
        print(files[i])
        raw = pd.read_csv(files[i], sep='|', header=None, low_memory=False)
        raw.columns = ['credit_score', 'first_pmt_date', 'first_time', 'mat_date', 'msa', 'mi_perc', 'units',
                       'occ_status', 'ocltv', 'odti', 'oupb', 'oltv', 'oint_rate', 'channel', 'ppm', 'fixed_rate',
                       'state', 'prop_type', 'zip', 'loan_num', 'loan_purpose', 'oterm', 'num_borrowers', 'seller_name',
                       'servicer_name', 'exceed_conform']

        raw1 = pd.read_csv(files1[i], sep='|', header=None, low_memory=False)
        raw1.columns = ['loan_num', 'yearmon', 'curr_upb', 'curr_delinq', 'loan_age', 'remain_months', 'repurchased',
                        'modified', 'zero_bal', 'zero_date', 'curr_rate', 'curr_def_upb', 'ddlpi', 'mi_rec',
                        'net_proceeds',
                        'non_mi_rec', 'exp', 'legal_costs', 'maint_exp', 'tax_insur', 'misc_exp', 'loss', 'mod_exp']

        data = pd.merge(raw, raw1, on='loan_num', how='inner')

        merged_data = merged_data.append(data)

    merged_data.drop(['seller_name', 'servicer_name', 'first_pmt_date', 'mat_date', 'msa', 'net_proceeds'], axis=1,
                     inplace=True)

    # all data must have the following: credit_score, ocltv, odti, oltv, oint_rate, curr_upb
    # remove any datapoints with missing values from the above features
    merged_data.dropna(subset=['credit_score', 'odti', 'oltv', 'oint_rate', 'curr_upb'], how='any', inplace=True)
    merged_data.credit_score = pd.to_numeric(data['credit_score'], errors='coerce')
    merged_data.yearmon = pd.to_datetime(data['yearmon'], format='%Y%m')
    merged_data.fillna(value=0, inplace=True, axis=1)

    merged_data.sort_values(['loan_num'], ascending=True).groupby(['yearmon'],
                                                                  as_index=False)  ##consider move this into the next func
    merged_data.set_index(['loan_num', 'yearmon'], inplace=True)  ## consider move this into the next func

    return merged_data



def get_cv_set(data):
    train, val = train_test_split(data, test_size=0.1)
    return train, val


def process_data(data):
    # data.sort_values(['loan_num'], ascending=True).groupby(['yearmon'], as_index=False)  ##consider move this out
    # data.set_index(['loan_num', 'yearmon'], inplace=True) ## consider move this out
    y = data['curr_delinq']
    y = y.apply(lambda x: 1 if x not in (0, 1) else 0)
    # data['prev_delinq'] = data.curr_delinq.shift(1) ## needs attention here
    # data['prev_delinq'] = data.groupby(level=0)['curr_delinq'].shift(1)
    # print(sum(data.prev_delinq.isnull()))
    data.fillna(value=0, inplace=True, axis=1)
    data.drop(['curr_delinq'], axis=1, inplace=True)
    print(y.shape)
    ## how many classes are y?
    ## remove y from X

    X = pd.get_dummies(data)
    # X.net_proceeds = X.net_proceeds.apply(lambda x:0 if x == 'C' else x)
    # y = label_binarize(y, classes=[0, 1, 2, 3]) ## do we really have to do this?
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #X[['credit_score', 'mi_perc', 'units', 'ocltv', 'oupb', 'oltv', 'oint_rate', 'zip',
    #   'curr_upb', 'loan_age', 'remain_months', 'curr_rate', 'curr_def_upb', 'ddlpi', 'mi_rec',
    #   'non_mi_rec', 'exp', 'legal_costs', 'maint_exp', 'tax_insur', 'misc_exp', 'loss', 'mod_exp']] = \
    #    scale(X[['credit_score', 'mi_perc', 'units', 'ocltv', 'oupb', 'oltv', 'oint_rate', 'zip',
    #             'curr_upb', 'loan_age', 'remain_months', 'curr_rate', 'curr_def_upb', 'ddlpi', 'mi_rec',
    #             'non_mi_rec', 'exp', 'legal_costs', 'maint_exp', 'tax_insur', 'misc_exp', 'loss', 'mod_exp']],
    #          with_mean=False)
    #return X, y
    return X, y.values


def initialize_instances(data, label):
    """Read the train.csv CSV data into a list of instances."""
    instances = []

    '''
    # Read in the CSV file
    with open(infile, "r") as dat:
        next(dat)
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(float(row[-1])))
            instances.append(instance)
    '''
    for i in range(len(data)):
        instance = Instance([float(value) for value in data[i][:-1]])
        instance.setLabel(Instance(float(label[i])))
        instances.append(instance)

    return instances


def train(oa, network, oaName, instances, measure):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print("\nError results for %s\n---------------------------" % (oaName,))

    for iteration in range(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        if iteration % 100 == 0:
            print("%0.03f" % error)


def main():
    """Run algorithms on the abalone dataset."""
    all_data = get_all_data()
    train_set, val_set = get_cv_set(all_data)
    train_data, train_label = process_data(train_set)
    val_data, val_label = process_data(val_set)
    training_ints = initialize_instances(train_data, train_label)
    testing_ints = initialize_instances(val_data, val_label)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)

    networks = []  # BackPropagationNetwork
    nnop = []  # NeuralNetworkOptimizationProblem
    oa = []  # OptimizationAlgorithm
    oa_names = ["RHC", "SA", "GA"]
    results = ""

    for name in oa_names:
        classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER])
        networks.append(classification_network)
        nnop.append(NeuralNetworkOptimizationProblem(data_set, classification_network, measure))

    oa.append(RandomizedHillClimbing(nnop[0]))
    oa.append(SimulatedAnnealing(1E11, .95, nnop[1]))
    oa.append(StandardGeneticAlgorithm(200, 100, 10, nnop[2]))

    for i, name in enumerate(oa_names):
        start = time.time()
        correct = 0
        incorrect = 0

        train(oa[i], networks[i], oa_names[i], training_ints, measure)
        end = time.time()
        training_time = end - start

        optimal_instance = oa[i].getOptimal()
        networks[i].setWeights(optimal_instance.getData())

        start = time.time()
        for instance in training_ints:
            networks[i].setInputValues(instance.getData())
            networks[i].run()

            predicted = instance.getLabel().getContinuous()
            actual = networks[i].getOutputValues().get(0)

            if abs(predicted - actual) < 0.5:
                correct += 1
            else:
                incorrect += 1

        end = time.time()
        testing_time = end - start

        results += "\nResults for %s: \nCorrectly classified %d instances." % (name, correct)
        results += "\nIncorrectly classified %d instances.\nPercent correctly classified: %0.03f%%" % (incorrect, float(correct)/(correct+incorrect)*100.0)
        results += "\nTraining time: %0.03f seconds" % (training_time,)
        results += "\nTesting time: %0.03f seconds\n" % (testing_time,)

    print(results)


if __name__ == "__main__":
    main()
