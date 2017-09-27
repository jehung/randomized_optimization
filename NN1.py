"""
RHC NN training on my classification problem

"""

"""
Do this before running the code in terminal / command line:
git clone https://github.com/originell/jpype.git
cd jpype
python setup.py install'

Additional reference doc:
https://stackoverflow.com/questions/35736763/practical-use-of-java-class-jar-in-python
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
Instance = jp.JPackage('shared').Instance
RELU = jp.JPackage('func').nn.activation.RELU


INPUT_LAYER = 109
HIDDEN_LAYER1 = 100
HIDDEN_LAYER2 = 100
HIDDEN_LAYER3 = 100
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5001
OUTFILE = 'LOG.txt'


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


def train(oa, network, oaName, training_ints,testing_ints, measure):
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
            MSE_tst, acc_tst = errorOnDataSet(network,testing_ints,measure)
            txt = '{},{},{},{},{},{}\n'.format(iteration,MSE_trg,MSE_tst,acc_trg,acc_tst,times[-1]);print(txt)
            with open(OUTFILE,'a+') as f:
                f.write(txt)

def main():
    """Run this experiment"""
    all_data = get_all_data()
    train_set, val_set = get_cv_set(all_data)
    train_data, train_label = process_data(train_set)
    val_data, val_label = process_data(val_set)
    training_ints = initialize_instances(train_data, train_label)
    testing_ints = initialize_instances(val_data, val_label)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    #rule = RPROPUpdateRule()
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1,HIDDEN_LAYER2,HIDDEN_LAYER3, OUTPUT_LAYER],relu)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = RandomizedHillClimbing(nnop)
    train(oa, classification_network, 'RHC', training_ints,testing_ints, measure)



if __name__ == "__main__":
    with open(OUTFILE,'w') as f:
        f.write('{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_tst','acc_trg','acc_tst','elapsed'))
    main()
    jp.shutdownJVM()
