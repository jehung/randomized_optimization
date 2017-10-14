import sys
import os
import time
from time import clock
from itertools import product
from array import *
import jpype as jp

sys.path.append("C:/MOOCs/CS 7641/proj2/ABAGAIL.jar")
jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=/Users/jennyhung/MathfreakData/School/OMSCS_ML/Assign2/abagail_py/ABAGAIL/ABAGAIL.jar')
jp.java.io.FileReader
jp.java.io.File
jp.java.lang.String
jp.java.lang.StringBuffer
jp.java.lang.Boolean
jp.java.util.Random
jp.java.dist.DiscreteDependencyTree
jp.java.dist.DiscreteUniformDistribution
jp.java.opt.DiscreteChangeOneNeighbor
jp.java.opt.EvaluationFunction
jp.java.opt.EvaluationFunction
jp.java.opt.HillClimbingProblem
jp.java.opt.NeighborFunction
jp.java.opt.RandomizedHillClimbing
jp.java.opt.SimulatedAnnealing
jp.java.opt.example.FourPeaksEvaluationFunction
jp.java.opt.ga.CrossoverFunction
jp.java.opt.ga.SingleCrossOver
jp.java.opt.ga.DiscreteChangeOneMutation
jp.java.opt.ga.GenericGeneticAlgorithmProblem
jp.java.opt.GenericHillClimbingProblem
jp.java.opt.ga.GeneticAlgorithmProblem
jp.java.opt.ga.MutationFunction
jp.java.opt.ga.StandardGeneticAlgorithm
jp.java.opt.ga.UniformCrossOver
jp.java.opt.prob.GenericProbabilisticOptimizationProblem
jp.java.opt.prob.MIMIC
jp.java.opt.prob.ProbabilisticOptimizationProblem
jp.java.shared.FixedIterationTrainer
jp.java.opt.example.ContinuousPeaksEvaluationFunction
jp.java.opt.example.FlipFlopEvaluationFunction
jp.java.opt.example.FlipFlopMODEvaluationFunction


FlipFlopEvaluationFunction = jp.JPackage('opt').example.FlipFlopEvaluationFunction
DiscreteUniformDistribution = jp.JPackage('dist').DiscreteUniformDistribution
DiscreteChangeOneNeighbor = jp.JPackage('opt').DiscreteChangeOneNeighbor
DiscreteChangeOneMutation = jp.JPackage('opt').ga.DiscreteChangeOneMutation
SingleCrossOver = jp.JPackage('opt').ga.SingleCrossOver
DiscreteDependencyTree = jp.JPackage('dist').DiscreteDependencyTree
GenericHillClimbingProblem = jp.JPackage('opt').GenericHillClimbingProblem
GenericGeneticAlgorithmProblem = jp.JPackage('opt').ga.GenericGeneticAlgorithmProblem
GenericProbabilisticOptimizationProblem = jp.JPackage('opt').prob.GenericProbabilisticOptimizationProblem
RandomizedHillClimbing = jp.JPackage('opt').RandomizedHillClimbing
FixedIterationTrainer = jp.JPackage('shared').FixedIterationTrainer
SimulatedAnnealing = jp.JPackage('opt').SimulatedAnnealing
StandardGeneticAlgorithm = jp.JPackage('opt').ga.StandardGeneticAlgorithm
MIMIC = jp.JPackage('opt').prob.MIMIC



"""
Commandline parameter(s):
   none
"""

N=1000
maxIters = 5001
numTrials=5
fill = [2] * N
ranges = array('i', fill)
outfile = './FLIPFLOP/FLIPFLOP_@ALG@_@N@_LOG.txt'
ef = FlipFlopEvaluationFunction()
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)


'''
# RHC
for t in range(numTrials):
    fname = outfile.replace('@ALG@','RHC').replace('@N@',str(t+1))
    with open(fname,'w') as f:
        f.write('algo,trial,iterations,param1,param2,param3,fitness,time,fevals\n')
    ef = FlipFlopEvaluationFunction()
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 10)
    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        fit.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        fevals = ef.fevals
        score = ef.value(rhc.getOptimal())
        ef.fevals -= 1
        st = '{},{},{},{},{},{},{},{},{}\n'.format('RHC', t, i, 'param1', 'param2', 'param3', score, times[-1],
                                                       fevals)
        print(st)
        with open(fname, 'a') as f:
            f.write(st)


# SA
for t in range(numTrials):
    for CE in [0.15,0.35,0.55,0.75,0.95]:
        fname = outfile.replace('@ALG@','SA{}'.format(CE)).replace('@N@',str(t+1))
        with open(fname,'w') as f:
            f.write('algo,trial,iterations,param1,param2,param3,fitness,time,fevals\n')
        ef = FlipFlopEvaluationFunction()
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(1E10, CE, hcp)
        fit = FixedIterationTrainer(sa, 10)
        times =[0]
        for i in range(0,maxIters,10):
            start = clock()
            fit.train()
            elapsed = time.clock()-start
            times.append(times[-1]+elapsed)
            fevals = ef.fevals
            score = ef.value(sa.getOptimal())
            ef.fevals -= 1
            st = '{},{},{},{},{},{},{},{},{}\n'.format('SA',t,i,CE,'param2','param3',score,times[-1],fevals)
            print(st)
            with open(fname,'a') as f:
                f.write(st)


#GA
for t in range(numTrials):
    for pop,mate,mutate in product([100],[50,30,10],[50,30,10]):
        fname = outfile.replace('@ALG@','GA{}_{}_{}'.format(pop,mate,mutate)).replace('@N@',str(t+1))
        with open(fname,'w') as f:
            f.write('algo,trial,iterations,param1,param2,param3,fitness,time,fevals\n')
        ef = FlipFlopEvaluationFunction()
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
        fit = FixedIterationTrainer(ga, 10)
        times =[0]
        for i in range(0,maxIters,10):
            start = clock()
            fit.train()
            elapsed = time.clock()-start
            times.append(times[-1]+elapsed)
            fevals = ef.fevals
            score = ef.value(ga.getOptimal())
            ef.fevals -= 1
            st = '{},{},{},{},{},{},{},{},{}\n'.format('GA',t,i,pop,mate,mutate,score,times[-1],fevals)
            print(st)
            with open(fname,'a') as f:
                f.write(st)

'''
#MIMIC
for t in range(numTrials):
    for samples,keep,m in product([100],[50],[0.5,0.7,0.9]):
        fname = outfile.replace('@ALG@','MIMIC{}_{}_{}'.format(samples,keep,m)).replace('@N@',str(t+1))
        with open(fname,'w') as f:
            f.write('algo,trial,iterations,param1,param2,param3,fitness,time,fevals\n')
        ef = FlipFlopEvaluationFunction()
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        df = DiscreteDependencyTree(m, ranges)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, keep, pop)
        fit = FixedIterationTrainer(mimic, 10)
        times =[0]
        for i in range(0,maxIters,10):
            start = clock()
            fit.train()
            elapsed = time.clock()-start
            times.append(times[-1]+elapsed)
            fevals = ef.fevals
            score = ef.value(mimic.getOptimal())
            ef.fevals -= 1
            st = '{},{},{},{},{},{},{},{},{}\n'.format('MIMIC',t,i,samples,keep,m,score,times[-1],fevals)
            print(st)
            with open(fname,'a') as f:
                f.write(st)
