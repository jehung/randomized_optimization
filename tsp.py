import sys
import os
import time

import sys
import os
import time
from time import clock
from itertools import product
from array import *
import random
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
jp.java.dist.DiscretePermutationDistribution
jp.java.opt.example.TravelingSalesmanEvaluationFunction
jp.java.opt.example.TravelingSalesmanRouteEvaluationFunction
jp.java.opt.SwapNeighbor
jp.java.opt.ga.SwapMutation
jp.java.opt.example.TravelingSalesmanCrossOver
jp.java.opt.example.TravelingSalesmanSortEvaluationFunction
jp.java.shared.Instance
jp.java.util.ABAGAILArrays



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
Random = jp.java.util.Random
TravelingSalesmanRouteEvaluationFunction = jp.JPackage('opt').example.TravelingSalesmanRouteEvaluationFunction
DiscretePermutationDistribution = jp.JPackage('dist').DiscretePermutationDistribution
SwapNeighbor = jp.JPackage('opt').SwapNeighbor
SwapMutation = jp.JPackage('opt').ga.SwapMutation
TravelingSalesmanCrossOver = jp.JPackage('opt').example.TravelingSalesmanCrossOver
TravelingSalesmanSortEvaluationFunction = jp.JPackage('opt').example.TravelingSalesmanSortEvaluationFunction



"""
Commandline parameter(s):
   none
"""

# set N value.  This is the number of points
N = 100
random = Random()
maxIters = 3001
numTrials=5
fill = [N] * N
ranges = array('i', fill)


points = [[0 for x in range(2)] for x in range(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()
outfile = './TSP/TSP_@ALG@_@N@_LOG.txt'
ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)



# RHC
for t in range(numTrials):
    fname = outfile.replace('@ALG@','RHC').replace('@N@',str(t+1))
    with open(fname,'w') as f:
        f.write('algo,trial,iterations,param1,param2,param3,fitness,time,fevals\n')
    ef = TravelingSalesmanRouteEvaluationFunction(points)
    odd = DiscreteUniformDistribution(ranges)
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
        st = '{},{},{},{},{},{},{},{},{}\n'.format('RHC', t, i, 'param1', 'param2', 'param3', score, times[-1], fevals)
        print(st)
        with open(fname, 'a') as f:
            f.write(st)



# SA
for t in range(numTrials):
    for CE in [0.15,0.35,0.55,0.75,0.95]:
        fname = outfile.replace('@ALG@','SA{}'.format(CE)).replace('@N@',str(t+1))
        with open(fname,'w') as f:
            f.write('algo,trial,iterations,param1,param2,param3,fitness,time,fevals\n')
        ef = TravelingSalesmanRouteEvaluationFunction(points)
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
        ef = TravelingSalesmanRouteEvaluationFunction(points)
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


#MIMIC
for t in range(numTrials):
    for samples,keep,m in product([100],[50],[0.1,0.3,0.5,0.7,0.9]):
        fname = outfile.replace('@ALG@','MIMIC{}_{}_{}'.format(samples,keep,m)).replace('@N@',str(t+1))
        with open(fname,'w') as f:
            f.write('algo,trial,iterations,param1,param2,param3,fitness,time,fevals\n')
        ef = TravelingSalesmanSortEvaluationFunction(points)
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


