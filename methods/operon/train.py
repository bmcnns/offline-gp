import pyoperon as Operon
from dataset import load_dataset
import numpy as np
import random
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import pickle

def train(dataset_name, num_independent_runs, save_dir, hyperparameters, starting_generation=0):
    X, y = load_dataset(dataset_name)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Starting at {starting_generation}")
    for run in range(starting_generation, num_independent_runs):
        models = []
        for action in range(y.shape[1]):
            A = np.column_stack((X, y[:, action]))
            
            ds = Operon.Dataset(np.asfortranarray(A))
            
            training_range = Operon.Range(0, ds.Rows - 1)
            test_range = Operon.Range(ds.Rows - 1, ds.Rows)
            
            target = ds.Variables[-1]
            inputs = [ h for h in ds.VariableHashes if h != target.Hash ]
            
            # Initialize a RNG
            rng = Operon.RandomGenerator(random.randint(1, 1_000_000))
            
            problem = Operon.Problem(ds)
            problem.TrainingRange = training_range
            problem.TestRange = test_range
            problem.Target = target
            problem.InputHashes = inputs
            
            config = Operon.GeneticAlgorithmConfig(
                generations=1, max_evaluations=1_000_000_000_000_000, local_iterations=5,
                population_size=1000, pool_size=1000, p_crossover=1.0,
                p_mutation=0.25, epsilon=1e-5, seed=1, max_time=86400)
            
            selector = Operon.TournamentSelector(objective_index=0)
            selector.TournamentSize = 5
            
            problem.ConfigurePrimitiveSet(Operon.NodeType.Constant |
                                          Operon.NodeType.Variable |
                                          Operon.NodeType.Add |
                                          Operon.NodeType.Mul |
                                          Operon.NodeType.Div |
                                          Operon.NodeType.Exp |
                                          Operon.NodeType.Log |
                                          Operon.NodeType.Sin |
                                          Operon.NodeType.Cos)
            
            pset = problem.PrimitiveSet
            
            minL, maxL = 1, 50
            maxD = 10
            
            btc = Operon.BalancedTreeCreator(pset, problem.InputHashes, bias=0.0)
            tree_initializer = Operon.UniformLengthTreeInitializer(btc)
            tree_initializer.ParameterizeDistribution(minL, maxL)
            tree_initializer.MaxDepth = maxD
            
            coeff_initializer = Operon.NormalCoefficientInitializer()
            coeff_initializer.ParameterizeDistribution(0,1)
            
            mut_onepoint = Operon.NormalOnePointMutation()
            mut_changeVar = Operon.ChangeVariableMutation(inputs)
            mut_changeFunc = Operon.ChangeFunctionMutation(pset)
            mut_replace = Operon.ReplaceSubtreeMutation(btc, coeff_initializer, maxD, maxL)
            
            mutation = Operon.MultiMutation()
            mutation.Add(mut_onepoint, 1)
            mutation.Add(mut_changeVar, 1)
            mutation.Add(mut_changeFunc, 1)
            mutation.Add(mut_replace, 1)
            
            # define crossover
            crossover_internal_probability = 0.9 # probability to pick an internal node as a cut point
            crossover = Operon.SubtreeCrossover(crossover_internal_probability, maxD, maxL)
            
            dtable         = Operon.DispatchTable()
            error_metric   = Operon.MSE()          # use the mean squared error as fitness
            evaluator      = Operon.Evaluator(problem, dtable, error_metric, True) # initialize evaluator, use linear scaling = True
            evaluator.Budget = config.Evaluations # computational budget
            
            optimizer      = Operon.LMOptimizer(dtable, problem, max_iter=config.Iterations)
            local_search   = Operon.CoefficientOptimizer(optimizer)
            
            # define how new offspring are created
            generator = Operon.BasicOffspringGenerator(evaluator, crossover, mutation, selector, selector, local_search)
            
            reinserter = Operon.ReplaceWorstReinserter(objective_index=0)
            
            def report():
                with open(f'{save_dir}/results.csv', 'a') as f:
                    print(f'Dataset: {dataset_name},Run: {run},Generation: {generation},Action: {action},Best: {gp.BestModel.GetFitness(0)}\n')
                    f.write(f'{dataset_name},{run},{generation},{action},{gp.BestModel.GetFitness(0)}\n')
                pass
            
            # run the algorithm
            num_generations = 1000

            for generation in range(num_generations):
                start = np.random.randint(0, 999) * 1000
                end = start + 1000
                training_range = Operon.Range(start, end)
                problem.TrainingRange = training_range

                gp = Operon.GeneticProgrammingAlgorithm(config, problem, tree_initializer, coeff_initializer, generator, reinserter)

                if (generation > 0):
                    gp.IsFitted = True
                    gp.RestoreIndividuals(individuals)
                gp.Run(rng, report, threads=20, warm_start=True)
                individuals = gp.Individuals
            
            best = gp.BestModel
            model_string = Operon.InfixFormatter.Format(best.Genotype, ds, 6)
            fit = evaluator(rng, gp.BestModel)
            print('gen=', generation, '\nfit=', fit)
            print(f'\n{model_string}')

            models.append(model_string)

        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/model_{run+1}.pkl', 'wb') as f:
            pickle.dump((models, scaler), f)
