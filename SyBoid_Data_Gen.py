import numpy as np
from math import factorial
import random
import pymop.factory
from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools
from sklearn.metrics.pairwise import nan_euclidean_distances
from SyntheticDataGenerator import SyntheticDataGenerator
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skewnorm
from Boid import Boid
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
np.seterr(divide='ignore', invalid='ignore')

class SyBoid():

    def Generate_Data(X, y, NumberOfBoids, Dimensions, Classes, F1_Score, 
                                    N1_Score, Mimic_Classes, Mimic_DataTypes, 
                                    Mimic_Dataset):
    
        def main(seed=None):
            random.seed(seed)

            # Initialize statistics object
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)

            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "std", "min", "avg", "max"

            pop = toolbox.population(n=MU)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Compile statistics about the population
            record = stats.compile(pop)
            logbook.record(gen=0, evals=len(invalid_ind), **record)

            hist_best = 1
            hist_best_gen = 0
            # Begin the generational process
            for gen in range(1, NGEN):
                offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                fits = [np.array(ind.fitness.values) for ind in pop if ind.fitness.valid]
                best_fit = np.nanmin(nan_euclidean_distances(np.array([0,0]).reshape(1,-1),np.vstack(fits)))
                avg_fit = np.nanmean(nan_euclidean_distances(np.array([0,0]).reshape(1,-1),np.vstack(fits)))
                worst_fit = np.nanmax(nan_euclidean_distances(np.array([0,0]).reshape(1,-1),np.vstack(fits)))

                print("Gen: %.0f, Best: %.3f, Avg: %.3f, Worst: %.3f" % (gen,best_fit,avg_fit,worst_fit))
                if best_fit < hist_best:
                    hist_best = best_fit
                    hist_best_gen = gen
                elif (gen - hist_best_gen) > 4:
                    return pop, fits

                if best_fit < 0.015:
                    return pop, fits
                # Select the next generation population from parents and offspring
                pop = toolbox.select(pop + offspring, MU)

                # Compile statistics about the new population
                record = stats.compile(pop)
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
            return pop, fits

        # Problem definition
        PROBLEM = "Sy:Boid"
        NOBJ = 2
        K = 8
        NDIM = 8
        P = 12
        H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
        BOUND_LOW, BOUND_UP = 0.0, 1
        problem = SyntheticDataGenerator(X, y, NumberOfBoids, Dimensions, Classes, F1_Score, 
                                        N1_Score,Mimic_Classes, Mimic_DataTypes, 
                                        Mimic_Dataset)
        ##

        # Algorithm parameters
        MU = 40
        NGEN = 10
        CXPB = 0.95
        MUTPB = 0.01
        ##

        # Create uniform reference point
        ref_points = tools.uniform_reference_points(NOBJ, P)

        # Create classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Toolbox initialization
        def uniform(low, up, size=None):
            try:
                return [random.uniform(a, b) for a, b in zip(low, up)]
            except TypeError:
                return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

        toolbox = base.Toolbox()
        toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded,low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
        ##
        pop, stats = main()
        return pop, stats

    def return_best_dataset(X,y,best_x, Mimic_Classes, Mimic_DataTypes, Mimic_Dataset):
        x = np.copy(best_x)
        NumberOfBoids = X.shape[0]
        Dimensions = X.shape[1]
        Classes = np.unique(y).shape[0]
        ll=np.array([0,0,0,0,0,0,-100,-100])
        ul=np.array([1,1,1,0.50,1,0.50,100,100])

        x[3] = ll[3] + (ul[3] * x[3])
        x[5] = ll[5] + (ul[5] * x[5])
        x[6] = ll[6] + (ul[6] * x[6])
        x[7] = ll[7] + (ul[7] * x[7])

        # Simulation Parameters
        timesteps = 15

        #initialise X and y
        if Mimic_Dataset:
            positions = np.zeros((NumberOfBoids,Dimensions,timesteps))
            positions[:,:,0] = X
            target = y
        elif Mimic_Classes:
            np.random.seed(0)
            positions = np.zeros((NumberOfBoids,Dimensions,timesteps))
            positions[:,:,0] = np.random.uniform(0,1,(NumberOfBoids,Dimensions))
            target = y
        else:
            np.random.seed(0)
            positions = np.zeros((NumberOfBoids,Dimensions,timesteps))
            positions[:,:,0] = np.random.uniform(0,1,(NumberOfBoids,Dimensions))
            target = np.random.randint(0,Classes,NumberOfBoids)

        if Mimic_DataTypes:
            correct_discrete = np.sum(X,axis=0) % 1 == 0

        #initialise obedience values
        ObedienceValues = skewnorm.rvs(x[6], size=NumberOfBoids, random_state=42)
        ObedienceValues = 1 - ((ObedienceValues-np.min(ObedienceValues))/((np.max(ObedienceValues))-np.min(ObedienceValues)))

        #Inititalise Feature Importance
        FeatureImportance = skewnorm.rvs(x[7], loc=1, scale=0.5, size=Dimensions, random_state=42)
        FeatureImportance[FeatureImportance<0] = 0
        FeatureImportance[FeatureImportance>1] = 1

        # simulate boids
        for i in range(1,timesteps):
            positions[:,:,i] = Boid.main(positions[:,:,i-1], target, x[0],x[1],x[2],x[3],x[4],x[5],FeatureImportance,ObedienceValues)
            

        if Mimic_DataTypes:
            for i in range(correct_discrete.shape[0]):
                if correct_discrete[i]:
                    X_fake = MinMaxScaler().fit_transform(positions[:,i,-1].reshape(-1, 1))
                    ul = max(X[:,i])
                    ll = min(X[:,i])
                    scaled = ll + (ul * X_fake).reshape(1,-1)
                    positions[:,i,-1] = np.round(scaled,0)


        return positions[:,:,-1], target