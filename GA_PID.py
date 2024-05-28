import numpy as np
import random
from tqdm import tqdm
from tinyphysics_modified import TinyPhysicsSimulator, TinyPhysicsModel
from pathlib import Path
import concurrent.futures
import argparse
import logging

class BaseController:
    def update(self, target_lataccel, current_lataccel, state):
        raise NotImplementedError

class PIDController(BaseController):
  def __init__(self, kp=0.08852291, ki=0.07724999, kd=-0.05190457):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.last_err = 0
    self.int = 0
    
  def update(self, target_lataccel, current_lataccel, state):
    curr_err = target_lataccel - current_lataccel
    deriv = curr_err - self.prev_err
    self.int += curr_err
    self.last_err = curr_err
    
    output = self.kp * curr_err + self.ki * self.int + self.kd * deriv
    
    return output

BOUNDS = {'kp': (0, .2), 'ki': (-.1, .1), 'kd': (-.1, .1)}

def init_ga(num_segs, population_size, debug):
    
    data = Path("./data/")
    files = sorted(data.iterdir())[:num_segs]
    
    population = np.random.rand(population_size, 3)
    for i, key in enumerate(BOUNDS.keys()):
        population[:, i] = population[:, i] * (BOUNDS[key][1] - BOUNDS[key][0]) + BOUNDS[key][0]

    return files, population

def test_pid_wrapper(params):
    kp, ki, kd, selected_files, debug = params
    env_model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=debug)
    return test_pid(kp, ki, kd, env_model, selected_files, debug)

def run_generation(files, population, num_segs, num_trial_segs, gen_num, cross_rate, mut_rate, debug):
    logging.info(f"Generation {gen_num}-------------")

    number_range = list(range(num_segs))
    selected_file_indices = random.sample(number_range, num_trial_segs)
    
    selected_files = [files[i] for i in selected_file_indices]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        runs = [(ind[0], ind[1], ind[2], selected_files, debug) for ind in population]
        fitness_results = list(tqdm(executor.map(test_pid_wrapper, runs), total=len(runs), desc="Evaluating PID Parameters"))

    fitness_results = np.array(fitness_results)
    parents_indices = np.argsort(fitness_results)[:len(population) // 2]
    parents = population[parents_indices]

    children = []
    for _ in range(len(population) - len(parents)):
        if np.random.rand() < cross_rate:
            p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
            crossover_point = np.random.randint(1,3)
            child = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
            children.append(child)
    
    if len(children) + len(parents) < 15:
        new_kids = np.random.rand(15 - len(parents) - len(children), 3)
        for i, key in enumerate(BOUNDS.keys()):
            new_kids[:, i] = new_kids[:, i] * (BOUNDS[key][1] - BOUNDS[key][0]) + BOUNDS[key][0]
    
        children = np.vstack([children, new_kids])

    for child in children:
        for i in range(3):
            if np.random.rand() < mut_rate:
                child[i] = np.random.rand() * (BOUNDS[list(BOUNDS.keys())[i]][1] - BOUNDS[list(BOUNDS.keys())[i]][0]) + BOUNDS[list(BOUNDS.keys())[i]][0]
        
    best_in = np.argmin(fitness_results)
    best_pid_cost = fitness_results[best_in]

    best_pid = population[best_in]
    
    logging.info(f"Best parama so far: {best_pid}, Cost: {best_pid_cost}")
    
    if len(children) > 0:
        population = np.vstack([parents, children])
    else:
        population = np.copy(parents)

    return population, best_in, best_pid, best_pid_cost

def run_rollout(kp, ki, kd, model, data_path, debug):
    pid_controller = PIDController(kp, ki, kd)
    simulator = TinyPhysicsSimulator(model, f"./{data_path}", controller=pid_controller, debug=debug)
    costs = simulator.rollout()
    return costs['total_cost']

def test_pid(kp, ki, kd, model, data_files, debug=False):
    total_cost = 0

    for data_file in data_files:
        total_cost += run_rollout(kp, ki, kd, model, data_file, debug)

    average_cost = total_cost / len(data_files)
    return average_cost

def run_ga(num_segs, pop_size=25, num_gens=50, mut_rate=0.1, cross_rate=0.75, num_trial_segs=20, debug=False):
    
    files, population = init_ga(num_segs, pop_size, debug)

    logging.info(f"Initial population: {population}")
    
    for generation in range(num_gens):
        population, best_index, best_pid, best_pid_cost = run_generation(files, population, num_segs, num_trial_segs, generation, cross_rate, mut_rate, debug)
    
    return best_pid 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_segs", type=int, default=500)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    best_pid = run_ga(args.num_segs)

    print("Best PID Params:", best_pid)
