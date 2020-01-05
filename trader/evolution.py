import concurrent.futures
from random import random

import numpy as np
from trader.simulation import TraderSimulation

from trader.trader import TraderConfiguration


class Specimen:

    def __init__(self, config, fitness):
        self.config = config
        self.fitness = fitness


class Evolution:

    def __init__(self, exchange_data, predictions, init_balance, fee, uncertainty, generation_size, num_generations,
                 mutation_rate):
        self.exchange_data = exchange_data
        self.predictions = predictions
        self.init_balance = init_balance
        self.fee = fee
        self.uncertainty = uncertainty
        self.generation_size = generation_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

    def run(self):
        specimens = []
        best_candidates = None

        for i in range(self.generation_size):
            specimens.append(Specimen(self.random_config(-1, 1), 0))

        for gen in range(self.num_generations):
            tested_specimens = self.simulate_generation(specimens)
            candidates = self.select_candidates(tested_specimens, 2)
            if best_candidates is None or candidates[0].fitness > best_candidates[0].fitness:
                best_candidates = candidates

            print('Generation %d Fitness: %f' % (gen, best_candidates[0].fitness))
            specimens = self.breed(best_candidates)

        return best_candidates[0]

    def simulate_generation(self, specimens):
        tested_specimens = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specimens)) as executor:
            workers = []
            for spec in specimens:
                workers.append(executor.submit(self.simulation_worker, spec))

            for w in concurrent.futures.as_completed(workers):
                if w.result() is not None:
                    tested_specimens.append(w.result())

        return tested_specimens

    def simulation_worker(self, specimen):
        simulation = TraderSimulation(self.exchange_data, self.predictions, specimen.config, self.init_balance,
                                      self.fee, self.uncertainty)
        specimen.fitness = simulation.run()
        return specimen

    def select_candidates(self, specimen_list, num_candidates):
        return sorted(specimen_list, key=lambda x: x.fitness, reverse=True)[:num_candidates]

    def breed(self, candidates):
        new_generation = []
        c1 = candidates[0].config
        c2 = candidates[1].config

        for i in range(0, self.generation_size):
            buy_thr = self.random_float(c1.buy_thr, c2.buy_thr)
            increase_thr = self.random_float(c1.increase_thr, c2.increase_thr)
            sell_thr = self.random_float(c1.sell_thr, c2.sell_thr)
            min_profit = self.random_float(c1.min_profit, c2.min_profit)
            max_loss = self.random_float(c1.max_loss, c2.max_loss)
            position_sizing = self.random_float(c1.position_sizing, c2.position_sizing)
            new_conf = TraderConfiguration(buy_thr, increase_thr, sell_thr, min_profit, max_loss, position_sizing)

            if random.uniform(0, 1) <= self.mutation_rate:
                new_conf = self.mutate(new_conf, -1, 1)

            new_generation.append(new_conf)

    def mutate(self, config, min_value, max_value):
        param_idx = random.randint(0, 5)
        value = random.uniform(min_value, max_value)

        if param_idx == 0:
            config.buy_thr = value
        elif param_idx == 1:
            config.increase_thr = value
        elif param_idx == 2:
            config.sell_thr = value
        elif param_idx == 3:
            config.min_profit = value
        elif param_idx == 4:
            config.max_loss = value
        elif param_idx == 5:
            config.position_sizing = value

        return config

    def random_config(self, min_value, max_value):
        params = np.random.uniform(low=min_value, high=max_value, size=(6,))
        return TraderConfiguration(params[0], params[1], params[2], params[3], params[4], params[5])

    def random_float(self, a, b):
        if a > b:
            return random.uniform(b, a)
        else:
            return random.uniform(a, b)
