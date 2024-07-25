import bernoulli_bandit as b

random_seed = 0

k = 10
steps = 5000
solvers = []
solver_name = []
ucb_coef = 1

if __name__ == '__main__':
    bandit_10_arm = b.BernoulliBandit(k)
    print(f'{k}-armed bandit: \n'
          f'  best scoring arm: {bandit_10_arm.best_idx} \n'
          f'  max scoring prob: {bandit_10_arm.best_prob} \n')

    metrics = b.ResultsMetrics(solvers=solvers, solver_name=solver_name)

    epsilon_greedy_solver = b.EpsilonGreedy(bandit=bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(num_steps=steps)
    solvers.append(epsilon_greedy_solver)
    solver_name.append('EpsilonGreedy')
    print(f'the cumulative regrets value of epsilon-greedy is: '
          f'{epsilon_greedy_solver.regret} \n')
    metrics.plot_results()
    solvers.clear()
    solver_name.clear()

    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    for e in epsilons:
        solvers.append(b.EpsilonGreedy(bandit=bandit_10_arm, epsilon=e))
        solver_name.append(f'epsilon={e}')
    for solver in solvers:
        solver.run(num_steps=steps)
        print(f'when epsilon={solver.epsilon} '
              f'the cumulative regrets value of epsilon-greedy is: '
              f'{solver.regret}')
    print('\n')
    metrics.plot_results()
    solvers.clear()
    solver_name.clear()

    decaying_epsilon_greedy_solver = (
        b.DecayingEpsilonGreedy(bandit=bandit_10_arm))
    decaying_epsilon_greedy_solver.run(num_steps=steps)
    solvers.append(decaying_epsilon_greedy_solver)
    solver_name.append('DecayingEpsilonGreedy')
    print(f'the cumulative regrets value of decaying-epsilon-greedy is: '
          f'{decaying_epsilon_greedy_solver.regret} \n')
    metrics.plot_results()
    solvers.clear()
    solver_name.clear()

    UCB_solver = b.UCB(bandit=bandit_10_arm, coef=ucb_coef)
    UCB_solver.run(num_steps=steps)
    solvers.append(UCB_solver)
    solver_name.append('UCB')
    print(f'the cumulative regrets value of UCB is: '
          f'{UCB_solver.regret} \n')
    metrics.plot_results()
    solvers.clear()
    solver_name.clear()

    thompson_sampling_solver = b.ThompsonSampling(bandit=bandit_10_arm)
    thompson_sampling_solver.run(num_steps=steps)
    solvers.append(thompson_sampling_solver)
    solver_name.append('ThompsonSampling')
    print(f'the cumulative regrets value of ThompsonSampling is: '
          f'{thompson_sampling_solver.regret} \n')
    metrics.plot_results()
    solvers.clear()
    solver_name.clear()

    co_solvers = [epsilon_greedy_solver, decaying_epsilon_greedy_solver,
                  UCB_solver, thompson_sampling_solver]
    co_names = ['EpsilonGreedy', 'DecayingEpsilonGreedy',
                'UCB', 'ThompsonSampling']

    for solver, name in zip(co_solvers, co_names):
        solvers.append(solver)
        solver_name.append(name)
    metrics.plot_results()
