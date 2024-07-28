import environment as env
import dynamic_programming as dp

if __name__ == '__main__':
    env = env.CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = dp.PolicyIteration(env=env, theta=theta, gamma=gamma)
    policy_runner = dp.DpSolver(agent=agent,
                                action_meaning=action_meaning,
                                disaster=list(range(37, 47)),
                                end=[47])
    policy_runner.run()
