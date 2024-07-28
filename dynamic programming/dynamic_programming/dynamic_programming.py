import copy


class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in
                   range(self.env.ncol * self.env.nrow)]
        self.theta = theta
        self.gamma = gamma

    def policy_evaluation(self):
        count = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.trans_matrix[s][a]:
                        prob, neat_state, r, done = res
                        qsa += prob * (r + self.gamma * self.v[neat_state] * (
                                1 - done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            count += 1
        print(f'epoch: {count}, policy evaluation finish!')

    def policy_improvement(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.trans_matrix[s][a]:
                    prob, neat_state, r, done = res
                    qsa += prob * (r + self.gamma * self.v[neat_state] * (
                            1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            num_maxq = qsa_list.count(maxq)
            self.pi[s] = [1 / num_maxq if q == maxq else 0 for q in qsa_list]
        print('policy improvement finish!')
        return self.pi

    def policy_iteration(self):
        conut = 0
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            conut += 1
            print(f'policy iteration running, epoch: {conut}')
            if old_pi == new_pi:
                break


class DpSolver:
    def __init__(self, agent, action_meaning, disaster, end):
        self.agent = agent
        self.action_meaning = action_meaning
        self.disaster = disaster
        self.end = end

    def print_agent(self):
        print('state value:')
        for i in range(self.agent.env.nrow):
            for j in range(self.agent.env.ncol):
                print(
                    '%6.6s' % ('%.3f' % self.agent.v[
                        i * self.agent.env.ncol + j]), end=' ')
            print('\n')
        print('policy:')
        for i in range(self.agent.env.nrow):
            for j in range(self.agent.env.ncol):
                if (i * self.agent.env.ncol + j) in self.disaster:
                    print('****', end=' ')
                elif (i * self.agent.env.ncol + j) in self.end:
                    print('EEEE', end=' ')
                else:
                    a = self.agent.pi[i * self.agent.env.ncol + j]
                    pi_str = ''
                    for k in range(len(self.action_meaning)):
                        pi_str += self.action_meaning[k] if a[k] > 0 else 'o'
                    print(pi_str, end=' ')
            print('\n')

    def run(self):
        self.agent.policy_iteration()
        self.print_agent()
