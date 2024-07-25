import copy


class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.trans_matrix = self.create_p()

    def create_p(self):
        p = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.nrow):
                for a in range(4):
                    if i == self.nrow - 1 and j > 0:
                        p[i * self.ncol + j][a] = [
                            (1, i * self.ncol + j, 0, True)]
                        continue

                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:
                            reward = -100
                    p[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return p
