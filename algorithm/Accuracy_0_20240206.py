


class DF_Accuracy:
    def __init__(self, monitored_groups, alpha, threshold):
        self.name = "Accuracy"
        self.groups = monitored_groups
        # self.group2idx = {str(group): i for i, group in enumerate(monitored_groups)}
        self.uf = [False]*len(monitored_groups)
        self.delta = [0]*len(monitored_groups)
        self.alpha = alpha
        self.threshold = threshold

    def initialization(self, uf, delta):
        self.uf = uf
        self.delta = delta

    def find(self, group):
        # idx = self.group2idx[str(group)]
        idx = self.groups.index(group)
        return self.uf[idx]

    def print(self):
        print("uf", self.uf)
        print("delta", self.delta)


    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    """
    label = correct or incorrect
    """
    def insert(self, tuple_, label):
        for group_idx, group in enumerate(self.groups):
            if self.belong_to_group(tuple_, group): # only for group that the tuple satisfies
                if not self.uf[group_idx]:
                    if label == 'correct':
                        self.delta[group_idx] += 1 - self.threshold
                    else:
                        if self.delta[group_idx] >= self.threshold:
                            self.delta[group_idx] -= self.threshold
                        else:
                            self.delta[group_idx] = self.threshold - self.delta[group_idx]
                            self.uf[group_idx] = True
                else:
                    if label == 'correct':
                        if self.delta[group_idx] >= 1 - self.threshold:
                            self.delta[group_idx] -= 1 - self.threshold
                        else:
                            self.delta[group_idx] = 1 - self.threshold - self.delta[group_idx]
                            self.uf[group_idx] = False
                    else:
                        self.delta[group_idx] += self.threshold



    def new_window(self):
        self.delta = [x * self.alpha for x in self.delta]


