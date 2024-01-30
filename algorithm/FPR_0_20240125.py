import config


class DF_FPR:
    def __init__(self, monitored_groups, alpha, threshold):
        self.name = "FPR"
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
        print("FPR, groups: ", self.groups)
        print("uf", self.uf)
        print("delta", self.delta)


    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    """
    here we assume that the tuple only satisfies one group
    label = "FP" or "TN"
    """
    def insert(self, tuple_, label):
        for group_idx, group in enumerate(self.groups):
            if self.belong_to_group(tuple_, group): # for group that the tuple satisfies
                if label == "TN":
                    if not self.uf[group_idx]:
                        self.delta[group_idx] += self.threshold
                    else:
                        if self.delta[group_idx] >= self.threshold:
                            self.delta[group_idx] -= self.threshold
                        else:
                            self.delta[group_idx] = self.threshold - self.delta[group_idx]
                            self.uf[group_idx] = False
                else:
                    if not self.uf[group_idx]:
                        if self.delta[group_idx] > 1 - self.threshold:
                            self.delta[group_idx] -= 1 - self.threshold
                        else:  # original unfair
                            self.delta[group_idx] = 1 - self.threshold - self.delta[group_idx]
                            self.uf[group_idx] = True
                    else:
                        self.delta[group_idx] += 1 - self.threshold
    def new_window(self):
        self.delta = [round(x * self.alpha, config.decimal) for x in self.delta]



