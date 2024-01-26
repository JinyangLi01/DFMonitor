
class FPR_baseline:
    def __init__(self, monitored_groups, alpha, threshold):
        self.name = "CR_baseline"
        self.groups = monitored_groups
        self.uf = [False]*len(monitored_groups)
        self.counters_TN = [0]*len(monitored_groups)
        self.counters_FP = [0]*len(monitored_groups)
        self.threshold = threshold
        self.alpha = alpha

    def find(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]

    def print(self):
        print("uf", self.uf)
        print("counters_TN", self.counters_TN)
        print("counters_FP", self.counters_FP)

    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    """
    label = "FP" or "TN"
    """
    def insert(self, tuple_, label):
        for i, group in enumerate(self.groups):
            if self.belong_to_group(tuple_, group):  # for group that the tuple satisfies
                if label == "TN":
                    self.counters_TN[i] += 1
                else:
                    self.counters_FP[i] += 1
                if self.counters_FP[i] / (self.counters_TN[i] + self.counters_FP[i]) <= self.threshold:
                    self.uf[i] = True


    def new_window(self):
        self.counters_FP = [x * self.alpha for x in self.counters_FP]
        self.counters_TN = [x * self.alpha for x in self.counters_TN]


