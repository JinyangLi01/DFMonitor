
class Accuracy_baseline:
    def __init__(self, monitored_groups, alpha, threshold):
        self.name = "Accuracy_baseline"
        self.groups = monitored_groups
        self.uf = [False] * len(monitored_groups)
        self.counters_correct = [0] * len(monitored_groups)
        self.counters_incorrect = [0] * len(monitored_groups)
        self.alpha = alpha
        self.threshold = threshold

    def find(self, group):
        # idx = self.group2idx[str(group)]
        idx = self.groups.index(group)
        return self.uf[idx]

    def print(self):
        print("uf", self.uf)
        print("counters_correct", self.counters_correct)
        print("counters_incorrect", self.counters_incorrect)

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
            if self.belong_to_group(tuple_, group):  # only for group that the tuple satisfies
                if label == 'correct':
                    self.counters_correct[group_idx] += 1
                else:
                    self.counters_incorrect[group_idx] += 1
                if (self.counters_correct[group_idx] /
                        (self.counters_correct[group_idx] + self.counters_incorrect[group_idx])
                        <= self.threshold):
                    self.uf[group_idx] = True
                else:
                    self.uf[group_idx] = False

    def new_window(self):
        self.counters_correct = [round(x * self.alpha, config.decimal) for x in self.counters_correct]
        self.counters_incorrect = [round(x * self.alpha, config.decimal) for x in self.counters_incorrect]
