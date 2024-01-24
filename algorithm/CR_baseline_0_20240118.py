
class CR_baseline:
    def __init__(self, monitored_groups, alpha, threshold):
        self.name = "CR_baseline"
        self.groups = monitored_groups
        self.uf = [False]*len(monitored_groups)
        self.counters = [0]*len(monitored_groups)
        self.counter_total = 0
        self.threshold = threshold
        self.alpha = alpha

    # def initialization(self, n):
    #     self.uf = [i for i in range(n)]
    #     self.counters = [0 for i in range(n)]

    def find(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]

    def print(self):
        print("uf", self.uf)
        print("counters", self.counters)
        print("counter_total", self.counter_total)

    def belong_to_group(self, tuple_, group):
        for key in group.keys():
            if tuple_[key] != group[key]:
                return False
        return True

    def insert(self, tuple_):
        self.counter_total += 1
        for i, group in enumerate(self.groups):
            if self.belong_to_group(tuple_, group):  # for group that the tuple satisfies
                self.counters[i] += 1
            if self.counters[i] / self.counter_total <= self.threshold:
                self.uf[i] = True
            else:
                self.uf[i] = False

    def new_window(self):
        self.counter_total *= self.alpha
        self.counters = [x * self.alpha for x in self.counters]

