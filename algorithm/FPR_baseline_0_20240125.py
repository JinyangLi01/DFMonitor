import sys


def sizeof(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict): return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)): return size + sum(map(sizeof, obj))
    return size

class FPR_baseline:
    def __init__(self, monitored_groups, alpha, threshold):
        self.groups = monitored_groups
        self.uf = [False]*len(monitored_groups)
        self.counters_TN = [0]*len(monitored_groups)
        self.counters_FP = [0]*len(monitored_groups)
        self.threshold = threshold
        self.alpha = alpha

    def find(self, group):
        idx = self.groups.index(group)
        return self.uf[idx]

    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + sys.getsizeof(self.groups[0]) * len(self.groups)
        size += sys.getsizeof(self.uf) + sys.getsizeof(self.uf[0]) * len(self.uf)
        size += sys.getsizeof(self.counters_TN) + sys.getsizeof(self.counters_TN[0]) * len(self.counters_TN)
        size += sys.getsizeof(self.counters_FP) + sys.getsizeof(self.counters_FP[0]) * len(self.counters_FP)
        size += sys.getsizeof(self.threshold)
        size += sys.getsizeof(self.alpha)
        return size

    def get_size_recursive(self):
        size = 0
        size += sizeof(self.groups)
        size += sizeof(self.uf)
        size += sizeof(self.counters_TN)
        size += sizeof(self.counters_FP)
        size += sizeof(self.threshold)
        size += sizeof(self.alpha)
        return

    def print(self):
        print("FPR_baseline, groups: ", self.groups)
        print("uf", self.uf)
        print("counters_TN", self.counters_TN)
        print("counters_FP", self.counters_FP)
        print("FPR of groups: ", [self.counters_FP[i] / (self.counters_TN[i] + self.counters_FP[i])
                                  for i in range(len(self.counters_FP))] )
        print("\n")


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
                if self.counters_FP[i] / (self.counters_TN[i] + self.counters_FP[i]) > self.threshold:
                    self.uf[i] = True
                else:
                    self.uf[i] = False


    def new_window(self):
        self.counters_FP = [x * self.alpha for x in self.counters_FP]
        self.counters_TN = [x * self.alpha for x in self.counters_TN]


