from algorithm import config
import sys


def sizeof(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict): return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)): return size + sum(map(sizeof, obj))
    return size


class DF_FPR:
    def __init__(self, monitored_groups, alpha, threshold):
        self.groups = monitored_groups
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

    def get_size(self):
        size = 0
        size += sys.getsizeof(self.groups) + sys.getsizeof(self.groups[0]) * len(self.groups)
        size += sys.getsizeof(self.uf) + sys.getsizeof(self.uf[0]) * len(self.uf)
        size += sys.getsizeof(self.delta) + sys.getsizeof(self.delta[0]) * len(self.delta)
        size += sys.getsizeof(self.alpha)
        size += sys.getsizeof(self.threshold)
        return size

    def get_size_recursive(self):
        size = 0
        size += sizeof(self.groups)
        size += sizeof(self.uf)
        size += sizeof(self.delta)
        size += sizeof(self.alpha)
        size += sizeof(self.threshold)
        return size

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
        # self.delta = [round(x * self.alpha, config.decimal) for x in self.delta]
        self.delta = [x * self.alpha for x in self.delta]



