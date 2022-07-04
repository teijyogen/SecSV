import itertools
import math


def make_all_subsets(list_of_members):
    # make every possible subsets of given list_of_members
    # for size in (list_of_members):
    # use combinations to enumerate all combinations of size elements
    # append all combinations to self.data

    set_of_all_subsets = set([])

    for i in range(len(list_of_members), -1, -1):
        for element in itertools.combinations(list_of_members, i):
            # element = sorted(element)
            set_of_all_subsets.add(frozenset(element))
    return set_of_all_subsets

class ShapleyValue:
    '''A class to produce a fuzzy measure of based on a list of criteria'''

    def __init__(self, list_of_members, mu):
        # initialize a class to hold all fuzzyMeasure related objects
        self.set_of_all_subsets = sorted(set(mu.keys()))
        self.mu = mu
        self.svs = {}
        self.list_of_members = frozenset(list_of_members)
        if len(self.mu) < 1:
            return
        self.calculate_svs()

    def calculate_svs(self):
        print("*************")
        memberShapley = 0
        total = 0
        factorialTotal = math.factorial(len(self.list_of_members))
        for member in self.list_of_members:
            for subset in self.set_of_all_subsets:
                if member in subset:
                    # The muSet is the mu of Mu(B U {j})
                    muSet = self.mu.get(subset)
                    remainderSet = subset.difference(set({member}))
                    muRemainer = self.mu.get(remainderSet)
                    difference = muSet - muRemainer
                    b = len(remainderSet)
                    factValue = (len(self.list_of_members) - b - 1)
                    divisor = (math.factorial(factValue) * math.factorial(b) * 1.0) / (factorialTotal * 1.0)
                    weightValue = divisor * difference
                    memberShapley = memberShapley + weightValue
            self.svs[member] = memberShapley
            print("Shapley Value of Client " + str(member) + ": " + str(memberShapley))
            total = total + memberShapley
            memberShapley = 0
        print("Total: " + str(total))
        print("*************")

# if __name__ == '__main__':


# mu = {frozenset(): 0.4, frozenset({1}): 0.5, frozenset({2}):0.6, frozenset({1,2}): 0.9}
# sv = ShapleyValue([1, 2], mu)
# print(sv.svs)

