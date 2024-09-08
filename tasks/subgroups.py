
from sympy.combinatorics import Permutation, PermutationGroup
from functools import reduce

class Subgroup():
    """
    Compute the cosets of a group G for a subgroup H.
    :G, H: sympy Permutation Groups
    """
    def __init__(self, G, H):
        self.G = G #group
        self.H = H #subgroup
        self.conjugate_H = self.compute_conjugates()

    def compute_conjugates(self):
        #compute conjugate subgroups
        conjugates = set()
        conjugates.add(frozenset(self.H._elements))
        for g in self.G._elements:
            union = self.set_union(conjugates)
            if g in union:
                continue
            elif union == self.G.elements:
                break
    
            conjugate = frozenset([g * g.mul_inv(h) for h in self.H._elements])
            conjugates.add(conjugate)
        return [PermutationGroup(list(conj)) for conj in list(conjugates)]

    def coset_indices(self, conj_idx, coset_type):
        H = self.conjugate_H[conj_idx]

        cosets = set()
        cosets.add(frozenset(H._elements))
        for g in self.G._elements:
            union = self.set_union(cosets)
            if g in union:
                continue
            elif union == self.G.elements:
                break
    
            if coset_type == 'left':
                coset = frozenset([g * h for h in H._elements])
            elif coset_type == 'right':    
                coset = frozenset([h * g for h in H._elements])
            cosets.add(coset)
        cosets = list(cosets)
        return self.get_coset_idxs(cosets)
    
    def get_coset_idxs(self, cosets):
        indices = []
        for coset in cosets:
            idxs = []
            for g in coset:
                idxs.append(self.G._elements.index(g))
            indices.append(idxs)
        return indices
            
    def set_union(self, cosets):
        if len(cosets) == 0:
            return cosets
        else:
            return reduce(lambda x, y: x.union(y), cosets)


all_s5_subgroups_subwiki = {
    "C2_single_transposition": {
        "order": 2,
        "copies": 10,
        "generators": [
            Permutation(4)(0,1)
        ]
    },
    "C2_double_transposition": {
        "order": 2,
        "copies": 15,
        "generators": [
            Permutation(4)(0,1)(2,3)
        ]
    },
    "K4_double_transpositions": {
        "order": 4,
        "copies": 5,
        "generators": [
            Permutation(4)(0,1)(2,3),
            Permutation(4)(0,2)(1,3),
            Permutation(4)(0,3)(1,2)
        ]
    },

    "K4_disjoint_transpositions": {
        "order": 4,
        "copies": 15,
        "generators": [
            Permutation(4)(0,1),
            Permutation(4)(2,3),
            Permutation(4)(0,1)(2,3)
        ]
    },
    "C3": {
        "order": 3,
        "copies": 10,
        "generators": [
            Permutation(4)(0,1,2),
            Permutation(4)(0,2,1)
        ]
    },
    "C4": {
        "order": 4,
        "copies": 15,
        "generators": [
            Permutation(4)(0,1,2,3),
            Permutation(4)(0,2)(1,3),
            Permutation(4)(0,3,2,1)
        ]
    },
    "C5": {
        "order": 5,
        "copies": 6,
        "generators": [
            Permutation(0,1,2,3,4)
        ]
    },
    "C6": {
        "order": 6,
        "copies": 10,
        "generators": [
            Permutation(4)(0,1,2),
            Permutation(0)(3,4)
        ]
    },
    "S3_twisted": {
        "order": 6,
        "copies": 10,
        "generators": [
            Permutation(4)(0,1,2),
            Permutation(0,1)(3,4)
        ]
    },
    "S3": {
        "order": 6,
        "copies": 10,
        "generators": [
            Permutation(4)(0,1,2),
            Permutation(4)(0,1)
        ]
    },
    "S4": {
        "order": 24,
        "copies": 5,
        "generators": [
            Permutation(4)(0,1,2,3),
            Permutation(4)(0,1)
        ]
    },
    "D8": {
        "order": 8,
        "copies": 15,
        "generators": [
            Permutation(4)(0,1,2,3),
            Permutation(4)(0,2)
        ]
    },
    "D10": {
        "order": 10,
        "copies": 6,
        "generators": [
            Permutation(0,1,2,3,4),
            Permutation(0)(1,4)(2,3)
        ]
    },
    "S2xS3": {
        "order": 12,
        "copies": 10,
        "generators": [
            Permutation(4)(0,1,2),
            Permutation(4)(0,1),
            Permutation(0)(3,4)
        ]
    
    },
    "A4": {
        "order": 12,
        "copies": 5,
        "generators": [
            Permutation(4)(0,1,2),
            Permutation(4)(0,1)(2,3)
        ]
    },
    "A5": {
        "order": 60,
        "copies": 1, 
        "generators": [
            Permutation(0,1,2,3,4),
            Permutation(4)(0,1,2)
        ]
    },

    "F20": {
        "order": 20,
        "copies": 6,
        "generators": [
            Permutation(0,1,2,3,4),
            Permutation(0)(1,2,4,3)
        ]
    }
}


"""
Thanks to dashstander/sn-grok!
https://groupprops.subwiki.org/wiki/Symmetric_group:S5#Table_classifying_subgroups_up_to_automorphisms
"""

all_s5_subgroups_dashiell = {
    "C2_single_transposition": {
        "order": 2,
        "copies": 10,
        "generators": [
            [(1, 0, 2, 3, 4)], #1 (0 1)
            [(2, 1, 0, 3, 4)], #2 (0 2)
            [(3, 1, 2, 0, 4)], #3 (0 3)
            [(4, 1, 2, 3, 0)], #4 (0 4)
            [(0, 2, 1, 3, 4)], #5 (1 2)
            [(0, 3, 2, 1, 4)], #6 (1 3)
            [(0, 4, 2, 3, 1)], #7 (1 4)
            [(0, 1, 3, 2, 4)], #8 (2 3)
            [(0, 1, 4, 3, 2)], #9 (2 4)
            [(0, 1, 2, 4, 3)], #10 (3 4)
        ]
    },
    "C2_double_transposition": {
        "order": 2,
        "copies": 15,
        "generators": [
            [(0, 2, 1, 4, 3)], #1
            [(0, 3, 4, 1, 2)], #2
            [(0, 4, 3, 2, 1)], #3
            [(1, 0, 2, 4, 3)], #4
            [(1, 0, 3, 2, 4)], #5
            [(1, 0, 4, 3, 2)], #6
            [(2, 1, 0, 4, 3)], #7
            [(2, 3, 0, 1, 4)], #8
            [(2, 4, 0, 3, 1)], #9
            [(3, 1, 4, 0, 2)], #10
            [(3, 2, 1, 0, 4)], #11
            [(3, 4, 2, 0, 1)], #12
            [(4, 1, 3, 2, 0)], #13
            [(4, 2, 1, 3, 0)], #14
            [(4, 3, 2, 1, 0)] #15
        ]
    },
    "K4_double_transpositions": {
        "order": 4,
        "copies": 5,
        "generators": [
            [(1, 0, 3, 2, 4), (2, 3, 0, 1, 4)], # (0 1)(2 3), (0 2)(1 3), (0 3)(1 2), fix 4
            [(1, 0, 4, 3, 2), (2, 4, 0, 3, 1)], # (0 1)(2 4), (0 2)(1 4), (0 4)(1 2), fix 3
            [(0, 2, 1, 4, 3), (0, 3, 4, 1, 2)], # (1 2)(3 4), (1 3)(2 4), (1 4)(2 3), fix 0
            [(3, 4, 2, 0, 1), (1, 0, 2, 4, 3)], # (0 3)(1 4), (0 1)(3 4), (0 4)(1 3), fix 2
            [(2, 1, 0, 4, 3), (3, 1, 4, 0, 2)]  # (0 2)(3 4), (0 3)(2 4), fix 1
        ]
    },

    "K4_disjoint_transpositions": {
        "order": 4,
        "copies": 15,
        "generators": [
            [(0, 1, 2, 3, 4), (0, 2, 1, 3, 4), (0, 1, 2, 4, 3), (0, 2, 1, 4, 3)],
            [(0, 1, 2, 3, 4), (0, 3, 2, 1, 4), (0, 1, 4, 3, 2), (0, 3, 4, 1, 2)],
            [(0, 1, 2, 3, 4), (0, 4, 2, 3, 1), (0, 1, 3, 2, 4), (0, 4, 3, 2, 1)],
            [(0, 1, 2, 3, 4), (1, 0, 2, 3, 4), (0, 1, 2, 4, 3), (1, 0, 2, 4, 3)],
            [(0, 1, 2, 3, 4), (1, 0, 2, 3, 4), (0, 1, 3, 2, 4), (1, 0, 3, 2, 4)],
            [(0, 1, 2, 3, 4), (1, 0, 2, 3, 4), (0, 1, 4, 3, 2), (1, 0, 4, 3, 2)],
            [(0, 1, 2, 3, 4), (2, 1, 0, 3, 4), (0, 1, 2, 4, 3), (2, 1, 0, 4, 3)],
            [(0, 1, 2, 3, 4), (2, 1, 0, 3, 4), (0, 3, 2, 1, 4), (2, 3, 0, 1, 4)],
            [(0, 1, 2, 3, 4), (2, 1, 0, 3, 4), (0, 4, 2, 3, 1), (2, 4, 0, 3, 1)],
            [(0, 1, 2, 3, 4), (3, 1, 2, 0, 4), (0, 1, 4, 3, 2), (3, 1, 4, 0, 2)],
            [(0, 1, 2, 3, 4), (3, 1, 2, 0, 4), (0, 2, 1, 3, 4), (3, 2, 1, 0, 4)],
            [(0, 1, 2, 3, 4), (3, 1, 2, 0, 4), (0, 4, 2, 3, 1), (3, 4, 2, 0, 1)],
            [(0, 1, 2, 3, 4), (4, 1, 2, 3, 0), (0, 1, 3, 2, 4), (4, 1, 3, 2, 0)],
            [(0, 1, 2, 3, 4), (4, 1, 2, 3, 0), (0, 2, 1, 3, 4), (4, 2, 1, 3, 0)],
            [(0, 1, 2, 3, 4), (4, 1, 2, 3, 0), (0, 3, 2, 1, 4), (4, 3, 2, 1, 0)]
        ]
    },
    "C3": {
        "order": 3,
        "copies": 10,
        "generators": [
            [(3, 1, 2, 4, 0)],
            [(2, 1, 3, 0, 4)],
            [(2, 1, 4, 3, 0)],
            [(0, 3, 2, 4, 1)],
            [(1, 3, 2, 0, 4)],
            [(0, 1, 3, 4, 2)],
            [(0, 2, 3, 1, 4)],
            [(1, 2, 0, 3, 4)],
            [(1, 4, 2, 3, 0)],
            [(0, 2, 4, 3, 1)]
        ]
    },
    "C4": {
        "order": 4,
        "copies": 15,
        "generators": [
            [(1, 0, 3, 2, 4)],
            [(1, 4, 2, 0, 3)],
            [(1, 0, 4, 3, 2)],
            [(2, 1, 4, 0, 3)],
            [(1, 4, 0, 3, 2)],
            [(1, 2, 4, 3, 0)],
            [(1, 0, 2, 4, 3)],
            [(0, 2, 4, 1, 3)],
            [(0, 2, 3, 4, 1)],
            [(1, 3, 2, 4, 0)],
            [(2, 1, 0, 4, 3)],
            [(0, 2, 1, 4, 3)],
            [(2, 1, 3, 4, 0)],
            [(1, 2, 3, 0, 4)],
            [(1, 3, 0, 2, 4)]
        ]
    },
    "C5": {
        "order": 5,
        "copies": 6,
        "generators": [
            [(1, 2, 4, 0, 3)], # 1
            [(1, 3, 0, 4, 2)], # 2
            [(1, 2, 3, 4, 0)], # 3
            [(1, 4, 0, 2, 3)], # 4
            [(1, 4, 3, 0, 2)], # 5
            [(1, 3, 4, 2, 0)]  # 6
        ]
    },
    "C6": {
        "order": 6,
        "copies": 10,
        "generators": [
            [(0, 1, 3, 4, 2), (1, 0, 2, 3, 4)], #1  fix 0 1
            [(0, 3, 2, 4, 1), (2, 1, 0, 3, 4)], #2  fix 0 2
            [(0, 2, 4, 3, 1), (3, 1, 2, 0, 4)], #3  fix 0 3
            [(0, 2, 3, 1, 4), (4, 1, 2, 3, 0)], #4  fix 0 4
            [(3, 1, 2, 4, 0), (0, 2, 1, 3, 4)], #5  fix 1 2
            [(2, 1, 4, 3, 0), (0, 3, 2, 1, 4)], #6  fix 1 3
            [(2, 1, 3, 0, 4), (0, 4, 2, 3, 1)], #7  fix 1 4
            [(1, 4, 2, 3, 0), (0, 1, 3, 2, 4)], #8  fix 2 3
            [(1, 3, 2, 0, 4), (0, 1, 4, 3, 2)], #9  fix 2 4
            [(1, 2, 0, 3, 4), (0, 1, 2, 4, 3)], #10 fix 3 4
        ]
    },
    "S3_twisted": {
        "order": 6,
        "copies": 10,
        "generators": [
            [(0, 1, 3, 4, 2), (1, 0, 2, 4, 3)],
            [(0, 2, 3, 1, 4), (4, 1, 3, 2, 0)],
            [(0, 2, 4, 3, 1), (3, 1, 4, 0, 2)],
            [(0, 3, 2, 4, 1), (2, 1, 0, 4, 3)],
            [(1, 2, 0, 3, 4), (0, 2, 1, 4, 3)],
            [(1, 3, 2, 0, 4), (0, 3, 4, 1, 2)],
            [(1, 4, 2, 3, 0), (0, 4, 3, 2, 1)],
            [(2, 1, 3, 0, 4), (0, 4, 3, 2, 1)],
            [(2, 1, 4, 3, 0), (0, 3, 4, 1, 2)],
            [(3, 1, 2, 4, 0), (0, 2, 1, 4, 3)]
        ]
    },
    "S3": {
        "order": 6,
        "copies": 10,
        "generators": [
            [(0, 1, 3, 4, 2), (0, 1, 3, 2, 4)], #1  fix 0 1
            [(0, 3, 2, 4, 1), (0, 1, 2, 4, 3)], #2  fix 0 2
            [(0, 2, 4, 3, 1), (0, 2, 1, 3, 4)], #3  fix 0 3
            [(0, 2, 3, 1, 4), (0, 2, 1, 3, 4)], #4  fix 0 4
            [(3, 1, 2, 4, 0), (3, 1, 2, 0, 4)], #5  fix 1 2
            [(2, 1, 4, 3, 0), (2, 1, 0, 3, 4)], #6  fix 1 3
            [(2, 1, 3, 0, 4), (2, 1, 0, 3, 4)], #7  fix 1 4
            [(1, 4, 2, 3, 0), (1, 0, 2, 3, 4)], #8  fix 2 3
            [(1, 3, 2, 0, 4), (1, 0, 2, 3, 4)], #9  fix 2 4
            [(1, 2, 0, 3, 4), (1, 0, 2, 3, 4)], #10 fix 3 4
        ]
    },
    "S4": {
        "order": 12,
        "copies": 5,
        "generators": [
            [(0, 2, 3, 4, 1), (0, 2, 1, 3, 4)], #1 fix 0
            [(2, 1, 3, 4, 0), (2, 1, 0, 3, 4)], #2 fix 1
            [(1, 3, 2, 4, 0), (1, 0, 2, 3, 4)], #3 fix 2
            [(1, 2, 4, 3, 0), (1, 0, 2, 3, 4)], #4 fix 3
            [(1, 2, 3, 0, 4), (1, 0, 2, 3, 4)], #5 fix 4
        ]
    },
    "D8": {
        "order": 8,
        "copies": 15,
        "generators": [
            [(0, 2, 3, 4, 1), (0, 1, 4, 3, 2)],
            [(0, 2, 4, 1, 3), (0, 1, 3, 2, 4)],
            [(0, 3, 4, 2, 1), (0, 1, 2, 4, 3)],
            [(1, 2, 3, 0, 4), (0, 3, 2, 1, 4)],
            [(1, 2, 4, 3, 0), (0, 4, 2, 3, 1)],
            [(1, 3, 0, 2, 4), (0, 2, 1, 3, 4)],
            [(1, 3, 2, 4, 0), (0, 4, 2, 3, 1)],
            [(1, 4, 0, 3, 2), (0, 2, 1, 3, 4)],
            [(1, 4, 2, 0, 3), (0, 3, 2, 1, 4)],
            [(2, 1, 3, 4, 0), (0, 1, 4, 3, 2)],
            [(2, 1, 4, 0, 3), (0, 1, 3, 2, 4)],
            [(2, 3, 1, 0, 4), (0, 1, 3, 2, 4)],
            [(2, 4, 1, 3, 0), (0, 1, 4, 3, 2)],
            [(3, 1, 4, 2, 0), (0, 1, 2, 4, 3)],
            [(3, 4, 2, 1, 0), (0, 1, 2, 4, 3)]
        ]
    },
    "D10": {
        "order": 10,
        "copies": 6,
        "generators": [
            [(1, 2, 3, 4, 0), (0, 4, 3, 2, 1)],
            [(1, 2, 4, 0, 3), (0, 3, 4, 1, 2)],
            [(1, 3, 0, 4, 2), (0, 2, 1, 4, 3)],
            [(1, 3, 4, 2, 0), (0, 4, 3, 2, 1)],
            [(1, 4, 0, 2, 3), (0, 2, 1, 4, 3)],
            [(1, 4, 3, 0, 2), (0, 3, 4, 1, 2)]
        ]
    },
    "S2xS3": {
        "order": 12,
        "copies": 10,
        "generators": [
            [(0, 1, 3, 4, 2), (0, 1, 3, 2, 4), (1, 0, 2, 3, 4)], #1
            [(0, 3, 2, 4, 1), (0, 1, 2, 4, 3), (2, 1, 0, 3, 4)], #2
            [(0, 2, 4, 3, 1), (0, 2, 1, 3, 4), (3, 1, 2, 0, 4)], #3
            [(0, 2, 3, 1, 4), (0, 2, 1, 3, 4), (4, 1, 2, 3, 0)], #4
            [(3, 1, 2, 4, 0), (3, 1, 2, 0, 4), (0, 2, 1, 3, 4)], #5
            [(2, 1, 4, 3, 0), (2, 1, 0, 3, 4), (0, 3, 2, 1, 4)], #6
            [(2, 1, 3, 0, 4), (2, 1, 0, 3, 4), (0, 4, 2, 3, 1)], #7
            [(1, 4, 2, 3, 0), (1, 0, 2, 3, 4), (0, 1, 3, 2, 4)], #8
            [(1, 3, 2, 0, 4), (1, 0, 2, 3, 4), (0, 1, 4, 3, 2)], #9
            [(1, 2, 0, 3, 4), (1, 0, 2, 3, 4), (0, 1, 2, 4, 3)], #10
        ]
    
    },
    "A4": {
        "order": 12,
        "copies": 5,
        "generators": [
            [(0, 2, 1, 4, 3), (0, 3, 1, 2, 4)], # 1 fix 0
            [(2, 1, 0, 4, 3), (0, 1, 4, 2, 3)], # 2 fix 1
            [(1, 0, 2, 4, 3), (3, 0, 2, 1, 4)], # 3 fix 2
            [(1, 0, 4, 3, 2), (2, 0, 1, 3, 4)], # 4 fix 3
            [(1, 0, 3, 2, 4), (2, 0, 1, 3, 4)], # 5 fix 4
        ]
    },
    "A5": {
        "order": 60,
        "copies": 1, 
        "generators": [
            [(4, 0, 1, 2, 3), (2, 0, 1, 3, 4)]
        ]
    },

    "F20": {
        "order": 20,
        "copies": 6,
        "generators": [
            [(0, 2, 3, 4, 1), (1, 2, 4, 0, 3)],
            [(0, 2, 3, 4, 1), (1, 4, 3, 0, 2)],
            [(0, 2, 4, 1, 3), (1, 2, 3, 4, 0)],
            [(0, 2, 4, 1, 3), (1, 3, 4, 2, 0)],
            [(0, 3, 4, 2, 1), (1, 3, 0, 4, 2)],
            [(0, 3, 4, 2, 1), (1, 4, 0, 2, 3)]
        ]
    }
}


def get_s5_subgroup(name, gen_idx):
    generators = all_s5_subgroups[name]['generators'][gen_idx]
    permutations = [Permutation(gen) for gen in generators]
    subgroup = PermutationGroup(permutations)
    return subgroup