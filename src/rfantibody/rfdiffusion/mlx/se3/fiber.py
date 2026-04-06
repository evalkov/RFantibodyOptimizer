"""
Fiber type system for SE3-Transformer in MLX.

Describes the structure of equivariant features split by degree type.
Type-k features have dimension 2k+1:
  - Type 0: invariant scalars (dim 1)
  - Type 1: equivariant 3D vectors (dim 3)
  - Type 2: equivariant symmetric traceless matrices (dim 5)
"""

from collections import namedtuple
from itertools import product
from typing import Dict

import mlx.core as mx


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


FiberEl = namedtuple('FiberEl', ['degree', 'channels'])


class Fiber(dict):
    """Structure descriptor for SE3 equivariant features."""

    def __init__(self, structure):
        if isinstance(structure, dict):
            structure = [FiberEl(int(d), int(m))
                         for d, m in sorted(structure.items(), key=lambda x: x[1])]
        elif not isinstance(structure[0], FiberEl):
            structure = list(map(lambda t: FiberEl(*t),
                                 sorted(structure, key=lambda x: x[1])))
        self.structure = structure
        super().__init__({d: m for d, m in self.structure})

    @property
    def degrees(self):
        return sorted([t.degree for t in self.structure])

    @property
    def channels(self):
        return [self[d] for d in self.degrees]

    @property
    def num_features(self):
        """Total size if all features were concatenated."""
        return sum(t.channels * degree_to_dim(t.degree) for t in self.structure)

    @staticmethod
    def create(num_degrees: int, num_channels: int):
        """Create Fiber with degrees 0..num_degrees-1, all same multiplicity."""
        return Fiber([(degree, num_channels) for degree in range(num_degrees)])

    @staticmethod
    def from_features(feats: Dict[str, mx.array]):
        """Infer Fiber structure from a feature dict."""
        structure = {}
        for k, v in feats.items():
            degree = int(k)
            assert len(v.shape) == 3, 'Feature shape should be (N, C, 2D+1)'
            assert v.shape[-1] == degree_to_dim(degree)
            structure[degree] = v.shape[-2]
        return Fiber(structure)

    def __getitem__(self, degree: int):
        return dict(self.structure).get(degree, 0)

    def __iter__(self):
        return iter(self.structure)

    def __mul__(self, other):
        if isinstance(other, Fiber):
            return product(self.structure, other.structure)
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels * other for t in self.structure})

    def __add__(self, other):
        if isinstance(other, Fiber):
            return Fiber({t.degree: t.channels + other[t.degree]
                          for t in self.structure})
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels + other for t in self.structure})

    def __repr__(self):
        return str(self.structure)

    @staticmethod
    def combine_max(f1, f2):
        new_dict = dict(f1.structure)
        for k, m in f2.structure:
            new_dict[k] = max(new_dict.get(k, 0), m)
        return Fiber(list(new_dict.items()))

    @staticmethod
    def combine_selectively(f1, f2):
        new_dict = dict(f1.structure)
        for k in f1.degrees:
            if k in f2.degrees:
                new_dict[k] += f2[k]
        return Fiber(list(new_dict.items()))

    def to_attention_heads(self, tensors: Dict[str, mx.array], num_heads: int):
        """Convert fiber dict to attention head format: (N, num_heads, -1)."""
        fibers = [
            tensors[str(degree)].reshape(
                *tensors[str(degree)].shape[:-2], num_heads, -1
            )
            for degree in self.degrees
        ]
        return mx.concatenate(fibers, axis=-1)
