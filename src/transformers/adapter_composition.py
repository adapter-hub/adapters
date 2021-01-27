import itertools
from collections.abc import Sequence
from typing import List, Set, Union


class AdapterCompositionBlock(Sequence):
    def __init__(self, *children):
        self.children = [parse_composition(b, None) for b in children]

    def __getitem__(self, key):
        return self.children[key]

    def __len__(self):
        return len(self.children)

    def __repr__(self):
        child_repr = ", ".join(map(str, self.children))
        return f"{self.__class__.__name__}[{child_repr}]"

    def first(self):
        if not isinstance(self.children[0], AdapterCompositionBlock):
            return self.children[0]
        else:
            return self.children[0].first()

    def last(self):
        if not isinstance(self.children[-1], AdapterCompositionBlock):
            return self.children[-1]
        else:
            return self.children[-1].last()

    def flatten(self) -> Set[str]:
        return set(itertools.chain(*[[b] if isinstance(b, str) else b.flatten() for b in self.children]))


class Stack(AdapterCompositionBlock):
    def __init__(self, *stack_layers: List[Union[AdapterCompositionBlock, str]]):
        super().__init__(*stack_layers)


class Fuse(AdapterCompositionBlock):
    def __init__(self, *fuse_stacks: List[Union[AdapterCompositionBlock, str]]):
        super().__init__(*fuse_stacks)

    # TODO-V2 pull this up to all block classes?
    @property
    def name(self):
        return ",".join([c if isinstance(c, str) else c.last() for c in self.children])


class Split(AdapterCompositionBlock):
    def __init__(self, left: str, right: str, split_index: int):
        super().__init__(left, right)
        assert split_index > 0
        self.left = left
        self.right = right
        self.split_index = split_index


# TODO-V2: maybe add some checks regarding depth?
def parse_composition(adapter_composition, level=0) -> AdapterCompositionBlock:
    if isinstance(adapter_composition, AdapterCompositionBlock):
        return adapter_composition
    elif isinstance(adapter_composition, str):
        if level == 0:
            return Stack(adapter_composition)
        else:
            return adapter_composition
    elif isinstance(adapter_composition, Sequence):
        # for backwards compatibility
        if level == 1:
            block_class = Fuse
        else:
            block_class = Stack
        level = level + 1 if level is not None else None
        return block_class(*[parse_composition(b, level) for b in adapter_composition])
    else:
        raise TypeError(adapter_composition)
