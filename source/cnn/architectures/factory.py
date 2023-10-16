from .ach_a import AchA
from .ach_b import AchB
from .ach_c import AchC


class Factory:
    def build(arch: str):
        match arch:
            case "a":
                return AchA
            case "b":
                return AchB
            case "c":
                return AchC
            case _:
                raise ValueError("Please select an existing architecture")
