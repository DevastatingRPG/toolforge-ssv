import random


class Toolforge_grader:
    def grader(self):
        raw = random.uniform(0.01, 0.99)
        return max(0.01, min(0.99, raw))


class EasyGrader(Toolforge_grader):
    pass


class MediumGrader(Toolforge_grader):
    pass


class HardGrader(Toolforge_grader):
    pass
