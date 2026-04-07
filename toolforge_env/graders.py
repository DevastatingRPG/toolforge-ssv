import random


class Toolforge_grader:
    def grader(self):
        return random.uniform(0.01, 0.99)


class EasyGrader(Toolforge_grader):
    def grade(self, *args, **kwargs):
        return self.grader()


class MediumGrader(Toolforge_grader):
    def grade(self, *args, **kwargs):
        return self.grader()


class HardGrader(Toolforge_grader):
    def grade(self, *args, **kwargs):
        return self.grader()