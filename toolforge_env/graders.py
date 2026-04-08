"""
Backward-compatible grader re-exports.

The canonical graders now live in server/graders.py.
This file re-exports them so existing imports still work.
"""

from server.graders import grade_easy, grade_medium, grade_hard

# Legacy class-based aliases (kept for any old references)
class Toolforge_grader:
    def grader(self):
        return grade_easy()

class EasyGrader(Toolforge_grader):
    def grade(self, *args, **kwargs):
        return grade_easy(*args, **kwargs)

class MediumGrader(Toolforge_grader):
    def grade(self, *args, **kwargs):
        return grade_medium(*args, **kwargs)

class HardGrader(Toolforge_grader):
    def grade(self, *args, **kwargs):
        return grade_hard(*args, **kwargs)