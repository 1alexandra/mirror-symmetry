from zhu import Axis


class SymAxis(Axis):
    def __init__(self, p1, p2, q):
        super().__init__(p1, p2)
        self.q = q

    def __str__(self):
        return super().__str__() + f'\n q = {self.q}'
