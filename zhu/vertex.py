import cv2


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = x + 1j * y

        self.draw_kwargs = {
            'color': (255, 255, 0),
            'radius': 5,
            'thickness': -1,
        }

    @property
    def coord_cv(self):
        return (int(self.x), int(self.y))

    def __eq__(self, other):
        return self.coord_cv == other.coord_cv

    def __hash__(self):
        return hash(self.coord_cv)

    def __repr__(self):
        return f'Vertex {self.coord_cv}'

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx ** 2 + dy ** 2) ** 0.5

    def draw(self, board):
        board = cv2.circle(
            cv2.UMat(board), self.coord_cv, **self.draw_kwargs)
        if type(board) is cv2.UMat:
            board = board.get()
        return board
