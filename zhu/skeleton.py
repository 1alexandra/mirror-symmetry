import numpy as np
import cv2

from zhu.draw_tools import imread_bw


def bw_to_rgb(origin):
    rep = np.repeat(origin, 3, axis=1)
    new_shape = (*origin.shape, 3)
    return rep.reshape(new_shape)


class SkeletonVertex:
    def __init__(self, x, y, degree, radius):
        self.x = x
        self.y = y
        self.z = x + 1j * y
        self.skeleton_degree = degree
        self.radius = radius

        self._graph_index = None
        self.neibs = None

        self.draw_kwargs = {
            'color': (255, 255, 0),
            'radius': 5,
            'thickness': -1,
        }

    @property
    def coord_cv(self):
        return (int(self.x), int(self.y))

    @property
    def graph_index(self):
        return self._graph_index

    @graph_index.setter
    def graph_index(self, index):
        self._graph_index = index
        self.neibs = []

    def __eq__(self, other):
        return self.coord_cv == other.coord_cv

    def __hash__(self):
        return hash(self.coord_cv)

    def __repr__(self):
        return f'SkeletonVertex{self.graph_index} {self.coord_cv}'

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx ** 2 + dy ** 2) ** 0.5

    def draw(self, board):
        board = cv2.circle(
            board, self.coord_cv, **self.draw_kwargs)
        if type(board) is cv2.UMat:
            board = board.get()
        return board


class SkeletonEdge:
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1
        self.vec = v1.z - v0.z

        self.draw_kwargs = {
            'color': (255, 0, 0),
            'thickness': 5,
        }

    @property
    def length(self):
        return self.v0.distance(self.v1)

    def draw(self, board):
        start_point = self.v0.coord_cv
        end_point = self.v1.coord_cv

        board = cv2.line(
            board, start_point, end_point, **self.draw_kwargs)
        if type(board) is cv2.UMat:
            board = board.get()
        return board


class Skeleton:
    def __init__(self, edge_list, origin_path):
        self.origin_path = origin_path
        self.origin = imread_bw(origin_path)

        self.edge_list = edge_list
        self.vertexes = self.graph_vertexes(edge_list)

        self._graph = None

        self.degree_counts = {}

    @staticmethod
    def graph_vertexes(edge_list):
        vertex_vertex = {}
        for e in edge_list:
            assert e.v0 != e.v1
            if e.v0 in vertex_vertex:
                e.v0 = vertex_vertex[e.v0]
            else:
                e.v0.graph_index = len(vertex_vertex)
                vertex_vertex[e.v0] = e.v0
            if e.v1 in vertex_vertex:
                e.v1 = vertex_vertex[e.v1]
            else:
                e.v1.graph_index = len(vertex_vertex)
                vertex_vertex[e.v1] = e.v1
            e.v0.neibs.append(e.v1)
            e.v1.neibs.append(e.v0)
        vertexes = sorted(vertex_vertex.values(), key=lambda x: x.graph_index)
        return list(vertexes)

    @property
    def graph(self):
        if self._graph is None:
            raise('Not implemented')
        return self._graph

    def count_degree(self, degree):
        if degree not in self.degree_counts:
            cnt = sum([v.skeleton_degree == degree for v in self.vertexes])
            self.degree_counts[degree] = cnt
        return self.degree_counts[degree]

    def draw(self, board=None):
        if board is None:
            board = bw_to_rgb(self.origin)[::-1]
        for edge in self.edge_list:
            board = edge.draw(board)
            for v in (edge.v0, edge.v1):
                if len(v.neibs) != 2:
                    board = v.draw(board)
        return board
