import collections
import numpy as np
import cv2

from zhu.curve_fit import BezierCurve
from zhu.sym_contour import SymContour
from zhu.draw_tools import imread_bw

from zhu.vertex import Vertex


def bw_to_rgb(origin):
    rep = np.repeat(origin, 3, axis=1)
    new_shape = (*origin.shape, 3)
    return rep.reshape(new_shape)


class SkeletonVertex(Vertex):
    def __init__(self, x, y, degree, radius):
        super().__init__(x, y)
        self.skeleton_degree = degree
        self.radius = radius

        self._graph_index = None
        self.neibs = None

    @property
    def graph_index(self):
        return self._graph_index

    @graph_index.setter
    def graph_index(self, index):
        self._graph_index = index
        self.neibs = []

    @property
    def degree(self):
        return len(self.neibs) if self.neibs != [] else self.skeleton_degree

    def __repr__(self):
        return f'SkeletonVertex{self.graph_index} {self.coord_cv}'


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

    @property
    def radius_diff(self):
        return np.abs(self.v0.radius - self.v1.radius)

    @property
    def radius_min(self):
        return np.min([self.v0.radius, self.v1.radius])

    @property
    def radius_max(self):
        return np.max([self.v0.radius, self.v1.radius])

    def vertex_distance(self, vertex):
        return min(self.v0.distance(vertex), self.v1.distance(vertex))

    def draw(self, board):
        start_point = self.v0.coord_cv
        end_point = self.v1.coord_cv

        board = cv2.line(
            board, start_point, end_point, **self.draw_kwargs)
        if type(board) is cv2.UMat:
            board = board.get()
        return board


class Skeleton:
    def __init__(self, edge_list, origin_path, contour=None):
        self.origin_path = origin_path
        self.origin = imread_bw(origin_path)
        self.contour = contour

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
            board = cv2.imread(self.origin_path)[::-1]
        for edge in self.edge_list:
            board = edge.draw(board)
            for v in (edge.v0, edge.v1):
                if len(v.neibs) != 2:
                    board = v.draw(board)
        return board


class LeaveSkeleton(Skeleton):
    def __init__(self, edge_list, origin_path, contour=None):
        super().__init__(edge_list, origin_path, contour)

        self.stalkity_angle = np.pi / 180 * 7.5
        self.stalkity_radius = 0.1

        self.curve_points_count = 10
        self.curve_points_show_count = 50
        self.curve_degree = 2

        self.sc_kwargs = {}

        self._center = None
        self._stalk = None
        self._stalk_head = None
        self._leave_head = None
        self._body_vertexes = None
        self._body_curve = None
        self._leave_body_edge_length = None
        self._stalk_edges_final = None
        self._sym_contour = None

        self.draw_kwargs = {
            'text': {
                'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
                'fontScale': 3,
                'color': (0, 0, 0),
                'thickness': 5,
                'lineType': 2,
            }
        }

    @property
    def center_vertex(self):
        if self._center is None:
            radiuses = [v.radius for v in self.vertexes]
            self._center = self.vertexes[np.argmax(radiuses)]
        return self._center

    @staticmethod
    def skeleton_edge_extension(v, v_e):
        vec = v.z - v_e.z
        vec = vec / np.abs(vec)
        z = v.z + vec * v.radius
        x, y = int(round(np.real(z))), int(round(np.imag(z)))
        return Vertex(x, y)

    def skeleton_leave_end(self, v):
        assert v.degree == 1
        v_e = None
        for e in self.edge_list:
            if e.v0 == v:
                v_e = e.v1
            elif e.v1 == v:
                v_e = e.v0
            if v_e is not None:
                break
        assert v_e is not None
        return self.skeleton_edge_extension(v, v_e)

    @staticmethod
    def bfs_count_branching(root):
        branching = {}
        queue = collections.deque([root])
        branching[root] = 0
        while queue:
            parent = queue.popleft()
            for child in parent.neibs:
                if child not in branching:
                    branching[child] = branching[parent] \
                        + (1 if parent.degree > 2 else 0)
                    queue.append(child)
        return branching

    @staticmethod
    def bfs_edge_length(root):
        length = {}
        queue = collections.deque([root])
        length[root] = 0
        while queue:
            parent = queue.popleft()
            for child in parent.neibs:
                if child not in length:
                    length[child] = length[parent] + parent.distance(child)
                    queue.append(child)
        return length

    @staticmethod
    def bfs_path(root):
        path = {}
        queue = collections.deque([root])
        path[root] = [root]
        while queue:
            parent = queue.popleft()
            for child in parent.neibs:
                if child not in path:
                    path[child] = path[parent] + [child]
                    queue.append(child)
        return path

    def radius_stalkity(self, e):
        if e.length == 0:
            return np.inf
        return e.radius_max <= self.center_vertex.radius * self.stalkity_radius

    def angle_stalkity(self, e):
        if e.length == 0:
            return False
        return e.radius_diff / 2 / e.length <= np.tan(self.stalkity_angle)

    @property
    def _stalk_edges(self):
        if self._stalk is None:
            self._stalk = [
                e for e in self.edge_list
                if self.angle_stalkity(e) and self.radius_stalkity(e)
            ]
        return self._stalk

    @property
    def _stalk_vertexes(self):
        edges = self._stalk_edges
        return list(set([e.v0 for e in edges]) | set([e.v1 for e in edges]))

    @property
    def stalk_head(self):
        if self._stalk_head is None:
            branching = self.bfs_count_branching(self.center_vertex)
            vertexes = self._stalk_vertexes.copy()
            min_branching = min([branching[v] for v in vertexes])
            vertexes = [v for v in vertexes if branching[v] == min_branching]
            sorting = [v.distance(self.center_vertex) for v in vertexes]
            self._stalk_head = vertexes[np.argmin(sorting)]
        return self._stalk_head

    @property
    def leave_head(self):
        if self._leave_head is None:
            length = self.bfs_edge_length(self.stalk_head)
            max_length = max(length.values())
            vertexes = [v for v in self.vertexes if length[v] == max_length]
            distance = [v.distance(self.stalk_head) for v in vertexes]
            v = vertexes[np.argmax(distance)]
            self._leave_head = v
        return self.skeleton_leave_end(self._leave_head)

    @property
    def body_vertexes(self):
        if self._body_vertexes is None:
            path = self.bfs_path(self.stalk_head)
            assert self.leave_head is not None
            body_path = path[self._leave_head]
            self._body_vertexes = body_path + [self.leave_head]
        return self._body_vertexes

    @property
    def body_curve(self):
        if self._body_curve is None:
            assert self.leave_head is not None
            length = 0
            vertexes = self.body_vertexes
            for i, v in enumerate(vertexes):
                if i > 0:
                    length += v.distance(vertexes[i - 1])

            step = length / self.curve_points_count

            u = [p.z for p in vertexes]
            seg_ind, seg_start, cur_step = 0, u[0], step
            w = []
            for i in range(int(round(length / step)) + 1):
                w.append(seg_start)
                while True:
                    seg_end = u[(seg_ind + 1) % len(u)]
                    seg_vec = seg_end - seg_start
                    seg_len = abs(seg_vec)
                    if seg_len < cur_step:
                        seg_ind += 1
                        seg_start = seg_end
                        cur_step -= seg_len
                    else:
                        seg_start += seg_vec / seg_len * cur_step
                        cur_step = step
                        break
            points = np.array(w)
            points = np.array([np.real(w), np.imag(w)]).T
            self._body_curve = BezierCurve(points, self.curve_degree)
        return self._body_curve

    @property
    def stalk_edges(self):
        if self._stalk_edges_final is None:
            edges = []
            root = self.stalk_head
            visited = set(
                [root] + [
                    v for v in root.neibs
                    if v.distance(self.leave_head)
                    < root.distance(self.leave_head)])
            while True:
                childs = [v for v in root.neibs if v not in visited]
                if len(childs) != 1:
                    break
                edges.append(SkeletonEdge(root, childs[0]))
                root = childs[0]
                visited.add(root)
            self._stalk_edges_final = edges
        return self._stalk_edges_final

    @property
    def stalk_vertexes(self):
        edges = self.stalk_edges
        return list(set([e.v0 for e in edges]) | set([e.v1 for e in edges]))

    @property
    def stalk_tail(self):
        return self.stalk_edges[-1].v1

    @property
    def leave_body_length(self):
        assert self.body_curve is not None
        return self.body_curve.length

    @property
    def leave_stalk_length(self):
        return sum([e.length for e in self.stalk_edges]) \
            + self.stalk_tail.radius

    @property
    def leave_length(self):
        return self.leave_body_length + self.leave_stalk_length

    @property
    def leave_stalk_length_part(self):
        return self.leave_stalk_length / self.leave_body_length

    @property
    def SymContour(self):
        assert self.contour is not None
        if self._sym_contour is None:
            s = self.contour.origin
            vs = [Vertex(np.real(s_), np.imag(s_)) for s_ in s]
            coords = [self.body_curve.coord(v) for v in vs]
            x, y = np.array(coords).T
            u = x + 1j * y
            u = u[x > 0]
            self._sym_contour = SymContour(u, **self.sc_kwargs)
        return self._sym_contour

    def draw(self, board=None):
        board = super().draw(board)

        for edge in self.stalk_edges:
            edge.draw_kwargs['color'] = (0, 0, 255)
            board = edge.draw(board)

        board = self.body_curve.draw(board, n=self.curve_points_show_count)

        for vertex in (
            self.center_vertex,
            self.stalk_head,
            self.stalk_tail,
            self.leave_head,
        ):
            vertex.draw_kwargs['color'] = (255, 0, 255)
            vertex.draw_kwargs['radius'] = 15
            board = vertex.draw(board)

        for i, value in enumerate([
            # self.leave_stalk_length_part,
            self.SymContour.Sym_measure,
        ]):
            board = board[::-1]
            corner = (
                int(self.stalk_tail.x + 10),
                board.shape[0] - (int(self.stalk_tail.y) + 10 * (i + 1)))
            board = cv2.putText(
                board, str(round(value, 3)), corner,
                **self.draw_kwargs['text'])
            if type(board) is cv2.UMat:
                board = board.get()
            board = board[::-1]

        return board
