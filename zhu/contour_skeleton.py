import os
import time
import numpy as np
import cv2
from PIL import Image

from zhu import PATTERN_SPECTRUM_EXE
from zhu import SkeletonVertex, SkeletonEdge
from zhu import Skeleton, LeaveSkeleton
from zhu.draw_tools import imread_bw


class SkeletonBuilder:
    def __init__(self, sym_image, index=0, is_leave=False, **kwargs):
        assert 0 <= index < len(sym_image.Contours_list)
        self.sym_image = sym_image
        self.contour_index = index
        self.pattern_spectrum_kwargs = kwargs

        self.skeleton_class = LeaveSkeleton if is_leave else Skeleton
        self.edges = self.read_skeleton_edges()
        self.path = self.sym_image.img_path

    Contour = property()
    Skeleton = property()

    @Contour.getter
    def Contour(self):
        return self.sym_image.Contours_list[self.contour_index]

    def silhouette(self):
        assert self.sym_image is not None
        origin_bw = imread_bw(self.sym_image.img_path)
        cnt_cv = self.Contour.Contour_cv
        board = np.zeros(origin_bw.shape)
        board = cv2.fillPoly(board, pts=[cnt_cv], color=(255, 255, 255))
        return board

    def tmp_path(self, name):
        assert self.contour_index is not None
        path = os.path.join(
            self.sym_image.tmp_folder,
            f'{self.sym_image.name}_{self.contour_index}_{name}')
        return path

    def save_silhoette(self):
        path = self.tmp_path('silhoette.bmp')
        board = self.silhouette()
        im = Image.fromarray((255 - board.astype(np.uint8)))
        im.save(path)

        files_path = self.tmp_path('skeleton_in.txt')
        with open(files_path, 'w') as f:
            f.write(path)
            f.write('\n')

    @staticmethod
    def read_files(path):
        with open(path, 'r') as f:
            files = f.read().split('/n')
        files = [f for f in files if f.split() != '']
        return files

    def save_skeleton(self):
        args = {
            'a': 3,
            's': 0.0366,
            'p': 3.67,
            'i': self.tmp_path('skeleton_in.txt'),
            'o': self.tmp_path('skeleton.txt'),
            't': self.tmp_path('skeleton_out.txt'),
            'n': 5126,
        }
        args.update(self.pattern_spectrum_kwargs)

        if not os.path.isfile(args['i']):
            self.save_silhoette()

        command = ' '.join(
            [PATTERN_SPECTRUM_EXE] + [f'-{k} {v}' for k, v in args.items()])
        result = os.system(command)
        assert result == 0

        files = self.read_files(args['i'])
        for sleeping in range(180):
            out = self.read_files(args['t'])
            if len(out) == len(files):
                assert out == files
                return
            assert len(out) < len(files)
            assert out == files[:len(out)]
            time.sleep(1)

        raise('PatternSpectrum runtime too long')
        return

    def read_skeleton_edges(self):
        skeleton_path = self.tmp_path('skeleton.txt')
        if not os.path.isfile(skeleton_path):
            self.save_skeleton()
        with open(skeleton_path, 'r') as f:
            descriptor = f.read().split('\n')[0]
        self.clear()

        descriptor_len = 8
        digits = descriptor.split(' ')[:-1]
        edge_list = []
        for edge_i in range(len(digits) // descriptor_len):
            part = digits[
                (edge_i * descriptor_len):((edge_i + 1) * descriptor_len)]
            x0, y0, d0, r0, x1, y1, d1, r1 = [float(d) for d in part]
            v0 = SkeletonVertex(x0, y0, d0, r0)
            v1 = SkeletonVertex(x1, y1, d1, r1)
            if v0 == v1:
                continue
            e = SkeletonEdge(v0, v1)
            edge_list.append(e)
        return edge_list

    def clear(self):
        for f in (
            'silhoette.bmp',
            'silhoette.bin',
            'skeleton_in.txt',
            'skeleton_out.txt',
        ):
            path = self.tmp_path(f)
            if os.path.isfile(path):
                os.remove(path)

    @Skeleton.getter
    def Skeleton(self):
        return self.skeleton_class(self.edges, self.path, self.Contour)
