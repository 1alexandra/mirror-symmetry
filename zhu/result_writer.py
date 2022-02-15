import os
from time import time
from timeit import timeit

from zhu import SymContour
from zhu import DataFolder

from zhu import TIMEIT_ITERS
from zhu import Q_MAX_MULT


def default_format(kwargs, signs=6, add_time=True):
    image_filename = kwargs['image_filename']
    p1, p2 = kwargs['vertexes']
    q = round(kwargs['sym_measure'], signs)
    ans = f'{image_filename} {p1.z} {p2.z} {q}'
    if 'time' in kwargs:
        t = round(kwargs['time'], signs)
        ans += f' {t}'
    return ans


class ResultWriter:
    def __init__(
        self,
        format_function=default_format,
        timeit_number=TIMEIT_ITERS,
        force_single_axis=True
    ):
        self.format_f = format_function
        self.iters = timeit_number
        self.force_single = force_single_axis

    def axis(self, line):
        return {'sym_measure': line.q}

    def contour(self, sym_cnt):
        kwargs_list = []
        for line in sym_cnt.Axis_list:
            kwargs = self.axis(line)
            kwargs['vertexes'] = line.vertexes(sym_cnt.Pixels)
            if self.iters > 0:
                kwargs['time'] = timeit(
                    'lines = SymContour(u).Axis_list',
                    number=self.iters,
                    globals={
                        'SymContour': SymContour,
                        'u': sym_cnt.origin
                    })
            kwargs_list.append(kwargs)
        return kwargs_list

    def image(self, sym_img):
        kwargs_list = []
        for cnt in sym_img:
            kwargs = self.contour(cnt)
            for kw in kwargs:
                kw['image_filename'] = sym_img.name
                kw['q0'] = kwargs[0]['sym_measure']
            kwargs_list += kwargs
        return kwargs_list

    def folder(self, data_folder, q_max_mult=Q_MAX_MULT):
        kwargs_list = []
        for image in data_folder:
            kwargs_list += self.image(image)
        if self.force_single:
            q_max = max([kw['q0'] for kw in kwargs_list])
            kwargs_list = [kw for kw in kwargs_list
                           if kw['sym_measure'] <= q_max * q_max_mult]
        return kwargs_list

    def to_text(self, kwargs_list):
        return '\n'.join([self.format_f(line) for line in kwargs_list])

    def parent_folder(
        self,
        main_folder,
        subfolders=None,
        sym_image_kwargs={},
        res_folder='../writer_results',
        format_='txt',
        log=True
    ):
        if subfolders is None:
            subfolders = os.listdir(path='./' + main_folder)
        if not os.path.isdir(res_folder):
            os.mkdir(res_folder)
        for folder in subfolders:
            if log:
                print(folder)
                start = time()
            df = DataFolder(main_folder + '/' + folder,
                            sym_image_kwargs=sym_image_kwargs)
            text = self.to_text(self.folder(df))
            res_path = res_folder + '/' + folder.split('/')[-1] + '.' + format_
            with open(res_path, 'w+') as f:
                f.write(text)
            if log:
                print('=====')
                print(text)
                print('=====')
                print(f'Total time: {round(time()-start, 3)} s.')
