import glob
import logging
import multiprocessing
import os
import threading
import time
from math import ceil

import openslide
from openslide.deepzoom import DeepZoomGenerator

from src.config import WSI_OUTPUT_DIR

logger = logging.getLogger(__name__)


class StartAnalysis:
    def __init__(self, tile_size=64, overlap=0, limit_bounds=True, lev_sec=2):
        self.lev_sec = lev_sec
        self.limit_bounds = limit_bounds
        self.overlap = overlap
        self.tile_size = tile_size
        self.levi = 1000
        self.file_path = ''
        self.generator = ''
        self.ntiles_y = 0
        self.path_folder = ''
        self.path_th = ''
        self.newshape = (64, 64, 3)

    def list_files(self, path_svs, save_path, progress_callback, view):
        list_name = os.listdir(path_svs)
        list_f_svs = glob.glob(os.path.join(path_svs, '*.svs'))
        print(list_f_svs)
        for n, j in enumerate(list_f_svs, 0):
            print(n, j)
            self.openSvs(j, flag=1)
            numx = self.tile_gen(state=2)
            self.process_create_dataset(numx, list_name[n], save_p=save_path)
            progress_callback.emit(100*(n+1)/len(list_f_svs))

    def openSvs(self, file_path, flag=0):
        try:
            self.file_path = file_path
            self.slide = openslide.OpenSlide(file_path)
            self.get_prop()
            #self.get_thumb()
            if flag == 0:
                self.base_folder_manager()
        except openslide.OpenSlideError:
            print("Cannot find file '" + file_path + "'")

    def base_folder_manager(self):
        """Create the folders where the thumbnail and the tiles of the image are
        stored.

        By default the output folder is placed next to the input ``.svs``
        file, i.e. ``<svs_dir>/data/<svs_name>_<lev_sec>/``. If the
        ``WSI_OUTPUT_DIR`` environment variable is set (see
        :mod:`src.config`), the output folder is placed under that
        directory instead — useful for read-only SVS sources and Docker
        containers that bind-mount a dedicated output volume.
        """

        svs_dir = os.path.dirname(self.file_path)
        svs_name = os.path.basename(self.file_path)
        base = svs_name[: svs_name.rindex('.')]
        folder_name = base + '_' + str(self.lev_sec)
        root = WSI_OUTPUT_DIR or svs_dir
        newpath = os.path.join(root, 'data', folder_name)
        os.makedirs(os.path.join(newpath, 'thumbnail'), exist_ok=True)

        self.path_th = os.path.join(newpath, 'thumbnail')
        self.path_folder = newpath + os.sep

    def get_thumb(self):
        """"Create the thumbnail of the image, ready for the classification phase."""

        n_levels = len(self.list_levels)
        lev = self.lev_sec
        if not (-n_levels <= lev < n_levels):
            logger.warning(
                "lev_sec=%d out of range for level_count=%d; falling back to "
                "the most detailed level (0).", lev, n_levels,
            )
            lev = 0
        image = self.slide.get_thumbnail(self.list_levels[lev])
        image.save(self.path_th + '/th.png')
        return self.path_folder

    def get_prop(self):
        pro = self.slide.properties
        tile_w = pro['openslide.level[0].tile-width']
        lev_count = self.slide.level_count
        lev_down = self.slide.level_downsamples
        print(lev_down)
        mag = int(pro[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        available_mag = tuple(mag / x for x in lev_down)
        acq_date = pro.get('aperio.date')
        self.list_levels = self.slide.level_dimensions

    def tile_gen(self, state=9):
        """Call this function to divide the slice in tiles, it manage the dimension and the edge cuts.
        This function call the method 'manage_process' that create same vectors for the next step, run the theads"""

        # Defensive clamp: small/thumbnail SVS fixtures may have fewer native
        # OpenSlide levels than the default ``lev_sec`` requested by the
        # caller (e.g. GetTheta test fixtures have level_count=2 but the app
        # default is lev_sec=2). Without this, ``self.list_levels[self.lev_sec]``
        # below raised IndexError. Pick the finest *available* level instead
        # of failing hard — the rest of the loop merely searches the DeepZoom
        # grid for a match and falls back to max resolution if none found.
        n_levels = len(self.list_levels)
        if not (0 <= self.lev_sec < n_levels):
            logger.warning(
                "lev_sec=%d out of range for level_count=%d; clamping to %d.",
                self.lev_sec, n_levels, n_levels - 1,
            )
            self.lev_sec = n_levels - 1

        self.generator = DeepZoomGenerator(self.slide, tile_size=self.tile_size, overlap=self.overlap, limit_bounds=self.limit_bounds)
        dim = self.generator.level_dimensions
        ntile = self.generator._t_dimensions

        for i, a in enumerate(dim):
            if self.list_levels[self.lev_sec][1] == a[1] or self.list_levels[self.lev_sec][1] == (a[1]-1) or self.list_levels[self.lev_sec][1] == (a[1]+1):
                self.levi = i
                print(f'found the right level {i} -- rr = {self.list_levels[self.lev_sec][1]} --- a = {a[1]}')
                print(self.list_levels)
            else:
                pass

        try:
            numx, numy = ntile[self.levi]
            print(f'{numx}---{numy}')
        except IndexError:
            numx, numy = ntile[-1]
            self.levi = len(ntile)-1
            print(f'------There is a problem in if, add combinations, max resolution selected------{numx},{numy}')

        self.ntiles_y = numy

        numx_start, numx_stop, list_proc, start_indexs, stop_index = self.manage_process(numx, numy)

        if state == 0:
            return numx_start, numx_stop, list_proc, start_indexs, stop_index, numy, self.levi
        elif state == 1:
            return self.generator
        elif state == 2:
            return numx
        else:
            self.start_thread(numx_start, numx_stop, list_proc, start_indexs)

    def folder_manage(self, name_process):
        """Test if the folder alredy exist, if true return 1 and the thread will stop"""

        fold = os.listdir(self.path_folder)
        flag = 0
        for k in fold:
            if k == name_process:
                print(f'Folder alredy exist {name_process}')
                flag += 1
            else:
                pass

        if flag > 0:
            return True
        else:
            return False

    def process_create_dataset(self, numx, fname, save_p):
        """Divide the wsi in tiles, thanks to get_tile, if the test with fold managere is false."""
        start = 0

        for x in range(0, numx):
            for y in range(0, self.ntiles_y):
                im = self.generator.get_tile(self.levi, (x, y))
                im.thumbnail(size=self.newshape)
                nome = save_p + '/pz_' + fname[:fname.index('.svs')] + '_tile_' + str(start) + '_' + str(x) + '_' + str(y) + '.png'
                print(nome)
                im.save(nome, 'PNG')
                start += 1
        return 'End of First Analysis'

    def process_to_start(self, n_start, n_stop, name_process, start):
        """Divide the wsi in tiles, thanks to get_tile, if the test with fold managere is false."""

        f_manager = self.folder_manage(name_process)
        if not f_manager:
            create_fold = os.path.join(self.path_folder, name_process)
            os.mkdir(create_fold)
            for x in range(n_start, n_stop):
                for y in range(0, self.ntiles_y):
                    im = self.generator.get_tile(self.levi, (x, y))
                    nome = os.path.join(create_fold, 'tile_' + str(start) + '_' + str(x) + '_' + str(y) + '.png')
                    print(nome)
                    im.save(nome, 'PNG')
                    start += 1
            return 'End of First Analysis'
        else:
            return 'End of First Analysis, exit code 1'

    def manage_process(self, numtotx, numtoty):
        """Manage the starting and ending point for the reading phase of the SVS file.
        The image is only divided on x axis, respect the number of CPU core"""

        num_train_images = numtotx*numtoty
        n_core = multiprocessing.cpu_count()

        if n_core >= numtotx:
            n_core = 1
            step_x = ceil(numtotx / n_core)
            images_per_process = numtoty*step_x
        else:
            step_x = ceil(numtotx / n_core)
            images_per_process = numtoty*step_x

        print(f'Number cores:                          {n_core:>5}')
        print(f'Total number of training images:       {num_train_images:>5}')
        print(f'Number of training images for process: {images_per_process:>5}')
        print(f'Step on x for 1 process:               {step_x:>5}')

        start_index, end_index, numx_start, numx_stop, list_proc = [], [], [], [], []

        for num_pro in range(1, n_core + 1):
            #print('.......................{}'.format(int(num_pro - 1)*step_x))
            if int(num_pro - 1)*step_x < numtotx:
                start_index.append(int((num_pro - 1) * images_per_process + 1))
                end_index.append(int(num_pro * images_per_process))
                numx_start.append(int((num_pro - 1) * step_x))
                numx_stop.append(int(numx_start[num_pro - 1] + step_x))
                if numx_stop[-1] > numtotx:
                    numx_stop[-1] = numtotx
                name_process = 'p_' + str(numx_start[num_pro-1]) + '_' + str(numx_stop[num_pro-1]) + '_' + str(numtoty)
                list_proc.append(name_process)
            else:
                end_index[-1] = num_train_images
                break

        print(numx_start, numx_stop, list_proc, start_index, end_index)
        return numx_start, numx_stop, list_proc, start_index, end_index

    def start_thread(self, numx_start, numx_stop, list_proc, start_indexs):
        """Start the theads, in this way the process is faster."""

        th = []
        for i in range(0, len(list_proc)):
            p = threading.Thread(target=self.process_to_start, args=(numx_start[i], numx_stop[i], list_proc[i], start_indexs[i],))
            th.append(p)
            p.start()

        for t, y in enumerate(th):
            # if t is main_thread:
            #     continue
            # logging.debug('joining %s', t.getName())
            y.join()

        return 'Finisched'


if __name__ == '__main__':

    t = time.perf_counter()
    #test1 = StartAnalysis()
    #test1.openSvs('C:/Users/piero/10002.svs')
    #test1.tile_gen()
    tete = StartAnalysis(lev_sec=0)
    tete.list_files('C:/Users/piero/Desktop/train/AC', 'C:/Users/piero/test2')
    t1 = time.perf_counter()
    s = t1-t
    print(s)


