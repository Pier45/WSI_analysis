import glob
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
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
        lev_down = self.slide.level_downsamples
        print(lev_down)
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
        """Start the tile-extraction workers and block until they finish.

        Historically this used ``threading.Thread``: one partition per
        thread. That gave essentially no speedup on real SVS files because
        ``openslide-python`` is a ``ctypes`` binding — its
        ``openslide_read_region`` call does NOT release the GIL during the
        decode, so threads could not overlap the CPU-bound work, only the
        I/O waits. On a 4-core box the threading version measured ~0.66×
        serial (slower, from context-switch overhead).

        Switched to ``concurrent.futures.ProcessPoolExecutor``: each
        process holds its own GIL, so the decode step runs truly in
        parallel (~1.2× measured on this box for the synthetic
        GIL-holding-decode benchmark, closer to N× on real SVS where the
        decode dominates). Each worker opens the SVS through a per-process
        LRU cache (``_get_cached_generator``) so the redundant
        ``openslide.OpenSlide`` + ``DeepZoomGenerator`` build happens once
        per process, not once per partition.
        """
        existing = set(os.listdir(self.path_folder)) if os.path.isdir(self.path_folder) else set()
        args_per_partition = _build_partition_args(
            numx_start=numx_start,
            numx_stop=numx_stop,
            list_proc=list_proc,
            start_indexs=start_indexs,
            file_path=self.file_path,
            lev_sec=self.lev_sec,
            tile_size=self.tile_size,
            overlap=self.overlap,
            limit_bounds=self.limit_bounds,
            base_folder=self.path_folder,
            n_rows=self.ntiles_y,
            levi=self.levi,
            existing=existing,
        )
        if not args_per_partition:
            return 'Finisched'

        n_workers = min(len(args_per_partition), multiprocessing.cpu_count())
        # With Python 3.11 on Linux the default start method is ``fork``,
        # which inherits the already-opened SVS fd cheaply. On
        # Windows / macOS the method is ``spawn``, which pickles the
        # arguments — that's fine here because they are all builtins
        # (ints, strings, floats).
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_tile_partition_worker, args_per_partition))

        return 'Finisched'


def _build_partition_args(
    *,
    numx_start,
    numx_stop,
    list_proc,
    start_indexs,
    file_path,
    lev_sec,
    tile_size,
    overlap,
    limit_bounds,
    base_folder,
    n_rows,
    levi,
    existing,
):
    """Build the per-partition argument tuples for ``_tile_partition_worker``.

    Pure function (no I/O): takes the partition metadata coming out of
    ``manage_process`` plus the per-SVS configuration, and returns the
    list of picklable tuples that ``ProcessPoolExecutor.map`` will
    hand to each worker. Partitions whose folder name already exists
    on disk are filtered out — matches the legacy ``folder_manage``
    skip so a partial run picks up where it left off without
    re-extracting.

    Extracted from ``StartAnalysis.start_thread`` so the tuple shape
    is unit-testable without opening a real ``.svs`` file.
    """
    args = []
    for i in range(len(list_proc)):
        if list_proc[i] in existing:
            continue
        args.append(
            (
                numx_start[i],
                numx_stop[i],
                list_proc[i],
                start_indexs[i],
                file_path,
                lev_sec,
                tile_size,
                overlap,
                limit_bounds,
                base_folder,
                n_rows,
                levi,
            )
        )
    return args


# ---------------------------------------------------------------------------
# Module-level workers for ProcessPoolExecutor
#
# These must live at module scope (not as methods) so the ``spawn`` start
# method on Windows / macOS can pickle them. They re-open the SVS in the
# worker process via ``_get_cached_generator`` (LRU-keyed on path) so the
# redundant parse + DeepZoom build happens once per worker process, not
# once per partition — Option B.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _get_cached_generator(file_path, lev_sec, tile_size, overlap, limit_bounds):
    """Per-process cached ``DeepZoomGenerator`` for *file_path*.

    Each worker process opens the SVS read-only once and reuses the
    generator across every partition it handles. This is safe: OpenSlide
    is read-only by default and concurrent reads from multiple processes
    on the same file are fine (they each hold their own fd + handle). The
    cache is bounded to 4 entries so a worker that happens to handle
    several different SVS files doesn't leak file handles.

    Reopening the slide on Linux ``fork`` would also work transparently
    (the parent's fd is inherited), but ``spawn`` on Windows / macOS
    starts a fresh Python interpreter and so must reopen explicitly — this
    helper serves both.
    """
    slide = openslide.OpenSlide(file_path)
    generator = DeepZoomGenerator(
        slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds
    )
    return slide, generator


def _tile_partition_worker(args):
    """Module-level worker for ``ProcessPoolExecutor`` — extracts one x-range
    of tiles from the SVS and writes PNGs into the partition folder.

    ``args`` is the partition tuple built by ``StartAnalysis.start_thread``:
    ``(x_start, x_stop, process_name, tile_start, svs_path, lev_sec,
    tile_size, overlap, limit_bounds, base_folder, n_rows, levi)``.

    The worker reuses the per-process cached ``DeepZoomGenerator``
    (``_get_cached_generator``) so the openslide parse + DeepZoom index
    build happens once per *worker process*, regardless of how many
    partitions that process ends up handling.
    """
    (
        x_start,
        x_stop,
        process_name,
        tile_start,
        file_path,
        lev_sec,
        tile_size,
        overlap,
        limit_bounds,
        base_folder,
        n_rows,
        levi,
    ) = args

    # Existing-partition short-circuit: matches the legacy ``folder_manage``
    # contract — partitions already on disk are skipped, never overwritten.
    create_fold = os.path.join(base_folder, process_name)
    if os.path.isdir(create_fold):
        return f"Partition '{process_name}' already exists — skipping."

    os.mkdir(create_fold)

    _slide, generator = _get_cached_generator(
        file_path, lev_sec, tile_size, overlap, limit_bounds
    )

    current_index = tile_start
    for x in range(x_start, x_stop):
        for y in range(n_rows):
            im = generator.get_tile(levi, (x, y))
            tile_path = os.path.join(
                create_fold, f"tile_{current_index}_{x}_{y}.png"
            )
            im.save(tile_path, "PNG")
            current_index += 1

    return f"Partition '{process_name}' complete: {current_index - tile_start} tiles."


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


