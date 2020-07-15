from pathlib import Path
from datetime import datetime

segmentation_data_path = Path('data/supervisely_segmentation/').expanduser()

def get_segmentation_model_path(size):
    return '../weights/unet_with_wh_{size}.hdf5'.format(size=size)

def load_segmentation_data(imdir, maskdir):
    image_paths = imdir.glob("*.*")
    data = []
    for im_path in image_paths:
        mask_path = maskdir/(im_path.stem + ".png")
        if mask_path.exists():
            data.append((im_path, mask_path))
    return data



def split(l, ratio):
    s = int(len(l) * ratio)
    return l[:s], l[s:]


class Run:
    def __init__(self):
        self.timetag = datetime.now().strftime("%Y%m%d-%H%M%S")

    def get_run_dir(self, tag):
        return "runs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/{tag}/"
