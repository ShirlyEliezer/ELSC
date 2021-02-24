# ----------------------- imports -------------------------
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import matplotlib.markers as markers
import numpy as np


# -------------------- CONSTANTS --------------------
PATH_TO_DATA = "phantom_table.xlsx"
DEFAULT_X = 'iron concentration [mg/ml]'


NO_FER_TRANS = 0
IRON = '[Fe] sigma [mg/ml]'
LIPID = 'lipid [%]'
R1 = 'R1 [1/sec]'
R2 = 'R2 [1/sec]'
R2S = 'R2s [1/sec]'
MTV = 'MTV [fraction]'
MT = 'MT [p.u.]'

TITLE = "Predict R1 according to iron concentration\n"
labels = {IRON: '[iron]', LIPID: '[lipid]', 'type': 'lipid type', 'interaction': '[iron] * [lipid]'}

lipid_amount_dict = {0.0: 'c', 10.0: 'm', 17.5: 'y', 25.0: 'g'}
labels_legend_dict = {0.0: 'lipid 0.0', 10.0: 'lipid 10.0', 17.5: 'lipid 17.5', 25.0: 'lipid 25.0'}

lipid_type_dict = {'BSA+Ferritin': "+", 'Fe2': "x", 'Fe3': "o",
                   'PC+Chol+Fe2': "_", 'PC+Chol+Fe3': "D", 'Ferritin': "1", 'PC+Chol+Ferritin': "2",
                   'PC+Fe2': "3", 'PC+Ferritin': "4", 'PC+SM+Fe2':
                       markers.CARETLEFT, 'Transferrin': markers.CARETRIGHT,
                   'PC+SM+Fe3': markers.CARETDOWN, 'PC+SM+Ferritin': markers.CARETUP,
                   'PC+SM+Transferrin': markers.TICKDOWN}

target_measure_dict = {'R1': '[1/sec]', 'R2s': '[1/sec]', 'R2': '[1/sec]', 'MT': '[p.u.]',
                       'MTV': '[fraction]'}
for_title_dict = {R1: 'R1', R2: 'R2', R2S: 'R2s', MT: 'MT', MTV: 'MTV'}
bad_samples = [6, 11]
coefficients = ['a', 'b', 'c', 'd']


# ------------ subclass for markers legend ----------------
class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup, xdescent, ydescent, width, height, fontsize, trans):
        return [plt.Line2D([width/4], [height/4.], ls="", marker=tup[1], color='b',
                           transform=trans)]


def get_colors_and_markers(data):
    colors = [lipid_amount_dict[lipid] for lipid in data[LIPID]]
    markers = [lipid_type_dict[type] for type in data.type]

    markers_types = []
    markers_labels = []
    for type in np.unique(data.type):
        if type not in markers_labels:
            markers_labels.append(type)
            markers_types.append(lipid_type_dict[type])
    return colors, markers, markers_types, markers_labels
