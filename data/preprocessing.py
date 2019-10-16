import os
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_PATH = Path(os.path.realpath(__file__)).parent
_VERSION = 'data_v0'

def raw_to_csv(fname_in=None, fname_out=None):
    if fname_in is None:
        fname_in = str(_THIS_PATH.joinpath(_VERSION, 'raw', 'digits.dat'))
    if fname_out is None:
        csv_path = _THIS_PATH.joinpath(_VERSION, 'csv')
        if not os.path.isdir(csv_path):
            csv_path.mkdir()
        fname_out = str(csv_path.joinpath('digits.csv'))

    with open(fname_in, 'r') as f:
        lines = f.readlines()

    entries = []

    for evt_id, line in enumerate(lines):
        stems = line.split()
        assert (len(stems) // 3) * 3 == len(stems)

        for j in range(0, len(stems), 3):
            #                       ## ipad        ## itime           ## amplitude
            entries.append((evt_id, int(stems[j]), int(stems[j + 1]), float(stems[j + 2])))
    
    data = pd.DataFrame(entries, columns=['evtId', 'ipad', 'itime', 'amp'])
    
    data.to_csv(fname_out, index=False)

def read_csv_2d(filename=None, pad_range=(40, 50), time_range=(265, 280)):
    if filename is None:
        filename = str(_THIS_PATH.joinpath(_VERSION, 'csv', 'digits.csv'))
    
    df = pd.read_csv(filename)

    sel = lambda df, col, limits: (df[col] >= limits[0]) & (df[col] < limits[1])

    g = df[sel(df, 'itime', time_range) & 
           sel(df, 'ipad' , pad_range )].groupby('evtId')

    def convert_event(event):
        result = np.zeros(dtype=float, shape=(pad_range [1] - pad_range [0],
                                              time_range[1] - time_range[0]))
    
        indices = tuple(event[['ipad', 'itime']].values.T - np.array([[pad_range[0]], [time_range[0]]]))
        result[indices] = event.amp.values

        return result

    data = np.stack(g.apply(convert_event).values)
    return data