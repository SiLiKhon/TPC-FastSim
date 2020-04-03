import os
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_PATH = Path(os.path.realpath(__file__)).parent
_VERSION = 'data_v1'

class Reader:
    def __init__(self, variables, types):
        assert len(variables) == len(types)
        self.vars = variables
        self.types = types
        self.data = []

    def read_line(self, line, index):
        stems = line.split()
        assert (len(stems) // len(self.vars)) * len(self.vars) == len(stems), [line, self.vars]

        for i_group in range(0, len(stems), len(self.vars)):
            self.data.append((index,) + tuple(_T(stems[i_group + i_var]) for i_var, _T in enumerate(self.types)))

    def build(self):
        return pd.DataFrame(self.data, columns=['evtId'] + self.vars).set_index('evtId')



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

    reader_main = Reader(
        variables = ['ipad', 'itime', 'amp'],
        types    = [int   , int    , float]
    )

    data_sources = [lines]
    readers = [reader_main]

    if 'params:' in lines[0]:
        assert len(lines) % 2 == 0

        reader_angles = Reader(
            variables = ["crossing_angle", "dip_angle"],
            types    = [float           , float      ]
        )
        lines, lines_angles = lines[1::2], lines[::2]
        lines_angles = [' '.join(l.split()[1:]) for l in lines_angles]

        data_sources = [lines, lines_angles]
        readers = [reader_main, reader_angles]

    for evt_id, lines_tuple in enumerate(zip(*data_sources)):
        for r, l in zip(readers, lines_tuple):
            r.read_line(l, evt_id)
            
    result = pd.concat([r.build() for r in readers], axis=1).reset_index()
    result.to_csv(fname_out, index=False)

def read_csv_2d(filename=None, pad_range=(40, 50), time_range=(265, 280)):
    if filename is None:
        filename = str(_THIS_PATH.joinpath(_VERSION, 'csv', 'digits.csv'))
    
    df = pd.read_csv(filename)

    sel = lambda df, col, limits: (df[col] >= limits[0]) & (df[col] < limits[1])

    selection = (
        sel(df, 'itime', time_range) &
        sel(df, 'ipad' , pad_range )
    )

    if not selection.all():
        print(f"WARNING: current selection ignores {(~selection).sum() / len(selection) * 100}% of the data!")

    g = df[selection].groupby('evtId')

    def convert_event(event):
        result = np.zeros(dtype=float, shape=(pad_range [1] - pad_range [0],
                                              time_range[1] - time_range[0]))
    
        indices = tuple(event[['ipad', 'itime']].values.T - np.array([[pad_range[0]], [time_range[0]]]))
        result[indices] = event.amp.values

        return result

    data = np.stack(g.apply(convert_event).values)

    if 'crossing_angle' in df.columns:
        assert (g[['crossing_angle', 'dip_angle']].std() == 0).all().all()
        return data, g[['crossing_angle', 'dip_angle']].mean().values

    return data
