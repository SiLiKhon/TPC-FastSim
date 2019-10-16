import os
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_PATH = Path(os.path.realpath(__file__)).parent

def raw_to_csv(fname_in=None, fname_out=None):
    if fname_in is None:
        fname_in = str(_THIS_PATH.joinpath('data_v0', 'raw', 'digits.dat'))
    if fname_out is None:
        csv_path = _THIS_PATH.joinpath('data_v0', 'csv')
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
