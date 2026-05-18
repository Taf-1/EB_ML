import os 
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from astropy.table import Table, vstack

class Stack_LCS:
    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = input_dir
        self.output_file = output_file

    def stack_light_curves(self):
        all_files = [f for f in os.listdir(self.input_dir) if f.endswith('.fits')]
        stacked_table = None

        for file in tqdm(all_files, desc="Stacking light curves"):
            file_path = os.path.join(self.input_dir, file)
            with fits.open(file_path) as hdul:
                data = hdul[1].data
                table = Table(data)

                if stacked_table is None:
                    stacked_table = table
                else:
                    stacked_table = vstack([stacked_table, table])

        self.logger.info(f"Stacking completed. Saving results to {self.output_file}")
        stacked_table.write(self.output_file, format='fits', overwrite=True)