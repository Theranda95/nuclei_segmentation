import pandas as pd
import os
from natsort import os_sorted

def main():
    data_dir = 'replicate2_nuclei_stacks/roi_measurements'
    csv_files = os_sorted(os.listdir(data_dir))
    dfs = []
    cols = ['genotype', 'image', 'Area', 'Circ.', 'Solidity', 'Round']
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        df['genotype'] = csv_file.split("_")[0]
        df['image'] = csv_file.split("_")[1]
        df = df[cols]
        dfs.append(df)

    df_all = pd.concat(dfs, axis=0)
    df_all.to_csv(os.path.join(data_dir, 'nuclei_desc.csv'), index=None)
    print('eof')


if __name__ == '__main__':
    main()
