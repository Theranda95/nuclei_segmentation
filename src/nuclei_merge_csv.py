import pandas as pd
import os
from natsort import os_sorted


def merge_csv_files(data_dir, output_dir, nuclei_desc_name):
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
    df_all.to_csv(os.path.join(output_dir, nuclei_desc_name), index=None)
    print('eof')


def compute_means(output_dir, nuclei_desc_name, nuclei_mean_name):
    df = pd.read_csv(os.path.join(output_dir, nuclei_desc_name))
    mean_area = df.pivot_table(index=['genotype', 'image'], values='Area', aggfunc='mean')
    mean_circ = df.pivot_table(index=['genotype', 'image'], values='Circ.', aggfunc='mean')
    mean_solidity = df.pivot_table(index=['genotype', 'image'], values='Solidity', aggfunc='mean')
    mean_round = df.pivot_table(index=['genotype', 'image'], values='Round', aggfunc='mean')
    mean_all = pd.concat([mean_area, mean_circ, mean_solidity, mean_round], axis=1)
    mean_all.to_csv(os.path.join(output_dir, nuclei_mean_name))


def main():
    nuclei_desc_name = 'nuclei_desc.csv'
    nuclei_mean_name = 'nuclei_mean.csv'

    # -------------------
    # replicate2
    # -------------------
    data_dir = 'replicate2_nuclei_stacks/roi_measurements'
    output_dir = os.path.dirname(data_dir)
    # merge_csv_files(data_dir, output_dir, nuclei_desc_name)
    # compute_means(output_dir, nuclei_desc_name, nuclei_mean_name)

    # -------------------
    # replicate1
    # -------------------
    data_dir = 'replicate1_nuclei_stacks/roi_measurements'
    output_dir = os.path.dirname(data_dir)
    merge_csv_files(data_dir, output_dir, nuclei_desc_name)
    compute_means(output_dir, nuclei_desc_name, nuclei_mean_name)


if __name__ == '__main__':
    main()
