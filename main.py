import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_DATA = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Iron_Lipids/phantom_table.xlsx"


def read_data():
    df = pd.read_excel(PATH_TO_DATA)
    return df


def view(df, lipid_type):
    df_sub_1 = df.loc[df['type'] == lipid_type]
    df_sub_1_lipid = df_sub_1['lipid'].unique()
    fig, ax = plt.subplots()

    for lipid_con in df_sub_1_lipid:
        df_lipid = df_sub_1.loc[df_sub_1['lipid'] == lipid_con]
        df_lipid = pd.DataFrame(df_lipid)
        ax.scatter(x=df_lipid['iron'], y=df_lipid['R1'], label='lipid = ' + str(lipid_con))

    plt.legend()
    plt.xlabel("iron concentration")
    plt.ylabel("R1")
    plt.title("type = " + lipid_type)
    plt.show()


if __name__ == '__main__':
    df = read_data()
    lipid_type = np.unique(np.array(df['type']))
    for lipid in lipid_type:
        view(df, lipid)
        print("yay")