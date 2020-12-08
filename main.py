import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, geom_point, ggtitle, scale_color_manual, \
    scale_color_grey, theme_classic, theme, guides, guide_legend, scale_fill_manual, labs, xlab, \
    facet_wrap, ggsave

PATH_TO_DATA = "phantom_table.xlsx"
DEFAULT_X = 'iron concentration'


def read_data():
    df = pd.read_excel(PATH_TO_DATA)
    return df


def view(df, lipid_name, x_lab, y):
    df_sub_1 = df.loc[df['type'] == lipid_name]
    g = ggplot(data=df_sub_1, mapping=aes(x='iron', y=y, group='lipid', colour='lipid'))\
        + geom_point() + geom_line()
    g = g + theme(legend_position=(0.95, 0.6)) + xlab(x_lab) + ggtitle(str(lipid_name))
    
    # file_name = lipid_name + "_" + y + ".png"
    # ggsave(plot=g, filename=file_name, path="figure")
    
    print(g)


if __name__ == '__main__':
    df = read_data()
    df.lipid = df.lipid.astype(str)
    lipid_type = np.unique(np.array(df['type']))
    y_var = ['R1', 'R2', 'R2s', 'MT']
    for y in y_var:
        for lipid in lipid_type:
            if 'Ferritin' in lipid:
                view(df, lipid, 'Ferritin', y)
            elif 'Transferrin' in lipid:
                view(df, lipid, 'Transferrin', y)
            else:
                view(df, lipid, DEFAULT_X, y)



