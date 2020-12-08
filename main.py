import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, geom_point, ggtitle, scale_color_manual, \
    scale_color_grey, theme_classic, theme, guides, guide_legend, scale_fill_manual, labs, xlab, \
    facet_wrap, ggsave
from sklearn import linear_model

# -------------------- CONSTANTS --------------------
PATH_TO_DATA = "phantom_table.xlsx"
DEFAULT_X = 'iron concentration'
FERRITIN_TRANSFERRIN = 1
FREE_IRON = 0


# -------------------- FUNCTIONS --------------------


def read_data():
    df = pd.read_excel(PATH_TO_DATA)
    return df


def view(df, lipid_name, x_lab, y):
    df_sub_1 = df.loc[df['type'] == lipid_name]
    g = ggplot(data=df_sub_1, mapping=aes(x='iron', y=y, group='lipid', colour='lipid')) \
        + geom_point() + geom_line()
    g = g + theme(legend_position=(0.95, 0.6)) + xlab(x_lab) + ggtitle(str(lipid_name))

    # file_name = lipid_name + "_" + y + ".png"
    # ggsave(plot=g, filename=file_name, path="figure")

    print(g)


def view_graphs(data):
    data.lipid = data.lipid.astype(str)
    lipid_type = np.unique(np.array(data['type']))
    y_var = ['R1', 'R2', 'R2s', 'MT']
    for y in y_var:
        for lipid in lipid_type:
            if 'Ferritin' in lipid:
                view(data, lipid, 'Ferritin', y)
            elif 'Transferrin' in lipid:
                view(data, lipid, 'Transferrin', y)
            else:
                view(data, lipid, DEFAULT_X, y)


def get_data_by_iron_type(data, type):
    # get the data of all the free iron
    if type == FREE_IRON:
        return data[data.type.str.contains('Fe|Iron') & (data.type.str.contains('Ferritin|Transferrin') == False)]
    # return the data containing Transferrin or Ferritin
    return data[data.type.str.contains('Ferritin|Transferrin')]


def predict_R1(data):
    """
    the function predicts R1 accordind to iron values.
    :param data: data containing the training set
    :return: prediction of R1 according to iron
    """
    # plot the data before prediction
    plt.xlabel('iron')
    plt.ylabel('R1')
    plt.scatter(data.iron, data.R1, color='red', marker='+')
    plt.show()


def get_sliced_data(data, columns):
    """
    the function returns subset of the data according to the columns received as input
    :param data: original data
    :param columns: list of columns
    :return: sliced dataframe
    """
    return data[columns]


if __name__ == '__main__':
    df = read_data()
    data = get_data_by_iron_type(df, FERRITIN_TRANSFERRIN)
    data = data[data['lipid'] == 10.0]
    print(data)
    data = get_sliced_data(data, ['iron', 'R1'])
    predict_R1(data)
