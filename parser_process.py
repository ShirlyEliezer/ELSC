"""The purpose of the document is to prepare the data for forecasting and presentation"""
# ----------------------- constants -----------------------
import pandas as pd
from ToolBox import *


# ----------------------- functions -----------------------
def get_data_without_ferritin_or_transferrin(data):
    """
    return samples of lipids without Ferritin or Transferrin
    :param data: data to manipulate
    :return: sliced data
    """
    return data[data.type.str.contains('Ferritin|Transferrin') == False]


def get_sliced_data(data, columns):
    """
    the function returns subset of the data according to the columns received as input
    :param data: original data
    :param columns: list of columns
    :return: sliced dataframe
    """
    return data[columns]


def pre_processing():
    data = pd.read_excel(PATH_TO_DATA)
    # ignore experiments 6 and 11
    for bad_sample in bad_samples:
        data = data[data.ExpNum != int(bad_sample)]

    # ignore experiments where the lipid type is BSA+Ferritin
    data = get_data_without_ferritin_or_transferrin(data)
    data = data[data.type.str.contains('Fe3') == False]
    # remove data with free iron
    data = data[data.type != 'Fe2']
    return data[data.type != 'BSA+Ferritin']

