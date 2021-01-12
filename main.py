import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, geom_point, ggtitle, theme, xlab
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
import matplotlib.markers as markers
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path


# -------------------- CONSTANTS --------------------
PATH_TO_DATA = "phantom_table.xlsx"
DEFAULT_X = 'iron concentration'
FERRITIN_TRANSFERRIN = 1
FREE_IRON = 0
IRON = 2
TITLE = "Predict R1 according to iron concentration\n"
labels = {'iron': 'iron concentration [mg/ml]', 'lipid': 'lipid amount [%]', 'type': 'lipid type',
          'interaction': 'interaction', '1-iron': 'iron 1-complement',
          '1-lipid': 'lipid 1-complement', 'interaction only': 'pure interaction'}

lipid_amount_dict = {0.0: 'c', 10.0: 'm', 17.5: 'y', 25.0: 'g'}
labels_legend_dict = {0.0: 'lipid 0.0', 10.0: 'lipid 10.0', 17.5: 'lipid 17.5', 25.0:'lipid 25.0'}

lipid_type_dict = {'PC+Chol+Fe': "+", 'Iron': "x", 'PC+Fe': "|",
                   'PC+SM+Fe': "_", 'PC+SM+Ferritin': "D", 'Ferritin': "1", 'PC+Ferritin': "2",
                   'PC+Chol+Ferritin': "3", 'BSA+Ferritin': "4", 'PC+SM+Transferrin':
                       markers.CARETLEFT, 'Transferrin': markers.CARETRIGHT}

target_measure_dict = {'R1': '[1/sec]', 'R2s': '[1/sec]', 'R2': '[1/sec]', 'MT': '[p.u.]',
                       'MTV': '[fraction]'}

# -------------------- FUNCTIONS --------------------


def view(df, lipid_name, x_lab, y):
    """
    this function plots the data
    """
    df_sub_1 = df.loc[df['type'] == lipid_name]
    g = ggplot(data=df_sub_1, mapping=aes(x='iron', y=y, group='lipid', colour='lipid')) \
        + geom_point() + geom_line()
    g = g + theme(legend_position=(0.95, 0.6)) + xlab(x_lab) + ggtitle(str(lipid_name))
    print(g)


def get_data_by_lipid_type(data, type):
    """
    return the rows in the data that the lipid type in them match the type parameter.
    The function also add specific marker and color for each lipid type and lipid amount.
    :param data: data to manipulate
    :param type: type of lipid
    :return: sliced data
    """
    # get the data of all the free iron
    if type == IRON:
        cur_data = data[data.type.str.contains('Fe') &
                        (data.type.str.contains('Ferritin|Transferrin') == False)]

    elif type == FREE_IRON:
        cur_data = data[data.type.str.contains('Fe|Iron') &
                        (data.type.str.contains('Ferritin|Transferrin') == False)]

    # return the data containing Transferrin or Ferritin
    else:
        cur_data = data[data.type.str.contains('Ferritin|Transferrin')]

    return cur_data


def get_sliced_data(data, columns):
    """
    the function returns subset of the data according to the columns received as input
    :param data: original data
    :param columns: list of columns
    :return: sliced dataframe
    """
    return data[columns]


def predict_R1_from_iron(data, lipid=-1):
    """
    the function predicts R1 accordind to iron values.
    :param data: data containing the training set
    :param lipid: represent the lipid amount in the data.
    value -1 represent all lipids amount, i.e 10.0, 17.5, 25.0
    :return: prediction of R1 according to iron
    """
    data = get_sliced_data(data, ['iron', 'R1', 'lipid'])
    reg = LinearRegression()
    reg.fit(data[['iron']], data.R1)
    plt.xlabel('iron')
    plt.ylabel('R1')
    scatter = plt.scatter(data.iron, data.R1, c=data.lipid, marker='+')
    plt.plot(data.iron, reg.predict(data[['iron']]), color='blue')
    if lipid != -1:
        labels_legend = ['lipid ' + str(lipid)]
        title = TITLE + "lipid amount = " + str(lipid)
    else:
        labels_legend = ['lipid 0.0', 'lipid 10.0', 'lipid 17.5', 'lipid 25.0']
        title = TITLE + "lipid amount = 0.0, 10.0, 17.5, 25.0"
    score = float("{:.2f}".format(reg.score(data[['iron']], data.R1)))
    plt.title(title + '\nCoefficient of determination: ' + str(score))
    plt.legend(handles=scatter.legend_elements()[0], labels=labels_legend)
    plt.show()
    compare_prediction_to_data(reg, data, 'iron')


def compare_prediction_to_data(regression_model, data, predictor):
    """
    this function comapre between the predicted data and the measured one.
    :param regression_model: linear regression model which predict the values
    :param data: data to predict from
    :param predictor: predictor of the model
    """
    y = data.R1
    predicted = regression_model.predict(data[[str(predictor)]])

    plt.xlabel('R1 Measured')
    plt.ylabel('R1 Predicted')
    plt.scatter(y, predicted)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.title("R1 measured vs. R1 predicted")
    plt.show()


def predict_R1(data):
    lipid_amount = np.unique(np.array(data['lipid']))
    for lipid in lipid_amount:
        df = data.loc[data['lipid'] == lipid]
        predict_R1_from_iron(df, lipid)

    predict_R1_from_iron(data)


def predict_R1_from_lipid_amount(data, iron =- 1):
    """
    the function predicts R1 accordind to lipid amounts.
    :param data: data containing the training set
    :return: prediction of R1 according to lipid amount
    """
    data = get_sliced_data(data, ['iron', 'R1', 'lipid'])
    reg = LinearRegression()
    reg.fit(data[['lipid']], data.R1)
    plt.xlabel('lipid')
    plt.ylabel('R1')
    scatter = plt.scatter(data.lipid, data.R1, c=data.iron, marker='+')
    plt.plot(data.lipid, reg.predict(data[['lipid']]), color='blue')
    if iron != -1:
        labels_legend = ['iron ' + str(iron)]
        title = TITLE + "iron concentration = " + str(iron)
    else:
        labels_legend = [str(iron) for iron in np.unique(np.array(data['iron']))]
        title = TITLE + "iron concentration = " + " ".join([str(label) for label in labels])
    score = float("{:.2f}".format(reg.score(data[['lipid']], data.R1)))
    plt.title(title + '\nCoefficient of determination: ' + str(score))
    plt.legend(handles=scatter.legend_elements()[0], labels=labels_legend)
    plt.show()
    compare_prediction_to_data(reg, data, 'lipid')


def predict_R1_lipid(data):
    iron_con = np.unique(np.array(data['iron']))
    for iron in iron_con:
        df = data.loc[data['iron'] == iron]
        predict_R1_from_lipid_amount(df, iron)

    predict_R1_from_lipid_amount(data)


# cross validation
def cross_val_single_predictor(data, predictor, target):
    """
    This function returns predictor values, and target values according to existing data.
    The function should be called when pre-processing of regression with single predictor.
    """

    # define predictor and response variables
    X = np.array(data[predictor]).reshape(-1, 1)
    y = np.array(data[target])
    return X, y


def cross_val_multy_predictors(data, vars, target, interaction=False, interaction_only=False):
    """
    This function returns predictors values, and target values according to existing data.
    The function should be called when pre-processing of regression with several predictors.
    """

    if interaction:
        # create interaction column
        interaction_col = np.array(data[vars[0]] * data[vars[1]]).reshape(-1, 1)
        X = np.array(data[vars[:-1]]).reshape(-1, len(vars) - 1)
        X = np.hstack((X, interaction_col))
        y = np.array(data[target])
    else:
        if interaction_only:

            interaction_col = np.array(data[vars[0]] * data[vars[1]]).reshape(-1, 1)
            X = interaction_col
            y = np.array(data[target])

        else:
            features = vars
            # define predictor and response variables
            X = np.array(data[features]).reshape(-1, len(features))
            y = np.array(data[target])

    return X, y


def cross_val_prediction_helper(data, vars, target):
    """
    This function performs cross validation using 'leave one out' method.
    The function predicts the target variable using one or several predictors.
    """

    # several predictors
    if len(vars) > 1:
        if vars[-1] == 'interaction':
            X, y = cross_val_multy_predictors(data, vars, target, True)
        elif vars[-1] == 'interaction only':
            data = data.loc[data['iron'] != 0.00]
            data = data.loc[data['lipid'] != 0.0]
            X, y = cross_val_multy_predictors(data, vars, target, False, True)
        else:
            X, y = cross_val_multy_predictors(data, vars, target)
    # only one predictor
    else:
        X, y = cross_val_single_predictor(data, vars, target)

    colors = [lipid_amount_dict[lipid] for lipid in data.lipid]
    markers = [lipid_type_dict[type] for type in data.type]

    markers_labels = []
    types = []
    for type in lipid_type_dict:
        if type not in types:
            types.append(type)
            markers_labels.append(lipid_type_dict[type])

    # define cross-validation method to use
    cv = LeaveOneOut()
    # build multiple linear regression model
    model = LinearRegression()

    # use LOOCV to evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    predictions = cross_val_predict(model, X, y, cv=cv)
    marker_type_dict = {}
    for type, marker in lipid_type_dict.items():
        marker_type_dict[marker] = type

    color_type_dict = {}
    for lipid, color in lipid_amount_dict.items():
        color_type_dict[color] = lipid

    plt.figure()
    for i in range(len(y)):
        plt.scatter(y[i], predictions[i], c=colors[i], marker=markers[i])
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

    labels_legend_color = list()
    for lipid in lipid_amount_dict:
        labels_legend_color.append(mpatches.Patch(color=lipid_amount_dict[lipid],
                                                  label=str(labels_legend_dict[lipid])))

    plt.legend(handles=labels_legend_color)
    # print(markers_labels)
    # print(types)
    # plt.gca().add_artist(markers_labels, types)

    #
    # legend_elements = []
    # for marker in set_markers:
    #     legend_elements.append(Line2D([0], [0], marker=marker, label=marker_type_dict[marker]))

    # plt.legend([label for label in set(labels)])

    accuracy = r2_score(y, predictions)
    # the lower the MAE, the more closely a model is able to predict the actual observations.
    mae = mean(absolute(scores))
    predictors = ""
    for pred in vars:
        predictors += labels[pred] + ", "
    predictors = predictors[:len(predictors)-2]
    plt.title(str(target) + " measured vs. " + str(target) + " predicted\n"
              "predictors: " + predictors + "\n"
              "R^2: " + str(float("{:.2f}".format(accuracy))) +
              " Mean absolute squared error: " + str(float("{:.2f}".format(mae))))
    plt.xlabel(str(target) + target_measure_dict[target] + " measured")
    plt.ylabel(str(target) + target_measure_dict[target] + " predicted")
    fig_name = str(target) + "_" + str(predictors)
    plt.savefig(fname=fig_name, format='png')
    plt.show()



def cross_val_prediction(data):
    predictors = [['iron'], ['lipid'], ['iron', 'lipid'], ['iron', 'lipid', 'interaction'],
                  ['iron', 'lipid', 'interaction only']]
    targets = ['R1', 'R2', 'R2s', 'MT', 'MTV']
    for predictor in predictors:
        for target in targets:
            cross_val_prediction_helper(data, predictor, target)


if __name__ == '__main__':
    # pre-processing of the data
    df = pd.read_excel(PATH_TO_DATA)

    data_ferritin_transferrin = get_data_by_lipid_type(df, FERRITIN_TRANSFERRIN)
    data_iron_without_free = get_data_by_lipid_type(df, IRON)
    data_iron = get_data_by_lipid_type(df, FREE_IRON)

    # predicting
    cross_val_prediction(data_iron)
