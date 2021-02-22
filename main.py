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
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

# ------------ subclass for markers legend ----------------

class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup, xdescent, ydescent, width, height, fontsize, trans):
        return [plt.Line2D([width/4], [height/4.], ls="", marker=tup[1], color='b',
                           transform=trans)]

# -------------------- CONSTANTS --------------------
PATH_TO_DATA = "phantom_table.xlsx"
DEFAULT_X = 'iron concentration [mg/ml]'


NO_FER_TRANS = 0
IRON = '[Fe] sigma [mg/ml]'
LIPID = 'lipid [%]'
R1 = 'R1 [1/sec]'
R2 = 'R2 [1/sec]'
R2S = 'R2s [1/sec]'
MTV = 'MTV [fraction]'
MT = 'MT [p.u.]'

TITLE = "Predict R1 according to iron concentration\n"
labels = {IRON: 'iron concentration', LIPID: 'lipid amount', 'type': 'lipid type',
          'interaction': 'iron concentration * lipid amount', '1-iron': 'iron 1-complement',
          '1-lipid': 'lipid 1-complement', 'interaction only': 'pure interaction'}

lipid_amount_dict = {0.0: 'c', 10.0: 'm', 17.5: 'y', 25.0: 'g'}
labels_legend_dict = {0.0: 'lipid 0.0', 10.0: 'lipid 10.0', 17.5: 'lipid 17.5', 25.0:'lipid 25.0'}

lipid_type_dict = {'BSA+Ferritin': "+", 'Fe2': "x", 'Fe3': "o",
                   'PC+Chol+Fe2': "_", 'PC+Chol+Fe3': "D", 'Ferritin': "1", 'PC+Chol+Ferritin': "2",
                   'PC+Fe2': "3", 'PC+Ferritin': "4", 'PC+SM+Fe2':
                       markers.CARETLEFT, 'Transferrin': markers.CARETRIGHT,
                   'PC+SM+Fe3': markers.CARETDOWN, 'PC+SM+Ferritin': markers.CARETUP,
                   'PC+SM+Transferrin': markers.TICKDOWN}

target_measure_dict = {'R1': '[1/sec]', 'R2s': '[1/sec]', 'R2': '[1/sec]', 'MT': '[p.u.]',
                       'MTV': '[fraction]'}

for_title_dict = {R1: 'R1', R2: 'R2', R2S: 'R2s', MT: 'MT', MTV: 'MTV'}

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
    # data without ferritin and transfferin
    if type == NO_FER_TRANS:
        cur_data = data[data.type.str.contains('Ferritin|Transferrin') == False]

    # data of all lipid types without free iron
    elif type == IRON:
        cur_data = data[data.type.str.contains('Iron') == False]

    # get data of free iron
    else:
        cur_data = data[data.type.str.contains('Iron')]

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


# -------------------------- cross validation ----------------------------

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
            data = data.loc[data[IRON] != 0.00]
            data = data.loc[data[LIPID] != 0.0]
            X, y = cross_val_multy_predictors(data, vars, target, False, True)
        else:
            X, y = cross_val_multy_predictors(data, vars, target)
    # only one predictor
    else:
        X, y = cross_val_single_predictor(data, vars, target)

    colors, markers, markers_types, markers_labels = get_colors_and_markers(data)

    # define cross-validation method to use
    cv = LeaveOneOut()
    # build multiple linear regression model
    model = LinearRegression()

    # use LOOCV to evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    predictions = cross_val_predict(model, X, y, cv=cv)

    plt.figure()
    for i in range(len(y)):
        plt.scatter(y[i], predictions[i], c=colors[i], marker=markers[i])
    plt.plot([y.min(), y.max()], [predictions.min(), predictions.max()], 'k--', lw=2)

    labels_legend_color = list()
    # for lipid in lipid_amount_dict:
    #   labels_legend_color.append(mpatches.Patch(color=lipid_amount_dict[lipid],
    #                                             label=str(labels_legend_dict[lipid])))
    #
    # legend1 = plt.legend(list(zip(colors, markers_types)), markers_labels,
    #         handler_map={tuple: MarkerHandler()}, loc=4)
    # plt.legend(handles=labels_legend_color, loc=2)
    # plt.gca().add_artist(legend1)

    accuracy = r2_score(y, predictions)
    # the lower the MAE, the more closely a model is able to predict the actual observations.
    mae = mean(absolute(scores))
    coeff = ['a', 'b', 'c', 'd']
    predictors = ""
    if 'interaction only' in vars:
        predictors = 'a * ' + labels['interaction']
    else:
        for i in range(0, len(vars)):
            if coeff[i] == 'c':
                predictors += '\n' + str(coeff[i]) + ' * ' + str(labels[vars[i]]) + ' + '
            else:
                predictors += str(coeff[i]) + ' * ' + str(labels[vars[i]]) + ' + '
        predictors = predictors[:len(predictors)-2]
    plt.title(str(for_title_dict[target]) + ' = ' + predictors + "\n"
              "R^2: " + str(float("{:.3f}".format(accuracy))) +
              " Mean absolute squared error: " + str(float("{:.3f}".format(mae))))
    plt.xlabel(str(target) + " measured")
    plt.ylabel(str(target) + " predicted")
    plt.xlim([0.05, 0.3])
    plt.ylim([0.225, 0.23])
    plt.show()


def cross_val_prediction(data):
    predictors = [[IRON], [LIPID],  [IRON, LIPID], [IRON, LIPID, 'interaction'], [IRON, LIPID, 'interaction only']]
    targets = [R1, R2, R2S, MT, MTV]
    for target in targets:
        for predictor in predictors:
            cross_val_prediction_helper(data, predictor, target)


def pre_processing():
    data = pd.read_excel(PATH_TO_DATA)
    # ignore experiments 6 and 11
    data = data[data.ExpNum != 6]
    data = data[data.ExpNum != 11]
    # ignore experiments where the lipid type is BSA+Ferritin
    return data[data.type != 'BSA+Ferritin']


def get_colors_and_markers(data):
    colors = [lipid_amount_dict[lipid] for lipid in data[LIPID]]
    markers = [lipid_type_dict[type] for type in data.type]

    markers_types = []
    markers_labels = []
    for type in np.unique(data.type):
        if type not in markers_labels:
            markers_labels.append(type)
            markers_types.append(lipid_type_dict[type])
    return colors, markers, markers_types, markers_labels


def parameter_fe_plot(data):
    colors, markers, markers_types, markers_labels = get_colors_and_markers(data)
    qmri_parameters = np.asarray(data.columns[2:7])
    x_axis = data['lipid [%]']
    for parameter in qmri_parameters:
        y_axis = data[parameter]
        fig, ax = plt.subplots()
        for xp, yp, c, m in zip(x_axis, y_axis, colors, markers):
            ax.scatter([xp], [yp], c=c, marker=m)

        plt.ylabel('{0}'.format(str(parameter)))
        plt.xlabel('lipid [%]')
        labels_legend_color = list()
        for lipid in lipid_amount_dict:
            labels_legend_color.append(mpatches.Patch(color=lipid_amount_dict[lipid],
                                                      label=str(labels_legend_dict[lipid])))
        plt.legend(handles=labels_legend_color, loc='best')

        plt.show()


if __name__ == '__main__':
    # pre-processing of the data
    df = pre_processing()
    parameter_fe_plot(df)
    # data = get_data_by_lipid_type(df, 0)
    # predicting
    # cross_val_prediction(df)

