import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, geom_point, ggtitle, theme, xlab
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt
import sklearn

# -------------------- CONSTANTS --------------------
PATH_TO_DATA = "phantom_table.xlsx"
DEFAULT_X = 'iron concentration'
FERRITIN_TRANSFERRIN = 1
FREE_IRON = 0
TITLE = "Predict R1 according to iron concentration\n"


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


def get_data_by_iron_type(data, type):
    """
    return the rows in the data that the lipid type in them match the type parameter.
    :param data: data to manipulate
    :param type: type of lipid
    :return: sliced data
    """
    # get the data of all the free iron
    if type == FREE_IRON:
        return data[data.type.str.contains('Fe|Iron') & (data.type.str.contains('Ferritin|Transferrin') == False)]
    # return the data containing Transferrin or Ferritin
    return data[data.type.str.contains('Ferritin|Transferrin')]


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
        labels = ['lipid ' + str(lipid)]
        title = TITLE + "lipid amount = " + str(lipid)
    else:
        labels = ['lipid 0.0', 'lipid 10.0', 'lipid 17.5', 'lipid 25.0']
        title = TITLE + "lipid amount = 0.0, 10.0, 17.5, 25.0"
    score = float("{:.2f}".format(reg.score(data[['iron']], data.R1)))
    plt.title(title + '\nCoefficient of determination: ' + str(score))
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
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


# cross validation
def leave_one_out(data):
    """
    This function performs cross validation using 'leave one out' method.
    :param data: data to train and test
    :return: -
    """
    # define predictor and response variables
    X = np.array(data['iron']).reshape(-1, 1)
    y = np.array(data['R1'])

    # define cross-validation method to use
    cv = LeaveOneOut()
    # build multiple linear regression model
    model = LinearRegression()

    # use LOOCV to evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    predictions = cross_val_predict(model, X, y, cv=cv)
    scatter = plt.scatter(y, predictions, c=data.lipid, marker='+')
    accuracy = r2_score(y, predictions)
    plt.title("Accuracy: " + str(accuracy))
    plt.show()
    # the lower the MAE, the more closely a model is able to predict the actual observations.
    mse = mean(absolute(scores))
    print(mse)


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
        labels = ['iron ' + str(iron)]
        title = TITLE + "iron concentration = " + str(iron)
    else:
        labels = [str(iron) for iron in np.unique(np.array(data['iron']))]
        title = TITLE + "iron concentration = " + " ".join([str(label) for label in labels])
    score = float("{:.2f}".format(reg.score(data[['lipid']], data.R1)))
    plt.title(title + '\nCoefficient of determination: ' + str(score))
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.show()
    compare_prediction_to_data(reg, data, 'lipid')


def predict_R1_lipid(data):
    iron_con = np.unique(np.array(data['iron']))
    for iron in iron_con:
        df = data.loc[data['iron'] == iron]
        predict_R1_from_lipid_amount(df, iron)

    predict_R1_from_lipid_amount(data)


if __name__ == '__main__':
    # pre-processing of the data
    df = pd.read_excel(PATH_TO_DATA)
    data_ferritin_transferrin = get_data_by_iron_type(df, FERRITIN_TRANSFERRIN)
    data_iron = get_data_by_iron_type(df, FREE_IRON)
    # predict R1
    # predict_R1(data_ferritin_transferrin)
    predict_R1(data_iron)
    # leave_one_out(data_iron)
    predict_R1_lipid(data_iron)
    # pc-sm

