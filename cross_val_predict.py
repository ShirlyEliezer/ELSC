from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
import matplotlib.patches as mpatches
from parser_process import *


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
    for lipid in lipid_amount_dict:
      labels_legend_color.append(mpatches.Patch(color=lipid_amount_dict[lipid],
                                                label=str(labels_legend_dict[lipid])))

    legend1 = plt.legend(list(zip(colors, markers_types)), markers_labels,
            handler_map={tuple: MarkerHandler()}, loc=4)
    plt.legend(handles=labels_legend_color, loc=1)
    plt.gca().add_artist(legend1)

    accuracy = r2_score(y, predictions)
    # the lower the MAE, the more closely a model is able to predict the actual observations.
    mae = mean(absolute(scores))
    predictors = ""
    if 'interaction only' in vars:
        predictors = 'a * ' + labels['interaction'] + ' + b'
    else:
        for i in range(0, len(vars)):
            if coefficients[i] == 'c':
                predictors += '\n' + str(coefficients[i]) + ' * ' + str(labels[vars[i]]) + ' + '
            else:
                predictors += str(coefficients[i]) + ' * ' + str(labels[vars[i]]) + ' + '
        predictors = predictors[:len(predictors)-2]
        predictors = predictors + " + " + str(coefficients[i + 1])
    plt.title(str(for_title_dict[target]) + ' = ' + predictors + "\n"
              "R^2: " + str(float("{:.3f}".format(accuracy))) +
              " Mean absolute squared error: " + str(float("{:.3f}".format(mae))))
    plt.xlabel(str(target) + " measured")
    plt.ylabel(str(target) + " predicted")
    # plt.xlim([0.05, 0.3])
    # plt.ylim([0.225, 0.23])
    plt.savefig("C:\\Users\\Shirly Eliezer\\Desktop\\university\\third year\\ELSC\\plots\\" +
            ', '.join(vars).replace('[Fe] sigma [mg/ml]', 'iron').replace(' ', '').
            replace('lipid[%]', 'lipid') + '_' + str(target.replace(' ', '')).split('[')[0] +
                ".jpg")
    plt.show()


