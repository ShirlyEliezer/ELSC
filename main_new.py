from cross_val_predict import *
from parser_process import pre_processing
# ------------------- functions -------------------------


def parameter_iron_lipid_plot(data, iron_lipid):
    colors, markers, markers_types, markers_labels = get_colors_and_markers(data)
    qmri_parameters = np.asarray(data.columns[2:7])
    x_axis = data[iron_lipid]
    for parameter in qmri_parameters:
        y_axis = data[parameter]
        fig, ax = plt.subplots()
        for xp, yp, c, m in zip(x_axis, y_axis, colors, markers):
            ax.scatter([xp], [yp], c=c, marker=m)

        plt.ylabel('{0}'.format(str(parameter)))
        plt.xlabel(str(iron_lipid))
        labels_legend_color = list()
        for lipid in lipid_amount_dict:
            labels_legend_color.append(mpatches.Patch(color=lipid_amount_dict[lipid],
                                                      label=str(labels_legend_dict[lipid])))
        plt.legend(handles=labels_legend_color, loc='best')
        plt.savefig("C:\\Users\\Shirly Eliezer\\Desktop\\university\\third year\\ELSC\\plots\\" +
                    parameter.split("[")[0] + '_' + str(iron_lipid.replace('lipid[%]', 'lipid').
                    replace('[Fe] sigma [mg/ml]', 'iron')) + ".jpg")
        plt.show()


def cross_val_prediction(data):
    predictors = [[IRON, LIPID], [IRON, LIPID, 'interaction'], [IRON, LIPID, 'interaction only']]
    targets = [R1, R2, R2S, MT, MTV]
    for target in targets:
        for predictor in predictors:
            cross_val_prediction_helper(data, predictor, target)


if __name__ == '__main__':
    # pre-processing of the data
    df = pre_processing()
    for lipid in get_lipids(df):
        data = df[df['Lipid type'] == lipid]
        data = data[data.type.str.contains(lipid) == True]
        # parameter_iron_lipid_plot(df, IRON)
        # parameter_iron_lipid_plot(df, LIPID)
        # predicting
        cross_val_prediction(data)

