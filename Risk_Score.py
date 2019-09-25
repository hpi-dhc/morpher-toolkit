import pandas as pd
from sklearn.calibration import calibration_curve
from collections import defaultdict
from scipy.stats import linregress
from sklearn.metrics import roc_auc_score


def main():

    # extract columns from csv file to data frame
    data = pd.read_csv('stroke_preprocessed_imputed_lvef.csv')

    # initialize new data frame
    df = pd.DataFrame()

    # paste required columns into the new data frame
    for column in data:
        if (column == 'AGE_AT_ADMISSION' or column == 'ADMISSION_TYPE' or column == 'GENDER' or column == 'LVEF' or
                column == 'DIABETES_COMPLICATED' or column == 'DIABETES_UNCOMPLICATED' or column == 'PERIPHERAL_VASCULAR'
                or column == 'RENAL_FAILURE' or column == 'STROKE'):
            df[column] = data[column]

    df = df[df['AGE_AT_ADMISSION'] < 99]

    # apply risk score calculation and paste into new column
    df['Risk_Score'] = df.apply(lambda row: risk_score(row['GENDER'], row['ADMISSION_TYPE'], row['AGE_AT_ADMISSION'],
                                                       row['LVEF'], row['DIABETES_COMPLICATED'], row['DIABETES_UNCOMPLICATED'],
                                                       row['PERIPHERAL_VASCULAR'], row['RENAL_FAILURE']), axis=1)

    # find values of confusion matrix
    df['TP'] = df.apply(lambda row: is_tp(row['STROKE'], row['Risk_Score']), axis=1)
    df['FP'] = df.apply(lambda row: is_fp(row['STROKE'], row['Risk_Score']), axis=1)
    df['FN'] = df.apply(lambda row: is_fn(row['STROKE'], row['Risk_Score']), axis=1)
    df['TN'] = df.apply(lambda row: is_tn(row['STROKE'], row['Risk_Score']), axis=1)
    df['RS_Prediction'] = df.apply(lambda row: rs_prediction(row['Risk_Score']), axis=1)

    print(df)

    # calculate confusion matrix
    tp = sum(df['TP'])
    fp = sum(df['FP'])
    fn = sum(df['FN'])
    tn = sum(df['TN'])

    print('True Positive:', tp, 'False Positive:', fp)
    print('False Negative:', fn, 'True Negative:', tn)

    print('Confusion matrix as ratio to all entries:')

    n = len(df)

    print('Rate of True Positive:', round(tp/n*100), '% | Rate  of False Positive:', round(fp/n*100), '%')
    print('Rate of False Negative:', round(fn/n*100), '% | Rate  of True Negative:', round(tn/n*100), '%')


    # calculate accuracy
    acc = ((tp + tn) / (tp + fp + fn + tn)) * 100

    print('Accuracy:', int(acc), '%')

    # calculate AUC
    y_true = df['STROKE']
    y_score = df['RS_Prediction']

    auc = roc_auc_score(y_true, y_score)

    print('AUC:', round(auc*100), '%')

    # calculate Net Benefit
    tr = 0.7

    net_benefit_treated = (tp/n) - (fp/n) * (tr/(1-tr))
    net_benefit_untreated = (tn/n) - (fn/n) * ((1-tr)/tr)

    print('Net Benefit:', int(net_benefit_treated))

    # save as new csv file
    df.to_csv('Stroke_Calculated_Risk_Score.csv')


# risk score calculation
def risk_score(GENDER, ADMISSION_TYPE, AGE_AT_ADMISSION, LVEF, DIABETES_COMPLICATED, DIABETES_UNCOMPLICATED, PERIPHERAL_VASCULAR, RENAL_FAILURE):

    stroke_score = 1.5 * DIABETES_COMPLICATED + 1.5 * DIABETES_UNCOMPLICATED + 2 * PERIPHERAL_VASCULAR + 2 * RENAL_FAILURE

    if 55 <= AGE_AT_ADMISSION <= 59:
        stroke_score += 1.5
    elif 60 <= AGE_AT_ADMISSION <= 64:
        stroke_score += 2.5
    elif 65 <= AGE_AT_ADMISSION <= 69:
        stroke_score += 3.5
    elif 70 <= AGE_AT_ADMISSION <= 74:
        stroke_score += 4
    elif 75 <= AGE_AT_ADMISSION <= 79:
        stroke_score += 4.5
    elif AGE_AT_ADMISSION >= 80:
        stroke_score += 5.5

    if GENDER == 0:
        stroke_score += 1

    if ADMISSION_TYPE == 1:
        stroke_score += 1.5
    elif ADMISSION_TYPE == 2:
        stroke_score += 2.5

    if LVEF < 40:
        stroke_score += 1.5

    return stroke_score


def is_tp(Stroke, Risk_Score):

    if Stroke == 1.0 and Risk_Score >= 5.5:
        return 1
    else:
        return 0


def is_fn(Stroke, Risk_Score):

    if Stroke == 1.0 and Risk_Score < 5.5:
        return 1
    else:
        return 0


def is_fp(Stroke, Risk_Score):

    if Stroke == 0.0 and Risk_Score >= 5.5:
        return 1
    else:
        return 0


def is_tn(Stroke, Risk_Score):

    if Stroke == 0.0 and Risk_Score < 5.5:
        return 1
    else:
        return 0

def rs_prediction(Risk_Score):

    if Risk_Score >= 5.5:
        return 1
    else:
        return 0


if __name__ == "__main__":
    main()

