import pandas as pd


def main():

    # extract needed columns from file

    data = pd.read_csv('stroke_preprocessed_imputed_lvef.csv')

    df = pd.DataFrame()

    for column in data:
        if (column == 'AGE_AT_ADMISSION' or column == 'ADMISSION_TYPE' or column == 'GENDER' or column == 'LVEF' or
                column == 'DIABETES_COMPLICATED' or column == 'DIABETES_UNCOMPLICATED' or column == 'PERIPHERAL_VASCULAR'
                or column == 'RENAL_FAILURE'):
            df[column] = data[column]

    # print(df.head(5))

    #df = df.reindex(['GENDER', 'ADMISSION_TYPE', 'AGE_AT_ADMISSION', 'LVEF', 'DIABETES_COMPLICATED',
     #             'DIABETES_UNCOMPLICATED', 'PERIPHERAL_VASCULAR', 'RENAL_FAILURE'], axis=1)
    # print(list(df.columns.values))

    #colNames = ['GENDER', 'ADMISSION_TYPE', 'AGE_AT_ADMISSION', 'LVEF', 'DIABETES_COMPLICATED', 'DIABETES_UNCOMPLICATED',
     #           'PERIPHERAL_VASCULAR', 'RENAL_FAILURE']

    #df['Risk_Score'] = df.apply(lambda row: risk_score(row[colNames[0]], row[colNames[1]], row[colNames[2]], row[colNames[3]], row[colNames[4]], row[colNames[5]], row[colNames[6]], row[colNames[7]]), axis=1)
    df['Risk_Score'] = df.apply(lambda row: risk_score(row['GENDER'], row['ADMISSION_TYPE'], row['AGE_AT_ADMISSION'],
                                                       row['LVEF'], row['DIABETES_COMPLICATED'], row['DIABETES_UNCOMPLICATED'],
                                                       row['PERIPHERAL_VASCULAR'], row['RENAL_FAILURE']), axis=1)

    print(df.head(5))

    df.to_csv('Stroke_Risk_Score.csv')

def risk_score(GENDER, ADMISSION_TYPE, AGE_AT_ADMISSION, LVEF, DIABETES_COMPLICATED, DIABETES_UNCOMPLICATED, PERIPHERAL_VASCULAR, RENAL_FAILURE):

    stroke_score = 1.5 * DIABETES_COMPLICATED + 1.5 * DIABETES_UNCOMPLICATED + 2 * PERIPHERAL_VASCULAR + 2 * RENAL_FAILURE


    if 18 <= AGE_AT_ADMISSION <= 54:
        if ADMISSION_TYPE == 0:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1
                else:
                    stroke_score += 1
            else:
                if LVEF < 40:
                    stroke_score += 1.5

        elif ADMISSION_TYPE == 1:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 1.5
                else:
                    stroke_score += 1 + 1.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 1.5
                else:
                    stroke_score += 1.5
        else:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 2.5
                else:
                    stroke_score += 1 + 2.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 2.5
                else:
                    stroke_score += 2.5
    elif 55 <= AGE_AT_ADMISSION <= 59:
        if ADMISSION_TYPE == 0:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 1.5
                else:
                    stroke_score += 1 + 1.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 1.5
                else:
                    stroke_score += 1.5
        elif ADMISSION_TYPE == 1:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 1.5 + 1.5
                else:
                    stroke_score += 1 + 1.5 + 1.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 1.5 + 1.5
                else:
                    stroke_score += 1.5 + 1.5
        else:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 2.5 + 1.5
                else:
                    stroke_score += 1 + 2.5 + 1.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 2.5 + 1.5
                else:
                    stroke_score += 2.5 + 1.5
    elif 60 <= AGE_AT_ADMISSION <= 64:
        if ADMISSION_TYPE == 0:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 2.5
                else:
                    stroke_score += 1 + 2.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 2.5
                else:
                    stroke_score += 2.5
        elif ADMISSION_TYPE == 1:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 1.5 + 2.5
                else:
                    stroke_score += 1 + 1.5 + 2.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 1.5 + 2.5
                else:
                    stroke_score += 1.5 + 2.5
        else:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 2.5 + 2.5
                else:
                    stroke_score += 1 + 2.5 + 2.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 2.5 + 2.5
                else:
                    stroke_score += 2.5 + 2.5
    elif 65 <= AGE_AT_ADMISSION <= 69:
        if ADMISSION_TYPE == 0:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 3.5
                else:
                    stroke_score += 1 + 3.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 3.5
                else:
                    stroke_score += 3.5
        elif ADMISSION_TYPE == 1:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 1.5 + 3.5
                else:
                    stroke_score += 1 + 1.5 + 3.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 1.5 + 3.5
                else:
                    stroke_score += 1.5 + 3.5
        else:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 2.5 + 3.5
                else:
                    stroke_score += 1 + 2.5 + 3.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 2.5 + 3.5
                else:
                    stroke_score = + 2.5 + 3.5
    elif 70 <= AGE_AT_ADMISSION <= 74:
        if ADMISSION_TYPE == 0:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 4
                else:
                    stroke_score += 1 + 4
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 4
                else:
                    stroke_score += 4
        elif ADMISSION_TYPE == 1:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 1.5 + 4
                else:
                    stroke_score += 1 + 1.5 + 4
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 1.5 + 4
                else:
                    stroke_score += 1.5 + 4
        else:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 2.5 + 4
                else:
                    stroke_score += 1 + 2.5 + 4
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 2.5 + 4
                else:
                    stroke_score += 2.5 + 4
    elif 75 <= AGE_AT_ADMISSION <= 79:
        if ADMISSION_TYPE == 0:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 4.5
                else:
                    stroke_score += 1 + 4.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 4.5
                else:
                    stroke_score += 4.5
        elif ADMISSION_TYPE == 1:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 1.5 + 4.5
                else:
                    stroke_score += 1 + 1.5 + 4.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 1.5 + 4.5
                else:
                    stroke_score += 1.5 + 4.5
        else:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 2.5 + 4.5
                else:
                    stroke_score += 1 + 2.5 + 4.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 2.5 + 4.5
                else:
                    stroke_score += 2.5 + 4.5
    elif 80 <= AGE_AT_ADMISSION <= 99:
        if ADMISSION_TYPE == 0:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 5.5
                else:
                    stroke_score += 1 + 5.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 5.5
                else:
                    stroke_score += 5.5
        elif ADMISSION_TYPE == 1:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 1.5 + 5.5
                else:
                    stroke_score += 1 + 1.5 + 5.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 1.5 + 5.5
                else:
                    stroke_score += 1.5 + 5.5
        else:
            if GENDER == 0:
                if LVEF < 40:
                    stroke_score += 1.5 + 1 + 2.5 + 5.5
                else:
                    stroke_score += 1 + 2.5 + 5.5
            else:
                if LVEF < 40:
                    stroke_score += 1.5 + 2.5 + 5.5
                else:
                    stroke_score += 2.5 + 5.5

    return stroke_score


if __name__ == "__main__":
    main()

