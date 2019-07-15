import pandas as pd


def main():

    # extract columns from csv file to data frame
    data = pd.read_csv('stroke_preprocessed_imputed_lvef.csv')

    # initialize new data frame
    df = pd.DataFrame()

    # paste required columns into the new data frame
    for column in data:
        if (column == 'AGE_AT_ADMISSION' or column == 'ADMISSION_TYPE' or column == 'GENDER' or column == 'LVEF' or
                column == 'DIABETES_COMPLICATED' or column == 'DIABETES_UNCOMPLICATED' or column == 'PERIPHERAL_VASCULAR'
                or column == 'RENAL_FAILURE'):
            df[column] = data[column]

    # apply risk score calculation and paste into new column
    df['Risk_Score'] = df.apply(lambda row: risk_score(row['GENDER'], row['ADMISSION_TYPE'], row['AGE_AT_ADMISSION'],
                                                       row['LVEF'], row['DIABETES_COMPLICATED'], row['DIABETES_UNCOMPLICATED'],
                                                       row['PERIPHERAL_VASCULAR'], row['RENAL_FAILURE']), axis=1)

    print(df.head(5))

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


if __name__ == "__main__":
    main()

