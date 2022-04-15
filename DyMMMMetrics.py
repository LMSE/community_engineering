
import pandas as pd
from pymoo.factory import get_performance_indicator


def printHV(inputFile):
    df=pd.read_csv(inputFile)
    df=df.drop(["biomass1_SS", "biomass2_SS"], axis=1)
    CSI_High=df.loc[df['CSI'] >= 0.9]
    print(df.shape)
    print(CSI_High.shape)
    X = CSI_High.drop('CSI', axis=1)
    y = CSI_High['CSI']
    max_hv=0
    max_point=None
    for index, row in X.iterrows():
        point=row.to_numpy()
        #reference=point.reshape(point.shape[1])
        hv = get_performance_indicator("hv", ref_point=point)
        hv_value=hv.calc(X.to_numpy())
        print("hv", hv_value)
        if hv_value>max_hv:
            max_hv=hv_value
            max_point=point
    print(max_hv)
    print(max_point)

if __name__ == '__main__':
    printHV("data/communitycomp_RESULT.csv")