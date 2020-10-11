from utils import load_white_wine_data, load_heart_disease_data
from sklearn.preprocessing import MinMaxScaler

def explore_data_wine():
    print("...EXPLORE DATA WINE....")
    white_wine_df = load_white_wine_data()

    # print(white_wine_df.shape)

    X = white_wine_df.iloc[:, :-1].to_numpy()
    y = white_wine_df.iloc[:, -1].to_numpy()
    # Apply threshold 6
    y[y < 6] = 0
    y[y >= 6] = 1

    print("Number of 0 rows: {}".format((y == 0).sum()))
    print("Number of 1 rows: {}".format((y == 1).sum()))

    return X,y

def explore_data_heart_disease():
    print("...EXPLORE DATA HEART DISEASE....")
    heart_disease_df = load_heart_disease_data()
    heart_disease_df = heart_disease_df.dropna()
    # print(heart_disease_df.shape)

    X = heart_disease_df.iloc[:, :-1].to_numpy()
    y = heart_disease_df.iloc[:, -1].to_numpy()
    # Apply threshold 6
    y[y >= 1] = 1

    print("Number of 0 rows: {}".format((y == 0).sum()))
    print("Number of 1 rows: {}".format((y == 1).sum()))

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaling.transform(X)

    return X, y
