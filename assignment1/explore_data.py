from utils import load_white_wine_data, load_heart_disease_data
import seaborn as sns


def explore_data_wine():
    print("...EXPLORE DATA WINE....")
    white_wine_df = load_white_wine_data()

    # print(white_wine_df.shape)

    ax = sns.distplot(white_wine_df["quality"], norm_hist=False, kde=False).set_title(
        'White wine quality histogram')
    fig = ax.get_figure()
    fig.savefig("plots\wine\white_wine_histogram_before.png")
    fig.clf()

    X = white_wine_df.iloc[:, :-1].to_numpy()
    y = white_wine_df.iloc[:, -1].to_numpy()
    # Apply threshold 6
    y[y < 6] = 0
    y[y >= 6] = 1

    print("Number of 0 rows: {}".format((y == 0).sum()))
    print("Number of 1 rows: {}".format((y == 1).sum()))

    ax = sns.distplot(y, norm_hist=False, kde=False).set_title('White wine quality histogram - binary')

    fig = ax.get_figure()
    fig.savefig("plots\wine\white_wine_histogram_binary.png")
    fig.clf()


    return X,y

def explore_data_heart_disease():
    print("...EXPLORE DATA HEART DISEASE....")
    heart_disease_df = load_heart_disease_data()
    heart_disease_df = heart_disease_df.dropna()
    # print(heart_disease_df.shape)
    ax = sns.distplot(heart_disease_df["target"], norm_hist=False, kde=False).set_title(
        'Heart disease histogram')
    fig = ax.get_figure()
    fig.savefig("plots\heart\heart_histogram_before.png")
    fig.clf()

    X = heart_disease_df.iloc[:, :-1].to_numpy()
    y = heart_disease_df.iloc[:, -1].to_numpy()
    # Apply threshold 6
    y[y >= 1] = 1

    print("Number of 0 rows: {}".format((y == 0).sum()))
    print("Number of 1 rows: {}".format((y == 1).sum()))

    ax = sns.distplot(y, norm_hist=False, kde=False).set_title('Heart disease histogram - binary')

    fig = ax.get_figure()
    fig.savefig("plots\heart\heart_histogram_binary.png")
    fig.clf()

    return X, y
