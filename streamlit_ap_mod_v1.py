# Imports #########################################################################
import streamlit as st
import joblib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.cm import RdYlGn
#from PIL import Image
from random import randint

# Constants #########################################################################
DEBUG = False

# Adress of the API server :
# HOST = 'http://127.0.0.1:8000'     # developement on local server
HOST = 'https://project-api.herokuapp.com'  # production server


# Functions #########################################################################
def optimum_threshold():
    """Gets the optimum threshold of the buisness cost function on the API server.
    Args :
    - None.
    Returns :
    - float between 0 and 1.
    """
    response = requests.get(HOST + '/optimum_threshold')
    return round(float(response.content), 3)


def get_prediction(id_client: int):
    """Gets the probability of default of a client on the API server.
    Args : 
    - id_client (int).
    Returns :
    - probability of default (float).
    """
    json_client = df_test_sample.loc[int(id_client)].to_json()
    response = requests.get(HOST + '/prediction/', data=json_client)
    proba_default = eval(response.content)["probability"]
    return proba_default


def rectangle_gauge(id, client_probability):
    """Draws a gauge for the result of credit application, and an arrow at the client probability of default.
        Args :
        - id (int) : client ID.
        - client_probability (float).
        Returns :
        - draws a matplotlib figure.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 1))
    fig.suptitle(f"Client {id}: probability of credit default (%)",
                 size=15,
                 y=1.1)
    ax.add_patch(
        Rectangle((0, 0),
                  width=optimum_threshold() * 100,
                  height=1,
                  color=(0.5, 0.9, 0.5, 0.5)))
    ax.add_patch(
        Rectangle((optimum_threshold() * 100, 0),
                  width=100 - optimum_threshold() * 100,
                  height=1,
                  color=(1, 0, 0, 0.5)))
    ax.plot((optimum_threshold() * 100, optimum_threshold() * 100), (0, 1),
            color='#FF8C00',
            ls=(0, (0.5, 0.5)),
            lw=6)
    ax.add_patch(
        FancyArrowPatch((client_probability * 100, 0.75),
                        (client_probability * 100, 0),
                        mutation_scale=20))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 105, 10))
    ax.set_yticks([])
    st.pyplot(fig)


def get_shap(id_client: int):
    """Gets the SHAP values of a client on the API server.
    Args : 
    - id_client (int).
    Returns :
    - pandas dataframe with 2 columns : features, SHAP values.
    """
    json_client = df_test_sample.loc[int(id_client)].to_json()
    response = requests.get(HOST + '/shap/', data=json_client)
    df_shap = pd.read_json(eval(response.content), orient='index')
    return df_shap


def load_data(filename, path='./resources/'):
    """Loads serialized data from the resources directory.
    Args :
    - file name (string).
    Returns :
    - unserialized data.
    """
    return joblib.load(path + filename + '.joblib')


def kdeplot_in_common(feature, bw_method=0.4):
    """KDE plot of a quantitative feature. Common to all clients.
    Args :
    - feature (string).
    Returns :
    - matplotlib figure.
    """
    # Extraction of the feature's data
    df = pd.DataFrame({
        feature: X_split_valid[feature],# à remplacer par data_train
        'y_true': y_split_valid# à remplacer par TARGET_data
    })
    ser_true0 = df.loc[df['y_true'] == 0, feature]
    ser_true1 = df.loc[df['y_true'] == 1, feature]
    xmin = df[feature].min()
    xmax = df[feature].max()
    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4, dpi=100)
    ser_true0.plot(kind='kde',
                   c='g',
                   label='Non-defaulting clients',
                   bw_method=bw_method,
                   ind=None)
    ser_true1.plot(kind='kde',
                   c='r',
                   label='Defaulting clients',
                   bw_method=bw_method,
                   ind=None)
    fig.suptitle(
        f'Observed distribution of {feature} based on clients true class',
        y=0.92)
    plt.legend()
    plt.xlabel(feature)
    plt.ylabel('Probability density')
    plt.xlim(xmin, xmax)
    return fig


def kdeplot(feature):
    """Plots a KDE of the quantitative feature.  
    Args :
    - feature (string).
    Returns :
    - matplotlib plot via st.pyplot.
    """
                        
    figure = kdeplot_in_common(feature)
    y_max = plt.ylim()[1]
    x_client = one_client_pandas[feature].iloc[0]
    if str(x_client) == "nan":
        x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
        plt.annotate(text=f" Client {id_client}\n  data not available",
                     xy=(x_center, 0),
                     xytext=(x_center, y_max * 0.8))
    else:
        plt.axvline(x=x_client,
                    ymin=-1e10,
                    ymax=1e10,
                    c='k',
                    ls='dashed',
                    lw=2)
        plt.annotate(text=f" Client {id_client}\n  {round(x_client, 3)}",
                     xy=(x_client, y_max * 0.8))
    st.pyplot(figure)
    st.caption(feature + ": " + feature_description(feature))


def barplot_in_common(feature):
    """Horizontal Barplot of a qualitative feature. Common to all clients.
    Args :
    - feature (string).
    Returns :
    - matplotlib figure.
    """
    # Extraction of the feature's data
    df_feature = pd.DataFrame({
        feature: X_split_valid[feature],# à remplacer par data_train
        'y_true':  y_split_valid # à remplacer par TARGET_data
    })
    # Observed probability of default for each value of the feature
    proba_for_each_value = []
    cardinality = len(
        dict_categorical_features[feature]) if feature != 'CODE_GENDER' else 2
    for index in range(
            cardinality):  # on parcourt toutes les modalités de la feature
        df_feature_modalite = df_feature[df_feature[feature] == index]
        proba_default = df_feature_modalite['y_true'].sum() / len(
            df_feature_modalite)
        proba_for_each_value.append(proba_default)
    df_modalites = pd.DataFrame()
    df_modalites['modalites'] = dict_categorical_features[
        feature] if feature != 'CODE_GENDER' else ['Female', 'Male']
    df_modalites['probas'] = proba_for_each_value
    df_modalites.sort_values(by='probas', inplace=True)
    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4)
    plt.ylim(-0.6, cardinality - 0.4)
    plt.barh(y=range(cardinality), width=df_modalites['probas'], color='#eb2f06')
    plt.barh(
        y=range(cardinality),
        left=df_modalites['probas'],
        width=(1 - df_modalites['probas']),
        color=' #a2ff62 ',
    )
    plt.xlabel('Observed probability of default')
    plt.ylabel(feature)
    fig.suptitle(
        f'Observed probability of default as a function of {feature} based on clients true class',
        y=0.92)
    size = 6 if cardinality > 30 else None
    plt.yticks(ticks=range(cardinality),
               labels=df_modalites['modalites'],
               size=size)
    return fig


def barplot(feature):
    """Barplot of a qualitative feature. 
    Args :
    - feature (string).
    Returns :
    - matplotlib plot via st.pyplot.
    """
    figure = barplot_in_common(feature)
    x_client = one_client_pandas[feature].iloc[0]
    plt.axvline(x=optimum_threshold(),
                ymin=-1e10,
                ymax=1e10,
                c='darkorange',
                ls='dashed',
                lw=1)  # line for the optimum_threshold
    plt.text(
        s=
        f" Client {id_client}: {dict_categorical_features[feature][x_client]} ",
        x=0.5,
        y=plt.ylim()[1] * 0.3)
    st.pyplot(figure)
    st.caption(feature + ": " + feature_description(feature))

def get_shap(id_client: int):
    """Gets the SHAP values of a client on the API server.
    Args : 
    - id_client (int).
    Returns :
    - pandas dataframe with 2 columns : features, SHAP values.
    """
    json_client = df_test_sample.loc[int(id_client)].to_json()
    response = requests.get(HOST + '/shap/', data=json_client)
    df_shap = pd.read_json(eval(response.content), orient='index')
    return df_shap


def contourplot_in_common(feature1, feature2):
    """Contour plot for the observed probability of default as a function of 2 features. Common to all clients.
    Args :
    - feature1 (string).
    - feature2 (string).
    Returns :
    - matplotlib figure.
    """
    target_mesh_size = 500  # target population for each mesh

    # Preparation of data
    df = pd.DataFrame({
        feature1: X_split_valid[feature1],# à remplacer par data_train
        feature2: X_split_valid[feature2],# à remplacer par data_train
        'y_true': y_split_valid # à remplacer par TARGET_data
    })
    df = df.dropna().copy()
    n_values = len(df)
    n_bins = int(np.ceil(np.sqrt(n_values / target_mesh_size)))
    bin_size = int(np.floor(n_values / n_bins))
    index_bin_start = sorted([bin_size * n for n in range(n_bins)])
    ser1 = df[feature1].sort_values().copy()
    ser2 = df[feature2].sort_values().copy()

    # Filling the grid
    grid_proba_default = np.full((n_bins, n_bins), -1.0)
    ser_true0 = (df['y_true'] == 0)
    ser_true1 = (df['y_true'] == 1)
    for i1, ind1 in enumerate(index_bin_start):
        for i2, ind2 in enumerate(index_bin_start):
            ser_inside_this_mesh = (df[feature1] >= ser1.iloc[ind1]) & (df[feature2] >= ser2.iloc[ind2]) \
                & (df[feature1] <= ser1.iloc[ind1+bin_size-1]) & (df[feature2] <= ser2.iloc[ind2+bin_size-1])
            # sum of clients true0 inside this square bin
            sum_0 = (ser_inside_this_mesh & ser_true0).sum()
            sum_1 = (ser_inside_this_mesh & ser_true1).sum()
            sum_ = sum_0 + sum_1
            if sum_ == 0:
                proba_default = 1
            else:
                proba_default = sum_1 / sum_
            grid_proba_default[i2, i1] = proba_default

    # X, Y of the grid
    X = [ser1.iloc[i + int(bin_size / 2)] for i in index_bin_start]
    Y = [ser2.iloc[i + int(bin_size / 2)] for i in index_bin_start]

    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4)
    plt.contourf(X, Y, grid_proba_default, cmap=' Reds ')
    plt.colorbar(shrink=0.8)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    fig.suptitle(
        f'Observed probability of default as a function of {feature1} and {feature2}',
        y=0.92)
    return fig


def contourplot(feature1, feature2):
    """Contour plot for the observed probability of default as a function of 2 features. 
    Args :
    - feature1 (string).
    - feature2 (string).
    Returns :
    - matplotlib plot via st.pyplot.
    """
    figure = contourplot_in_common(feature1, feature2)
    x_client = one_client_pandas[feature1].iloc[0]
    y_client = one_client_pandas[feature2].iloc[0]
    if str(x_client) == "nan" or str(y_client) == "nan":
        x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
        y_center = (plt.ylim()[1] + plt.ylim()[0]) / 2
        plt.text(s=f" Client {id_client}\n  data not available",
                 x=x_center,
                 y=y_center)
    else:
        plt.axvline(x=x_client,
                    ymin=-1e10,
                    ymax=1e10,
                    c='k',
                    ls='dashed',
                    lw=1)
        plt.axhline(y=y_client,
                    xmin=-1e10,
                    xmax=1e10,
                    c='k',
                    ls='dashed',
                    lw=1)
        # if I want to interpolate data : https://stackoverflow.com/questions/5666056/matplotlib-extracting-data-from-contour-lines
    st.pyplot(figure)
    st.caption(feature1 + ": " + feature_description(feature1))
    st.caption(feature2 + ": " + feature_description(feature2))


def lineplot_in_common(feature):
    """Line plot of a quantitative feature. Common to all clients.
    Plot smoothed over 4000 clients. One dot plotted every 1000 clients.
    Args :
    - feature (string).
    Returns :
    - matplotlib figure.
    """
    target_bin_size = 4000

    # preparation of data
    df = pd.DataFrame({
        feature: X_split_valid[feature],# à remplacer par data_train
        'y_true': y_split_valid # à remplacer par TARGET
    })
    df = df.dropna().sort_values(axis=0, by=feature).copy()
    n_values = len(df)
    n_bins = int(np.ceil(n_values / target_bin_size))
    bin_size = int(np.floor(n_values / n_bins))
    index_bin_start = [bin_size*n for n in range(n_bins)] + [int(bin_size*(n+0.25)) for n in range(n_bins)] \
        + [int(bin_size*(n+0.5)) for n in range(n_bins)] + \
        [int(bin_size*(n+0.75)) for n in range(n_bins)]
    index_bin_start = sorted(index_bin_start)

    # Observed probability of default for every bins
    proba_default = []
    feature_value_start = []
    for i in index_bin_start[2:-2]:
        some_bin = df.iloc[int(i - 0.5 * bin_size):int(i + 0.5 * bin_size)]
        some_bin_sum0 = (some_bin['y_true'] == 0).sum()
        some_bin_sum1 = (some_bin['y_true'] == 1).sum()
        some_bin_sum = some_bin_sum0 + some_bin_sum1
        proba_default_ = some_bin_sum1 / some_bin_sum
        proba_default.append(proba_default_)
        feature_value_start.append(df[feature].iloc[i])

    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4)
    plt.plot(feature_value_start, proba_default, color='k')
    ylim_high = plt.ylim()[1]
    plt.fill_between(x=feature_value_start, y1=proba_default, y2=0, color='#eb2f06')
    plt.fill_between(x=feature_value_start,
                     y1=proba_default,
                     y2=1,
                     color=' #a2ff62 ')
    plt.ylabel('Observed probability of default')
    plt.xlabel(feature)
    fig.suptitle(f'Observed probability of default as a function of {feature}',
                 y=0.92)
    plt.ylim(0, max(ylim_high, 0.3))
    return fig


def lineplot(feature):
    """Plots a lineplot of the quantitative feature.  
    Args :
    - feature (string).
    Returns :
    - matplotlib plot via st.pyplot.
    """
    figure = lineplot_in_common(feature)
    y_max = plt.ylim()[1]
    x_client = one_client_pandas[feature].iloc[0]
    if str(x_client) == "nan":
        x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
        plt.annotate(text=f" Client {id_client}\n  data not available",
                     xy=(x_center, 0),
                     xytext=(x_center, y_max * 0.9))
    else:
        plt.axvline(x=x_client,
                    ymin=-1e10,
                    ymax=1e10,
                    c='k',
                    ls='dashed',
                    lw=2)
        plt.axhline(y=optimum_threshold(),
                    xmin=-1e10,
                    xmax=1e10,
                    c='darkorange',
                    ls='dashed',
                    lw=1)  # line for the optimum_threshold
        plt.annotate(text=f" Client {id_client}\n  {round(x_client, 3)}",
                     xy=(x_client, y_max * 0.9))
    st.pyplot(figure)
    st.caption(feature + ": " + feature_description(feature))


def plot_selector(feature, dashboard='Advanced Dashboard'):
    """Chooses between a KDE plot (for quantitative features) and a bar plot (for qualitative features)
    Args :
    - feature (string).
    - dashboard (string) : 'Advanced Dashboard' or 'Basic Dashboard'.
    Returns :
    - matplotlib plot via st.pyplot of the called function.	
    """
    if feature in list_categorical_features:
        barplot(feature)
    else:
        if dashboard == 'Advanced Dashboard':
            kdeplot(feature)
        else:
            lineplot(feature)


def shap_barplot(df_shap):
    """Plots an horizontal barplot of 10 SHAP values (the 5 most positive contributions and the 5 most negatives to the probability of default)
    Args :
    - df_shap (dataframe) : SHAP values and feature names.
    Returns :
    - matplotlib plot via st.pyplot.
    """
    # Preparation of data
    df = df_shap.sort_values(by='SHAP value', ascending=False)
    df = df.head(5).append(df.tail(5)).copy()
    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4)
    colors = [RdYlGn(0.05*i) for i in range(5)] + \
        [RdYlGn(0.8 + 0.04*i) for i in range(5)]
    plt.barh(width=df['SHAP value'], y=df['feature'], color=colors)
    plt.xlabel('SHAP value')
    plt.ylabel('Features (top 5 contributors, both ways)')
    fig.suptitle('Impact on model output (credit default)', y=0.92, size=14)
    st.pyplot(fig)
    st.caption(
        "Horizontal scale : contribution to log odds of credit default.")
    with st.expander("Features description", expanded=False):
        for feature in list(df['feature']):
            st.caption(feature + ": " + feature_description(feature))


def feature_description(feature):
    """Returns a description of the feature, taken from the table HomeCredit_columns_description.csv.
    Args : 
    - feature (string).
    Returns :
    - its description (string.)
    """
    if feature in list(df_description.Row):
        description = df_description[df_description.Row ==
                                     feature]['Description'].iloc[0]
    else:
        description = "Description not available"
    return description


# Web page #########################################################################

# Load data
# sample of customers from the Kaggle test set
df_test_sample = load_data('df_test_sample', path='./src/') 

# Default settings. This must be the first Streamlit command used in your app, and must only be set once.
st.set_page_config(page_title="Project 7 Dashboard",
                   initial_sidebar_state="expanded",
                   layout="wide")

# Side bar
with st.sidebar:
    #image_HC = Image.open('./img/Home-Credit-logo.jpg')
    #st.image(image_HC, width=300)

    # Dashboard selector
    st.write('## Site Map:')
    dashboard_choice = st.radio('', [
        'Homepage', 'Basic Dashboard', 'Advanced Dashboard'
    ])
    st.write('## ')
    st.write('## ')

    if dashboard_choice in ['Basic Dashboard', 'Advanced Dashboard']:
        
        # Client selector
        st.write('## Client ID:')
        id_client = st.text_input("Enter client ID", value="100013")
        
        # st.caption("Example of client predicted negative (no default) : 324806")
        # st.caption("Example of client predicted positive (credit default) : 318063")
        st.caption(" ")
        # Button random
        if st.button("Select random client"):
            clients = [str(id) for id in df_test_sample.index]
            size = len(clients)
            client_index = np.random.randint(0, size - 1)
            id_client = clients[client_index]
   

# Homepage #######################################################
if dashboard_choice == 'Homepage':
    st.title("Home Credit Default Risk Prediction")
    " "
    " "
    "This site contains an **interactive dashboard** to explain to the bank's customers the reason of **approval or refusal of their credit applications.**"
    "Probability of credit default has been calculated by a prediction model based on machine learning."
    " "
    " "
    "The bullet points of the prediction model are:"
    "- The data used for model training contain the entire set of tables available for the [Home Credit data repository at Kaggle.](https://www.kaggle.com/c/home-credit-default-risk/data)"
    "- The prediction model used to determine the probability of default of a credit application is based on the **LGBM algorithm** (Light Gradient Boosting Machine)."
    "- This model has been optimized with the intent to **minimize the buisness cost function** : each defaulting client costs 10 times the gain of a non-defaulting client."
    f"- The optimization  has lead to an **optimum threshold for the probability of default : {100*optimum_threshold()}%**. In other words, customer with a probability of default below {100*optimum_threshold()}% get their credit accepted, and refused if above {100*optimum_threshold()}%."
    " "
    " "
    "The dashboard is available in 2 versions:"
    "- A **basic** version, to be used by customer relation management."
    "- An **advanced**, more detailed version for deeper understanding of the data."

    " "

# Basic and Advanced Dashboards #######################################################
elif dashboard_choice in ['Basic Dashboard', 'Advanced Dashboard']:
    # Load data
    # dataset used to plot features - sample of clients from the Kaggle train set
    X_split_valid = load_data('X_split_valid', path='./src/')
    # dataset used to plot features - sample of clients from the Kaggle train set
    y_split_valid = load_data('y_split_valid', path='./src/')
    list_categorical_features = load_data('list_categorical_features')
    dict_categorical_features = load_data('dict_categorical_features')
    list_quantitative_features = load_data('list_quantitative_features')
    df_description = load_data('df_description')
    #list_features_permutation_importances = load_data(
        #'list_features_permutation_importances')
    #list_summary_plot_shap = load_data('list_summary_plot_shap')

    # Main title of the dashboard
    st.title(f'Default Risk Prediction for client {id_client}')

    # Convert id_client into a dataframe
    one_client_pandas = df_test_sample[df_test_sample.index == int(id_client)]

    # Result of credit application
    "---------------------------"
    st.header('Result of credit application')
    probability = get_prediction(id_client)
    if probability < optimum_threshold():
        st.success(
            f"  \n __CREDIT ACCEPTED__  \n  \nThe probability of default of the applied credit is __{round(100*probability,1)}__% (lower than the threshold of {100*optimum_threshold()}% for obtaining the credit).  \n "
        )
    else:
        st.error(
            f"__CREDIT REFUSED__  \nThe probability of default of the applied credit is __{round(100*probability,1)}__% (higher than the threshold of {100*optimum_threshold()}% for obtaining the credit).  \n "
        )
    rectangle_gauge(id_client, probability)


    # Positioning of the client with comparison to other clients in the 6 main features
    "---------------------------"
    st.header(
        'Positioning of the client with comparison to other clients in the 6 main features'
    )

    # Basic dashboard
    st.subheader('Observed probability of default as a function of a feature')
    left_column, middle_column, right_column = st.columns([1, 1, 1])
    with left_column:
        plot_selector('EXT_SOURCE_1', dashboard='Basic Dashboard')
    with middle_column:
        plot_selector('EXT_SOURCE_2', dashboard='Basic Dashboard')
    with right_column:
        plot_selector('EXT_SOURCE_3', dashboard='Basic Dashboard')
    with left_column:
        plot_selector('AMT_ANNUITY', dashboard='Basic Dashboard')
    with middle_column:
        plot_selector('PAYMENT_RATE', dashboard='Basic Dashboard')
    with right_column:
        plot_selector('CODE_GENDER', dashboard='Basic Dashboard')

    # Advanced dashboard
    if dashboard_choice == 'Advanced Dashboard':
        st.subheader(
            'Distribution of a feature, based on the clients true class')
        left_column, right_column = st.columns(2)
        with left_column:
            plot_selector('EXT_SOURCE_1')
            plot_selector('EXT_SOURCE_2')
        with right_column:
            plot_selector('EXT_SOURCE_3')
            plot_selector('AMT_ANNUITY')

    # Positioning of the client with comparison to other clients (choice of feature)
    "---------------------------"
    st.header(
        'Positioning of the client with comparison to other clients (choice of feature)'
    )
    sorted_options = sorted(list(df_test_sample.columns))
    selected_feature = st.selectbox(
        f'Choose a feature among {len(sorted_options)}',
        options=sorted_options,
        index=sorted_options.index('OCCUPATION_TYPE'))

    # Basic dashboard
    plot_selector(selected_feature, dashboard='Basic Dashboard')

    # Advanced dashboard
    if dashboard_choice == 'Advanced Dashboard' and selected_feature in list_quantitative_features:
        plot_selector(selected_feature)

    # Positioning of the client with comparison to other clients (choice of two features)
    "---------------------------"
    st.header(
        'Positioning of the client with comparison to other clients (choice of two features)'
    )
    left_column, right_column = st.columns(2)
    with left_column:
        selected_feature1 = st.selectbox(
            'Choose feature 1',
            options=list_quantitative_features,
            index=list_quantitative_features.index('EXT_SOURCE_1'))
    with right_column:
        selected_feature2 = st.selectbox(
            'Choose feature 2',
            options=list_quantitative_features,
            index=list_quantitative_features.index('EXT_SOURCE_2'))
    contourplot(selected_feature1, selected_feature2)

    # Local SHAP
    "---------------------------"
    st.header('Impact of features on prediction of default')
    df_shap = get_shap(id_client)
    shap_barplot(df_shap)

    # Display client data
    "---------------------------"
    st.header(f'Data for client {id_client}')
    with st.expander("See data", expanded=False):
        for feature in list_categorical_features:
            encoded_feature = one_client_pandas[feature].iloc[0]
            decoded_feature = dict_categorical_features[feature][
                encoded_feature]
            one_client_pandas[feature].iloc[0] = decoded_feature
        one_client_pandas = one_client_pandas.append(pd.Series(),
                                                     ignore_index=True)
        for feature in one_client_pandas.columns:
            one_client_pandas[feature].iloc[1] = feature_description(feature)

        one_client_pandas = one_client_pandas.T
        one_client_pandas.columns = ['Value', 'Description']
        st.components.v1.html(one_client_pandas.to_html(
            na_rep='Value not available', justify='left'),
                              width=1200,
                              height=400,
                              scrolling=True)

    # Logs and debugs
    if DEBUG:
        "---------------------------"
        st.header("Logs and debugs")
        st.write("optimum_threshold :", optimum_threshold())

    # Final line
    "---------------------------"

