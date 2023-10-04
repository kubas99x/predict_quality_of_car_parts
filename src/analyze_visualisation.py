import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def make_and_save_pariplot(whole_df ,columns_to_analyze, file_name):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if not os.path.exists(os.path.join(parent_dir, 'not_in_repo', file_name)):
        plot = sns.pairplot(data=whole_df[columns_to_analyze + ['our_final_status']], hue="our_final_status", palette={ 1 : "green", 2:"red"}, plot_kws={"s": 3})
        plot.savefig(os.path.join(parent_dir, 'not_in_repo', file_name))

def make_and_save_heatmap(whole_df, col_to_heatmap, file_name):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if not os.path.exists(os.path.join(parent_dir, 'not_in_repo', file_name)):
        col_to_heatmap.append('our_final_status')
        selected_data = whole_df[col_to_heatmap]
        corrmat = selected_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corrmat, annot=True,cbar=True, cmap='Purples', fmt='.1f', linewidths=0.5,annot_kws={'size': 9})
        plt.title('Correlation Heatmap')
        plt.savefig(os.path.join(parent_dir, 'not_in_repo', file_name))

def make_and_save_barchart(whole_df, col_to_plot, file_name):
    mean_values_grouped = whole_df.groupby('nr_dgm')[col_to_plot].mean()

    # Plot mean values with grouped bars for each parameter
    x = np.arange(len(col_to_plot)) 
    width = 0.35 

    fig, ax = plt.subplots(figsize=(12, 6))
    bar1 = ax.bar(x - width/2, mean_values_grouped.iloc[0], width, label='nr_dgm = 1')
    bar2 = ax.bar(x + width/2, mean_values_grouped.iloc[1], width, label='nr_dgm = 2')

    ax.set_xlabel('Columns')
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Values of Specified Columns for nr_dgm=1 and nr_dgm=2')
    ax.set_xticks(x)
    ax.set_xticklabels(col_to_plot, rotation=45)
    ax.legend()

    plt.savefig()

def diff_between_dgm(whole_df, col_to_analyze):
    mean_values_grouped = whole_df.groupby('nr_dgm')[col_to_analyze].mean()
    percentage_difference = ((mean_values_grouped.iloc[1] - mean_values_grouped.iloc[0]) / mean_values_grouped.iloc[0]) * 100

    print("Percentage Difference in Mean Values (nr_dgm=2 - nr_dgm=1):")
    print(percentage_difference)

def make_box_plot(whole_df, column_name):

    fig = px.box(whole_df, y = column_name)
    fig.show()

    biggest_value_with_1_status = whole_df[whole_df['our_final_status'] == 1][column_name].max()
    print(biggest_value_with_1_status)

def describe_our_data(whole_df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(whole_df.describe().T)


def analyze_data(whole_df):
    #function take random data from dataframe and make pairplots and heatmap for all data
    status_1_data = whole_df[whole_df['our_final_status'] == 1].sample(n=200, random_state= 69)
    status_2_data = whole_df[whole_df['our_final_status'] == 2].sample(n=200, random_state= 69)
    random_to_analyze = pd.concat([status_1_data, status_2_data], ignore_index=True)

    col_dgm = ['czas_fazy_1', 'czas_fazy_2', 'czas_fazy_3', 'max_predkosc', 'cisnienie_tloka', 'cisnienie_koncowe','nachdruck_hub', 
                  'anguss', 'temp_pieca', 'oni_temp_curr_f1', 'oni_temp_curr_f2', 'oni_temp_fore_f1', 'oni_temp_fore_f2', 'vds_air_pressure',
                    'vds_vac_hose1', 'vds_vac_hose2', 'vds_vac_tank', 'vds_vac_valve1', 'vds_vac_valve2', 'czas_taktu']
    col_flow = [f'flow_{n}' for n in range(1,29)]
    col_delay = [f'start_delay_{n}' for n in range(1,29)]
    col_temp = [f'temp_{n}' for n in range(1,29)]

    make_and_save_pariplot(random_to_analyze, col_dgm, 'normalized_dirst_20_normalized.png')
    make_and_save_pariplot(random_to_analyze, col_flow, 'normalized_flow_normalized.png')
    make_and_save_pariplot(random_to_analyze, col_delay, 'normalize_delay_pairplot_normalized.png')
    make_and_save_pariplot(random_to_analyze, col_temp, 'normalize_temp_pairplot_normalized.png')
    make_and_save_heatmap(whole_df, col_dgm, 'cor1_hm.png')
    make_and_save_heatmap(whole_df, col_flow, 'flow_heatmap.png')
    make_and_save_heatmap(whole_df, col_delay, 'delay_heatmap.png')
    make_and_save_heatmap(whole_df, col_temp, 'temp_heatmap.png')
