import os
import seaborn as sns
import matplotlib.pyplot as plt

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