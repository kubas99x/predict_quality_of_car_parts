import os
def make_and_save_pariplot(whole_df ,columns_to_analyze, file_name):

    plot = sns.pairplot(data=whole_df[columns_to_analyze + ['our_final_status']], hue="our_final_status", palette={ 1 : "green", 2:"red"}, plot_kws={"s": 3})

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    plot.savefig(os.path.join(parent_dir, 'not_in_repo', file_name))