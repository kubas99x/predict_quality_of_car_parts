import pandas as pd
import sqlalchemy
import copy
from sqlalchemy.exc import SQLAlchemyError

from db_queries import username, password, dsn, dbhostname, service_name, dbtables, querys
from table_functions import *

def read_data_from_database():
    print('Reading from database')
    data = {}

    try:
        sqlalchemy_engine="oracle+cx_oracle://"+username+":"+password+"@"+dbhostname+"/?service_name="+service_name
        engine = sqlalchemy.create_engine(sqlalchemy_engine, arraysize=1000)
        for table, query in zip(dbtables, querys):
            data.update({table: pd.read_sql(query, engine)})
            print(f'Table {table} read')
    except SQLAlchemyError as e:
        print(e)

    data = drop_unused_columns(data)
    
    return data

def prepare_data(table_data):

    print('Create final status')
    table_data = create_final_status(table_data) # nie usuwa już ID z tabelii
    print('Drop columns not used in ml')
    table_data = drop_columns_not_used_in_ml(table_data)
    print('Categorize data')
    table_data = categorize_data(table_data)

    return table_data

def make_set_for_dgm(all_data, from_dgm=8, to_dgm=10):

    dgm_all = copy.deepcopy(all_data)
    dgm_table = combine_final_table(dgm_all, dgm_smallest=from_dgm, dgm_biggest=to_dgm)
    
    dgm = prepare_data(dgm_table)

    print("Make test set from October")
    filtered_data_dgm = dgm[(dgm['data_odlania'].dt.month >= 10) & (dgm['data_odlania'].dt.year >= 2023)]
    dgm= dgm.iloc[:-int(filtered_data_dgm.shape[0])]

    dgm.drop(columns=['data_odlania','nr_dgm', 'id'], inplace = True)

    dgm, high_corr_features_dgm = drop_columns_with_too_much_corr(dgm)
    filtered_data_dgm = filtered_data_dgm.drop(columns= high_corr_features_dgm)

    save_df_to_csv(dgm, f'final_table_dgm{from_dgm}_{to_dgm}.csv')
    save_df_to_csv(filtered_data_dgm, f'test_dgm{from_dgm}_{to_dgm}_from_october.csv')

    ml_data_dgm = split_data(dgm, samples=(dgm['our_final_status'] == 0).sum())
    normalize_and_save_to_csv(ml_data_dgm, file_name_=f'dgm{from_dgm}_{to_dgm}')

if __name__ == '__main__':

    readFromDatabase = True
    final_table = None
    specific_dgm = True

    if readFromDatabase:
        data = read_data_from_database()

        if specific_dgm:
            make_set_for_dgm(data, 9, 9)
            make_set_for_dgm(data, 10, 10)
            # print("Combine tables")
            # data9 = copy.deepcopy(data)
            # dgm9 = combine_final_table(data9, dgm_smallest=9, dgm_biggest=9)
            # dgm10 = combine_final_table(data, dgm_smallest=10, dgm_biggest=10)
            
            # dgm9 = prepare_data(dgm9)
            # dgm10 = prepare_data(dgm10)

            # print("Make test set from October")
            # filtered_data_dgm9 = dgm9[(dgm9['data_odlania'].dt.month >= 10) & (dgm9['data_odlania'].dt.year >= 2023)]
            # dgm9= dgm9.iloc[:-int(filtered_data_dgm9.shape[0])]

            # filtered_data_dgm10 = dgm10[(dgm10['data_odlania'].dt.month >= 10) & (dgm10['data_odlania'].dt.year >= 2023)]
            # dgm10 = dgm10.iloc[:-int(filtered_data_dgm10.shape[0])]

            # dgm9.drop(columns=['data_odlania','nr_dgm', 'id'], inplace = True)
            # dgm10.drop(columns=['data_odlania','nr_dgm', 'id'], inplace = True)

            # dgm9, high_corr_features_dgm9 = drop_columns_with_too_much_corr(dgm9)
            # filtered_data_dgm9 = filtered_data_dgm9.drop(columns= high_corr_features_dgm9)

            # save_df_to_csv(dgm9, 'final_table_dgm9.csv')
            # save_df_to_csv(filtered_data_dgm9, 'test_dgm9_from_october.csv')

            # dgm10, high_corr_features_dgm10 = drop_columns_with_too_much_corr(dgm10)
            # filtered_data_dgm10 = filtered_data_dgm10.drop(columns= high_corr_features_dgm10)

            # save_df_to_csv(dgm10, 'final_table_dgm10.csv')
            # save_df_to_csv(filtered_data_dgm10, 'test_dgm10_from_october.csv')

            # ml_data_dgm9 = split_data(dgm9, samples = dgm9[[dgm9['our_final_status'] == 0]].shape[0])
            # normalize_and_save_to_csv(ml_data_dgm9, file_name_='dgm9')

            # ml_data_dgm10 = split_data(dgm10, samples = dgm10[[dgm10['our_final_status'] == 0]].shape[0])
            # normalize_and_save_to_csv(ml_data_dgm10, file_name_='dgm10')


        else:
            print('Combine final table')
            final_table = combine_final_table(data)

            print('Create final status')
            final_table = create_final_status(final_table) # nie usuwa już ID z tabelii

            print('Drop columns not used in ml')
            final_table = drop_columns_not_used_in_ml(final_table)

            print('Categorize data')
            final_table = categorize_data(final_table)

            print("Make test set from October")
            filtered_data = final_table[(final_table['data_odlania'].dt.month >= 10) & (final_table['data_odlania'].dt.year >= 2023)]
            final_table = final_table.iloc[:-int(filtered_data.shape[0])]

            save_df_to_csv(final_table, 'final_table_test_delete.csv')
            
            final_table.drop(columns=['data_odlania','nr_dgm', 'id'], inplace = True)
            #filtered_data.drop(columns='data_odlania', inplace = True)
            
            print('Drop columns with too much correlation')
            final_table, high_corr_features = drop_columns_with_too_much_corr(final_table)

            print('Drop columns in test data from October')
            filtered_data = filtered_data.drop(columns= high_corr_features)

            print('Save final table')
            save_df_to_csv(final_table, 'final_table_before_normalization.csv')

            print('Save test data from October')
            save_df_to_csv(filtered_data, 'test_data_from_october.csv')
        
    else:
        print('Reading CSV file')
        final_table = load_csv('final_table_before_normalization.csv')

    if not read_data_from_database or not specific_dgm:
        print('Split data')
        ml_data = split_data(final_table)

        print("Saving training data nn")
        normalize_and_save_to_csv(ml_data, file_name_='n_n_lwd')

    # print("Saving training data 0_1")
    # normalize_and_save_to_csv(ml_data, file_name_='0_1_lwd', normalize_type='0_1')

    # print("Saving training data s_s")
    # normalize_and_save_to_csv(ml_data, file_name_='s_s_lwd', normalize_type='standard')
    