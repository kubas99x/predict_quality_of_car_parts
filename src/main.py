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

def make_set_for_dgm(all_data, version_of_status, from_dgm=8, to_dgm=10, start_year= 2021):

    print(f"Making set for dgm {from_dgm} to {to_dgm}, status version: {version_of_status}, from year: {start_year}")
    dgm_all = copy.deepcopy(all_data)
    dgm_table = combine_final_table(dgm_all, dgm_smallest=from_dgm, dgm_biggest=to_dgm)
    
    dgm_table['data_odlania'] = pd.to_datetime(dgm_table['data_odlania'])
    if start_year > 2021:
        dgm_table = dgm_table[dgm_table['data_odlania'].dt.year >= start_year]

    print('Create final status')
    if version_of_status == '1':
        print("VERSION 1")
        dgm_table = create_final_status(dgm_table, version_of_status)
    elif version_of_status == '2': 
        print("VERSION 2")
        dgm_table = create_final_status(dgm_table, version_of_status)

    print('Drop columns not used in ml')
    dgm_table= drop_columns_not_used_in_ml(dgm_table)
    print('Categorize data')
    dgm = categorize_data(dgm_table)

    print("Make test set from October")
    filtered_data_dgm = dgm[(dgm['data_odlania'].dt.month >= 10) & (dgm['data_odlania'].dt.year >= 2023)]
    dgm= dgm.iloc[:-int(filtered_data_dgm.shape[0])]

    dgm.drop(columns=['data_odlania','nr_dgm', 'id'], inplace = True)

    dgm, high_corr_features_dgm = drop_columns_with_too_much_corr(dgm)
    filtered_data_dgm = filtered_data_dgm.drop(columns= high_corr_features_dgm)

    save_df_to_csv(dgm, f'final_table_{from_dgm}_{to_dgm}_v{version_of_status}_{start_year}.csv')
    save_df_to_csv(filtered_data_dgm, f'test_{from_dgm}_{to_dgm}_from_october_v{version_of_status}_{start_year}.csv')

    ml_data_dgm = split_data(dgm, samples=(dgm['our_final_status'] == 1).sum() * 2)
    normalize_and_save_to_csv(ml_data_dgm, file_name_=f'dgm{from_dgm}_{to_dgm}_v{version_of_status}_{start_year}')

if __name__ == '__main__':

    data = read_data_from_database()

    make_set_for_dgm(data, '1', 9, 10)
    make_set_for_dgm(data, '1', 9, 10, start_year= 2023)
    make_set_for_dgm(data, '2', 9, 10)
    make_set_for_dgm(data, '2', 9, 10, start_year= 2023)

    make_set_for_dgm(data, '1', 9, 9)
    make_set_for_dgm(data, '1', 9, 9, start_year= 2023)
    make_set_for_dgm(data, '2', 9, 9)
    make_set_for_dgm(data, '2', 9, 9, start_year= 2023)

    make_set_for_dgm(data, '1', 10, 10)
    make_set_for_dgm(data, '1', 10, 10, start_year= 2023)
    make_set_for_dgm(data, '2', 10, 10)
    make_set_for_dgm(data, '2', 10, 10, start_year= 2023)


    