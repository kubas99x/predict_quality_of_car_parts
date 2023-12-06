import pandas as pd
import sqlalchemy
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


if __name__ == '__main__':

    readFromDatabase = True
    final_table = None

    if readFromDatabase:
        data = read_data_from_database()

        print('Combine final table')
        final_table = combine_final_table(data)

        print('Create final status')
        final_table = create_final_status(final_table)

        print('Drop columns not used in ml')
        final_table = drop_columns_not_used_in_ml(final_table)

        print('Categorize data')
        final_table = categorize_data(final_table)

        print('Drop columns with too much correlation')
        final_table = drop_columns_with_too_much_corr(final_table)

        print('Save final table')
        save_df_to_csv(final_table, 'final_table_before_normalization.csv')
        
    else:
        final_table = load_csv('final_table_before_normalization.csv')

    print('Split data')
    ml_data = split_data(final_table)

    print("Saving training data nn")
    normalize_and_save_to_csv(ml_data, file_name_='n_n')

    print("Saving training data 0_1")
    normalize_and_save_to_csv(ml_data, file_name_='0_1', normalize_type='0_1')

    print("Saving training data s_s")
    normalize_and_save_to_csv(ml_data, file_name_='s_s', normalize_type='standard')
    