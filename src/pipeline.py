import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
from ml_functions import *
from creating_datasets import USERNAME, PASSWORD, DBHOSTNAME, SERVICE_NAME
from table_functions import *
import xgboost as xgb
import time

# DGM id - oni id_dmc
# DGM dmc - MEB_DMC dmc_casting

def read_last_meb_dgm(last_id = 0):

    data = {}
    if last_id:
        query = f"""SELECT *
            FROM (
                SELECT
                    t.*,
                    ROW_NUMBER() OVER (PARTITION BY DMC ORDER BY ID DESC) AS rn
                FROM
                    Z3DMC.MEB_DGM t
            ) subquery
            WHERE rn = 1
            AND id > {last_id}"""

    else:
        query = """SELECT *
            FROM (
                SELECT *
                FROM Z3DMC.MEB_DGM
                ORDER BY ID DESC
            )
            WHERE ROWNUM = 1
            """
    
    try:
        sqlalchemy_engine="oracle+cx_oracle://"+USERNAME+":"+PASSWORD+"@"+DBHOSTNAME+"/?service_name="+SERVICE_NAME
        engine = sqlalchemy.create_engine(sqlalchemy_engine, arraysize=1000)
        data.update({'MEB_DGM': pd.read_sql(query, engine)})
    except SQLAlchemyError as e:
        print(e)
    
    data['MEB_DGM'].drop(columns=['timestamp','data_znakowania','data_odlania', 'metal_level', 'metal_pressure', 'max_press_kolbenhub', 'oni_temp_curr_f2'], inplace= True)

    last_id = data['MEB_DGM'].id.max()
    
    return data, last_id

def check_if_meb_base(data):
    data['MEB_DGM'].dmc = data['MEB_DGM']['dmc'].str.strip()
    data['MEB_DGM'] = data['MEB_DGM'][(data['MEB_DGM']['nr_dgm'].between(8, 10)) & (data['MEB_DGM']['dmc'].apply(lambda x: len(str(x)) == 21))]

    if data['MEB_DGM'].empty:
        print('There are not MEB_BASE+ part produced since last time')
        return False
    else:
        return data

def read_oni(data):
    id_list = list(data['MEB_DGM'].id)
    ids_ranges = [id_list[x:x+500] for x in range(0, len(id_list), 500)]
    ids_ranges_tuples = [tuple(sublist) for sublist in ids_ranges]

    result_df = pd.DataFrame()
    try:
        sqlalchemy_engine="oracle+cx_oracle://"+USERNAME+":"+PASSWORD+"@"+DBHOSTNAME+"/?service_name="+SERVICE_NAME
        engine = sqlalchemy.create_engine(sqlalchemy_engine, arraysize=1000)

        for ids in ids_ranges_tuples:
            query = f"""SELECT ID_DMC, CIRCUIT_NR, 
                    MAX(ASSIGMENT) AS ASSIGMENT, 
                    MAX(FLOW) AS FLOW, 
                    MAX(SET_POINT) AS SET_POINT,
                    MAX(START_DELAY) AS START_DELAY,
                    MAX(TEMP) AS TEMP,
                    MAX(WORKING_MODE) AS WORKING_MODE
                FROM Z3DMC.ONI_CIRCUITS
                WHERE ID_DMC IN {ids}
                GROUP BY ID_DMC, CIRCUIT_NR
                ORDER BY ID_DMC
                """
    
            df = pd.read_sql(query, engine)
            if result_df.empty:
                result_df = df
            else:
                result_df = pd.concat([result_df, df], ignore_index=True)
            #result_df = pd.concat([result_df, df], ignore_index=True)

            
    except SQLAlchemyError as e:
        print(e)

    data.update({'ONI_CIRCUITS': result_df})
    
    return data

def combine_into_one_table(data):
    data['ONI_CIRCUITS'].drop(columns = ['assigment', 'working_mode', 'set_point'], inplace = True)
    oni_circuits = data['ONI_CIRCUITS'].pivot(index='id_dmc', columns='circuit_nr', values=['flow', 'start_delay', 'temp'])
    oni_circuits.columns = oni_circuits.columns.map('{0[0]}_{0[1]}'.format) 
    oni_circuits.reset_index(inplace=True)
    final_table = data['MEB_DGM'].copy()
    final_table = final_table.merge(oni_circuits, left_on='id', right_on='id_dmc', how='inner')

    return final_table

def save_id_to_file(value, filename=r'src/pipeline_files/id.txt'):
    with open(filename, 'w') as file:
        file.write(str(value))

def read_id_from_file(filename=r'src/pipeline_files/id.txt'):
    try:
        with open(filename, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None
    
def delete_unused_columns(columns_needed, data):
            
    columns_to_drop = data.columns.difference(columns_needed)
    data = data.drop(columns=columns_to_drop)

    return data

def predict_status(model_, data, threshold = 0.9):

    dmatrix = xgb.DMatrix(data)
    predictions = model_.predict(dmatrix)
    y_pred = np.where(predictions < threshold, 0, 1)
    print(y_pred)

    return y_pred

def stats_about_predictions(predictions_):

    print(f"Number of checked parts in this part: {len(predictions_)}")
    print(f"Number of NOK parts: {np.sum(predictions_ == 1)}")
    print(f"Percent of NOK parts in this part: {np.sum(predictions_ == 1) / len(predictions_)}")


if __name__ == '__main__':

    TEST = True
    time_between_database_read = 1 # in minutes

    print('Reading latest id, model and column names')
    newest_id = read_id_from_file()
    columns_needed = pd.read_csv(r'src\pipeline_files\column_names.csv', header=None)[0].tolist()
    model = xgb.Booster(model_file=r'C:\Users\DLXPMX8\Desktop\Projekt_AI\meb_process_data_analysis\src\final_model\model\model.xgb')
    
    while True:
        
        if TEST:
            dgm, newest_id = read_last_meb_dgm(1474000)
            TEST = False
        else:
            print("Reading data from database")
            dgm, newest_id = read_last_meb_dgm(newest_id)

        print("Data loaded, saving newest id")
        save_id_to_file(newest_id)
        
        print("Prepering the data")
        dgm = check_if_meb_base(dgm)
        if dgm:
            print("Reading ONI table")
            dgm_oni = read_oni(dgm)
            final_tab = combine_into_one_table(dgm_oni)
            final_tab = delete_unused_columns(columns_needed, final_tab)

            print("Predict statues of parts")
            predictions = predict_status(model, final_tab)

            print("Stats about predicted parts:")
            stats_about_predictions(predictions)

        print(f"Wait {time_between_database_read * 60} seconds for next database check")
        time.sleep(time_between_database_read * 60)



        