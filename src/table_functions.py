from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
#from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import pandas as pd
import os

def drop_unused_columns(data):

    dbtables = ['MEB_DGM', 'MEB_DMC', 'MEB_GROB', 'MEB_KO', 'MEB_KO_DGM', 'MEB_KS']
    columns = [['timestamp','data_znakowania', 'metal_level', 'metal_pressure', 'max_press_kolbenhub', 'oni_temp_curr_f2'],                                     #MEB_DGM
    ['timestamp', 'update_time','id_meb_containers', 'packed_time', 'first_packed_time', 'production_step', 'status_koncowy'],                                  #MEB_DMC
    ['id_meb_grob', 'shift_number', 'last_operation', 'timestamp', 'production_date', 'reworkrequested',                                    
    'reworkdone', 'partcleaningisfinished', 'waitfortoolcheck', 'workingstep1', 'workingstep2', 
    'workingstep3', 'workingstep4', 'mms_ok', 'last_machine_number', 'last_pcf_number', 'machine_nr'],                                                          #MEB_GROB
    ['id_ko', 'data', 'timestamp', 'eks'],                                                                                                                      #MEB_KO
    ['id_ko','data_odlania', 'timestamp', 'operator'],                                                                                                          #MEB_KO_DGM
    ['id_ks', 'nrgniazda', 'liczbawystapien', 'nrformy', 'data', 'timestamp', 'gradedmc_max','gradedmc_aktualny']]                                              #MEB_KS

    for table, column in zip(dbtables, columns):
        data[table].drop(columns=column, inplace=True)
    
    return data

def combine_final_table(data_, dgm_smallest = 8, dgm_biggest = 10):

    data = data_.copy()
    # usuwanie znaków białych z DMC[MEB_DGM] i DMC_CASTING[MEB_DMC]
    data['MEB_DMC'].dmc_casting = data['MEB_DMC']['dmc_casting'].str.strip()
    data['MEB_DGM'].dmc = data['MEB_DGM']['dmc'].str.strip()

    # usuwanie z meb_dmc wierszy z 'WORKPIECE NIO' w kodzie DMC
    data['MEB_DMC'] = data['MEB_DMC'][~data['MEB_DMC']['dmc'].str.contains('WORKPIECE', case=False, na=False)]

    # wybieranie rekordów dla MEB+ 
    data['MEB_DGM'] = data['MEB_DGM'][(data['MEB_DGM']['nr_dgm'].between(dgm_smallest, dgm_biggest)) & (data['MEB_DGM']['dmc'].apply(lambda x: len(str(x)) == 21))]
    # usunięcie anomalii z MEB_DMC
    data['MEB_DMC'] = data['MEB_DMC'][data['MEB_DMC']['dmc'].str[:3] == '0MH']

    # łączę tabele MEB_KO i MEB_KO_DGM z tabelami MEB_KO_STREFA/RODZAJ
    data['MEB_KO'] = data['MEB_KO'].merge(data['MEB_KO_STREFA'], left_on='nok_strefa', right_on='indeks', how='inner')
    data['MEB_KO'].drop(columns=['indeks'], inplace=True)
    data['MEB_KO'] = data['MEB_KO'].merge(data['MEB_KO_RODZAJ'], left_on='nok_rodzaj', right_on='indeks', how='inner')
    data['MEB_KO'].drop(columns=['indeks'], inplace=True)
    data['MEB_KO_DGM'] = data['MEB_KO_DGM'].merge(data['MEB_KO_STREFA'], left_on='nok_strefa', right_on='indeks', how='inner')
    data['MEB_KO_DGM'].drop(columns=['indeks'], inplace=True)
    data['MEB_KO_DGM'] = data['MEB_KO_DGM'].merge(data['MEB_KO_RODZAJ'], left_on='nok_rodzaj', right_on='indeks', how='inner')
    data['MEB_KO_DGM'].drop(columns=['indeks'], inplace=True)

    # łączę tabelę MEB_DMC z tabelą MEB_KO
    data['MEB_DMC'] = data['MEB_DMC'].merge(data['MEB_KO'], on='id_dmc', how='left')
    data['MEB_DMC'].drop(columns=['rn'], inplace=True)

    # łączę tabelę MEB_DMC z tabelą MEB_GROB
    data['MEB_DMC'] = data['MEB_DMC'].merge(data['MEB_GROB'], on='id_dmc', how='left')
    data['MEB_DMC'].drop(columns=['rn'], inplace=True)

    # łączę tabelę MEB_DMC z tabelą MEB_KS
    data['MEB_DMC'] = data['MEB_DMC'].merge(data['MEB_KS'], on='id_dmc', how='left')
    data['MEB_DMC'].drop(columns=['rn'], inplace=True)

    # przygotowywuję tabelę ONI_CIRCUITS do połączenia
    data['ONI_CIRCUITS'].drop(columns = ['assigment', 'working_mode', 'set_point'], inplace = True)
    oni_circuits = data['ONI_CIRCUITS'].pivot(index='id_dmc', columns='circuit_nr', values=['flow', 'start_delay', 'temp'])
    oni_circuits.columns = oni_circuits.columns.map('{0[0]}_{0[1]}'.format) 
    oni_circuits.reset_index(inplace=True)

    final_table = data['MEB_DGM'].copy()
    final_table.drop(columns=['rn'], inplace=True)

    # łączę tabelę MEB_DGM z tabelą MEB_KO_DGM
    final_table = final_table.merge(data['MEB_KO_DGM'], left_on='id', right_on='id_dmc', how='left')
    final_table.drop(columns=['rn'], inplace=True)

    # łączę z tabelą MEB_DGM
    final_table = final_table.merge(oni_circuits, left_on='id', right_on='id_dmc', how='inner')
    final_table.drop(columns=['id_dmc_y'], inplace=True)
    final_table.rename(columns={'id_dmc_x': 'id_dmc'}, inplace=True)

    # łączę tabelę MEB_DMC z ONI_CIRCUITS
    final_table = final_table.merge(data['MEB_DMC'], left_on='dmc', right_on='dmc_casting', how='left', suffixes=('_DGM', '_DMC'))

    # duplicate_count_oni = final_table['dmc_DMC'].duplicated(keep=False).sum()
    # print(f"Number of rows with the same 'dmc' value: {duplicate_count_oni}")

    final_table.drop(columns=['nok_strefa_DGM', 'nok_rodzaj_DGM', 'status_ko_DGM', 'kod_pola_DGM', 'rodzaj_uszkodzenia_DGM'], inplace=True)
    final_table.rename(columns={'nok_strefa_DMC': 'nok_strefa', 'nok_rodzaj_DMC': 'nok_rodzaj', 
                                'status_ko_DMC': 'status_ko', 'kod_pola_DMC': 'kod_pola', 
                                'rodzaj_uszkodzenia_DMC': 'rodzaj_uszkodzenia'}, inplace=True)
                                
    final_table.drop(index=final_table[(final_table['dmc_DGM'].duplicated(keep=False)) & (~final_table['dmc_casting'].isna())].index, inplace=True)
    final_table.drop(columns = ['part_status'], inplace = True)

    return final_table

def create_final_status(final_table):
    # statusy dmc 2 zostały całkowicie wywalone (jest ich ok. 450)
    # co do statusu szczelności to czasami na to wpływ ma porowatość wynikająca z odlewania,
    # jednak jest dużo błędów wynikających z obróbki czy zepsutej uszczelki
#     status
# 1     687715
# 10     11808      # poczatkowe wtryski na rozgrzanie
# 5       8340
# 11      6989
# 3       6337      # zostawilem -  NIO DGM
# 7       3132
# 14      2940      # zostawilem - NIO prozni
# 4       2698
# 8        595

    final_table = final_table[~final_table['status'].isin(['4', '5', '7', '8', '10', '11'])]
    final_table['status'] = final_table['status'].replace(['3', '14'], '2')

    final_table = final_table.loc[~final_table['status_ko'].isin([0, 106])] # KO
    final_table = final_table.loc[~final_table['statusszczelnosc'].isin([0, 3])]
    final_table = final_table.loc[~final_table['statusdmc'].isin([0,2])]

    final_table = final_table.loc[final_table['nok_rodzaj'].isin([0, 102, 201, 103, 101])]
    final_table['nok_rodzaj'] = final_table['nok_rodzaj'].replace([102, 201, 103, 101], 2)
    final_table['nok_rodzaj'] = final_table['nok_rodzaj'].replace([0], 1)

    final_table['our_final_status'] = final_table.apply(lambda row: max(int(row['status']), row['nok_rodzaj'], row['statusszczelnosc'], row['statusdmc']), axis=1)

    print(f"Final number of NOK parts: {final_table['our_final_status'].value_counts()}")

    final_table.drop(columns=['status', 'status_ko', 'statusszczelnosc', 'statusdmc', 
                              'part_type', 'nrprogramu', 'id_dmc_DGM', 
                              'id_dmc_DGM', 'dmc_DGM', 'product_id', 'line_id', 
                              'dmc_DMC', 'dmc_casting', 'nok_strefa', 'nok_rodzaj'], inplace=True)  # 'nr_dgm' na razie nie kasuje bo testuje dane - JR 25.09

    return final_table

def create_final_status_2(final_table):
    # statusy dmc 2 zostały całkowicie wywalone (jest ich ok. 450)
    # co do statusu szczelności to czasami na to wpływ ma porowatość wynikająca z odlewania,
    # jednak jest dużo błędów wynikających z obróbki czy zepsutej uszczelki
    # nok_rodzaj

    final_table = final_table[final_table['status'].isin(['1'])]

    final_table = final_table.loc[~final_table['status_ko'].isin([0, 106])] # 'status_ko_DMC'
    final_table = final_table.loc[~final_table['statusszczelnosc'].isin([0, 3])]
    
    final_table = final_table.loc[~final_table['statusdmc'].isin([0,2])]

    print(f'Number of NOK parts of DGM on KO: {final_table["nok_rodzaj"].isin([102, 201, 103, 101]).sum()}')

    final_table = final_table.loc[final_table['nok_rodzaj'].isin([0, 102, 201, 103, 101])]
    final_table['nok_rodzaj'] = final_table['nok_rodzaj'].replace([102, 201, 103, 101], 2)
    final_table['nok_rodzaj'] = final_table['nok_rodzaj'].replace([0], 1)

    final_table['our_final_status'] = final_table.apply(lambda row: max(row['nok_rodzaj'], row['statusszczelnosc'], row['statusdmc']), axis=1)
    
    print(f'Final number of NOK parts: {final_table["our_final_status"].value_counts()}')
    
    final_table.drop(columns=['status', 'status_ko', 'statusszczelnosc', 'statusdmc', 
                              'part_type', 'nrprogramu', 'id_dmc_DGM', 
                              'id_dmc_DGM', 'dmc_DGM', 'product_id', 'line_id', 
                              'dmc_DMC', 'dmc_casting', 'nok_strefa', 'nok_rodzaj'], inplace=True)  # 'nr_dgm' na razie nie kasuje bo testuje dane - JR 25.09

    return final_table

def standarize_data(final_table):

    # Ograniczamy wartość maksymalną danych do określonego limitu, by w mniejszym stopniu wpływało to na normalizacje

    final_table.loc[final_table['nachdruck_hub'] > 1000, 'nachdruck_hub'] = 1000
    final_table.loc[final_table['czas_fazy_1'] > 5000, 'czas_fazy_1'] = 5000
    final_table.loc[final_table['czas_fazy_3'] > 3000, 'czas_fazy_3'] = 3000
    final_table.loc[final_table['anguss'] > 800, 'anguss'] = 800
    final_table.loc[final_table['vds_vac_hose1'] > 1000, 'vds_vac_hose1'] = 1000

    return final_table

def categorize_data(whole_df):
    final_table = whole_df.copy()
    # categorical_columns = []
    # for name in ['assigment', 'working_mode']:
    #     for x in range(1,29):
    #         categorical_columns.append(f'{name}_{x}')

    # categorical_data = final_table[categorical_columns].astype('category')
    # categorical_data = pd.get_dummies(categorical_data, drop_first=True, dtype=int)
    
    
    final_table['our_final_status'] = final_table['our_final_status'].astype(int) - 1
    final_table['our_final_status'] = final_table['our_final_status'].astype('category')

    #final_table.drop(columns=categorical_columns, inplace=True)
    #final_table = pd.concat([final_table, categorical_data], axis=1)

    return final_table

def normalize_data(whole_df, scaler=None):

    final_table = whole_df.copy()
    categorical_columns_ = [value for value in final_table if value.startswith('assigment') or value.startswith('working')]
    categorical_data = final_table[categorical_columns_]
    final_table.drop(columns=categorical_columns_, inplace=True)

    if not scaler:
        scaler = StandardScaler()
        final_table[final_table.columns] = scaler.fit_transform(final_table[final_table.columns])
        final_table = pd.concat([final_table, categorical_data], axis=1)
        return final_table, scaler

    final_table[final_table.columns] = scaler.transform(final_table[final_table.columns])
    final_table = pd.concat([final_table, categorical_data], axis=1)

    return final_table

def normalize_0_1(whole_df, scaler=None):
    whole_table = whole_df.copy()
    categorical_columns_ = [value for value in whole_table if value.startswith('assigment') or value.startswith('working')]
    categorical_data = whole_table[categorical_columns_]
    whole_table.drop(columns=categorical_columns_, inplace=True)

    if not scaler:
        scaler = MinMaxScaler()
        whole_table[whole_table.columns] = scaler.fit_transform(whole_table[whole_table.columns])
        whole_table = pd.concat([whole_table, categorical_data], axis=1)
        return whole_table, scaler

    whole_table[whole_table.columns] = scaler.transform(whole_table[whole_table.columns])
    whole_table = pd.concat([whole_table, categorical_data], axis=1)

    return whole_table

def normalize_and_save_to_csv(ml_data, file_name_, normalize_type = 'None'):
    ml_data_ = ml_data.copy()
    data_keys = ['x_train', 'x_valid', 'x_test', #'x_anomalies',
                 'y_train', 'y_valid', 'y_test'] #, 'y_anomalies']
    if normalize_type == '0_1':
        ml_data_['x_train'], scaler = normalize_0_1(ml_data_['x_train'])
        ml_data_['x_valid'] = normalize_0_1(ml_data_['x_valid'], scaler)
        ml_data_['x_test'] = normalize_0_1(ml_data_['x_test'], scaler)
        #ml_data_['x_anomalies'] = normalize_0_1(ml_data_['x_anomalies'], scaler)
    elif normalize_type == 'standard':
        ml_data_['x_train'], scaler = normalize_data(ml_data_['x_train'])
        ml_data_['x_valid'] = normalize_data(ml_data_['x_valid'], scaler)
        ml_data_['x_test'] = normalize_data(ml_data_['x_test'], scaler)
        #ml_data_['x_anomalies'] = normalize_data(ml_data_['x_anomalies'], scaler)

    for key_ in data_keys:
        save_df_to_csv(ml_data_[key_], f'{key_}_{file_name_}.csv')

def save_df_to_csv(dat_, file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dat_.to_csv(os.path.join(current_dir, '.data', file_name).replace("\\","/"), index= False)

def load_csv(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print('File to read:')
    print(os.path.join(current_dir, '.data', file_name).replace("\\","/"))
    df = pd.read_csv(os.path.join(current_dir, '.data', file_name).replace("\\","/"))
    return df

def distinct_machine(final_table):
    final_table_9 = final_table[final_table['nr_dgm'] == 9]
    final_table_10 = final_table[final_table['nr_dgm'] == 10]

    return final_table_9, final_table_10

def drop_columns_not_used_in_ml(final_table):
    whole_df = final_table.copy()
    columns = ['rodzaj_kontroli', 'id_dmc_DMC', 'kod_pola', 'rodzaj_uszkodzenia', 
            'temp_workpiece', 'temp_hydraulics', 'pressure_pcf_1', 'pressure_pcf_2', 
            'pressure_pcf_3', 'cisnienie', 'przeciek', 'temperaturatestu', 'temp_pieca']
    whole_df.drop(columns=columns, inplace=True)

    return whole_df

def drop_columns_with_too_much_corr(final_table, corrTreshold = 0.9):
    whole_df = final_table.copy()
    correlation_matrix = whole_df.corr()
    high_corr_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > corrTreshold:
                colname = correlation_matrix.columns[i]
                high_corr_features.add(colname)

    print(f'Features with correlation > {corrTreshold} : {len(high_corr_features)}')
    final_table_droped = whole_df.drop(columns = high_corr_features)

    return final_table_droped, high_corr_features

def read_data_for_traning(fileName):
    data_keys = ['x_train', 'x_valid', 'x_test', 'y_train', 'y_valid', 'y_test']#, 'y_anomalies',  'x_anomalies']
    ml_data_ = {key: None for key in data_keys}
    for key in ml_data_:
        ml_data_[key] = load_csv(f'{key}_{fileName}.csv')
    
    return ml_data_

def apply_lof(whole_df, n):

    target = whole_df.pop('our_final_status')

    lof = LocalOutlierFactor(n_neighbors=n)
    whole_df['is_outlier'] = lof.fit_predict(whole_df)
    print(f"Amount of outliers: {whole_df[whole_df['is_outlier'] == -1]['is_outlier'].count()}")
    print('\n')

    whole_df['our_final_status'] = target
    print(f"{whole_df['our_final_status'].value_counts()}")
    x_anomalies = whole_df[whole_df['is_outlier'] == -1]
    x_anomalies.drop(columns=['is_outlier'], inplace=True)
    y_anomalies = x_anomalies.pop('our_final_status')

    whole_df = whole_df[whole_df['is_outlier'] != -1]
    whole_df.drop(columns=['is_outlier'], inplace=True)
    target = whole_df.pop('our_final_status')

    return whole_df, target, x_anomalies, y_anomalies

def split_data(final_table, train_set_size=0.80, samples = 100000):
    
    # do modelowania:
    'czas_fazy_1', 'czas_fazy_2', 'czas_fazy_3', 'max_predkosc', 'cisnienie_tloka', 'cisnienie_koncowe', 'nachdruck_hub', 'anguss', 'oni_temp_curr_f1', 
    'oni_temp_curr_f2', 'oni_temp_fore_f1', 'oni_temp_fore_f2', 'vds_air_pressure', 'vds_vac_hose1', 'vds_vac_hose2', 'vds_vac_tank', 'vds_vac_valve1', 'vds_vac_valve2', 'czas_taktu',
    #assigment_1-28, flow_1-28, set_point_1-28, start_delay_1-28. temp_1-28, working_mode_1-28

    # nie do modelowania
    'nr_dgm', 'rodzaj_kontroli', 'id_dmc_DMC', 'kod_pola', 'rodzaj_uszkodzenia', 'temp_workpiece', 'temp_hydraulics', 'pressure_pcf_1', 'pressure_pcf_2', 'pressure_pcf_3', 
    'cisnienie', 'przeciek', 'temperaturatestu', 'temp_pieca'

    #nasz parametr klasy:
    'our_final_status'
    whole_df = final_table.copy()
    target = whole_df.pop('our_final_status')

    #print('Detecting and removing outliers:')
    #whole_df, target, x_anomalies, y_anomalies = apply_lof(whole_df, n=n_neighbors)
    
    x_train, x_test, y_train, y_test = train_test_split(whole_df, target, train_size=train_set_size, random_state=42, stratify=target)
    x_train, y_train = over_under_sampling(x_train, y_train, samples)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=42, stratify=y_test)
    #x_valid, y_valid = over_under_sampling(x_valid, y_valid, ok_samples=10000)
   

    return {'x_train' : x_train, 'x_valid' : x_valid, 'x_test' : x_test, #'x_anomalies': x_anomalies, 
            'y_train' : y_train, 'y_valid' : y_valid, 'y_test' : y_test} #, 'y_anomalies': y_anomalies}
            
    # return x_train, x_valid, x_test, y_train, y_valid, y_test

def over_under_sampling(data, target, samples):

    data_copy = data.copy()
    target_copy = target.copy()

    df = pd.concat([data_copy, target_copy], axis=1)

    data_copy_ok = df[df['our_final_status'] == 0]
    data_copy_ok = data_copy_ok.sample(n=samples, replace=True, axis=0)

    data_copy_nok = df[df['our_final_status'] == 1]
    data_copy_nok = data_copy_nok.sample(n=samples, axis=0, replace=True)
    
    data = pd.concat([data_copy_ok, data_copy_nok], axis=0)
    target = data.pop('our_final_status')
    
    print('Amount of classes:')
    print(target.value_counts())
    return data, target

def return_x_y_with_specific_status(x_data, y_data, status = 1):

    test_data = pd.concat([x_data, y_data], axis = 1)
    test_data = test_data[test_data['our_final_status']== status]
    y_test_one_status = test_data.pop('our_final_status')
    
    return test_data, y_test_one_status

def return_first_x_rows(x_train_, x_valid_, x_test_, number_of_rows):

    x_train_ = x_train_.iloc[:, :number_of_rows].copy()
    x_test_ = x_test_.iloc[:, :number_of_rows].copy()
    x_valid_ = x_valid_.iloc[:, :number_of_rows].copy()

    return x_train_, x_valid_, x_test_