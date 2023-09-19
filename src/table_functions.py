def drop_unused_columns(data):

    dbtables = ['MEB_DGM', 'MEB_DMC', 'MEB_GROB', 'MEB_KO', 'MEB_KO_DGM', 'MEB_KS']
    columns = [['timestamp', 'data_znakowania', 'data_odlania', 'metal_level', 'metal_pressure'],                                           #MEB_DGM
    ['timestamp', 'update_time','id_meb_containers', 'packed_time', 'first_packed_time', 'production_step', 'status_koncowy'],              #MEB_DMC
    ['id_meb_grob', 'shift_number', 'last_operation', 'timestamp', 'production_date', 'reworkrequested',                                    
    'reworkdone', 'partcleaningisfinished', 'waitfortoolcheck', 'workingstep1', 'workingstep2', 'workingstep3', 'workingstep4', 'mms_ok'],  #MEB_GROB
    ['id_ko', 'data', 'timestamp', 'eks'],                                                                                                  #MEB_KO
    ['id_ko','data_odlania', 'timestamp', 'operator'],                                                                                      #MEB_KO_DGM
    ['id_ks', 'nrgniazda', 'liczbawystapien', 'nrformy', 'data', 'timestamp', 'gradedmc_max','gradedmc_aktualny']]                          #MEB_KS

    for table, column in zip(dbtables, columns):
        data[table].drop(columns=column, inplace=True)
    
    return data

def combine_final_table(data):

    # usuwanie znaków białych z DMC[MEB_DGM] i DMC_CASTING[MEB_DMC]
    data['MEB_DMC'].dmc_casting = data['MEB_DMC']['dmc_casting'].str.strip()
    data['MEB_DGM'].dmc = data['MEB_DGM']['dmc'].str.strip()

    # usuwanie z meb_dmc wierszy z 'WORKPIECE NIO' w kodzie DMC
    data['MEB_DMC'] = data['MEB_DMC'][~data['MEB_DMC']['dmc'].str.contains('WORKPIECE', case=False, na=False)]

    # wybieranie rekordów dla MEB+ 
    data['MEB_DGM'] = data['MEB_DGM'][(data['MEB_DGM']['nr_dgm'].between(8, 10)) & (data['MEB_DGM']['dmc'].apply(lambda x: len(x)) == 21)]
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
    oni_circuits = data['ONI_CIRCUITS'].pivot(index='id_dmc', columns='circuit_nr', values=['assigment', 'flow', 'set_point', 'start_delay', 'working_mode', 'temp'])
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

    final_table = final_table[~final_table['status'].isin(['4', '5', '7', '8', '10', '11'])]
    final_table['status'] = final_table['status'].replace(['3', '14'], '2')
    final_table = final_table.loc[~final_table['status_ko'].isin([0, 106])]
    final_table = final_table.loc[~final_table['statusszczelnosc'].isin([0, 3])]
    final_table = final_table.loc[~final_table['statusdmc'].isin([0])]

    final_table['our_final_status'] = final_table.apply(lambda row: max(int(row['status']), row['status_ko'], row['statusszczelnosc'], row['statusdmc']), axis=1)
    print(final_table['our_final_status'].value_counts())
    final_table.drop(columns=['status', 'status_ko', 'statusszczelnosc', 'statusdmc'], inplace=True)

    return final_table