from sqlalchemy import create_engine
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
import re
import math 
import logging
import time
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os


# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)
# warning 무시
# warnings.filterwarnings(action='ignore')

#이상치 판단 데이터 선언
select_tag = {
            'STN1' : ['ANALOG', ['tag_40', 'tag_41', 'tag_42', 'tag_43', 'tag_44', 'tag_45']]
            , 'RCS31' : ['ANALOG', ['tag_10', 'tag_26']]
            , 'RCS32' : ['ANALOG', ['tag_9', 'tag_25', 'tag_14', 'tag_15', 'tag_16', 'tag_17', 'tag_12', 'tag_11', 'tag_13'
                                    , 'tag_31', 'tag_32', 'tag_33', 'tag_28', 'tag_27', 'tag_29']]
            , 'RCS33' : ['ANALOG', ['tag_8', 'tag_24', 'tag_30']]
            , 'RCS13' : ['AI', ['tag_13', 'tag_16']]
            , 'ANA3' : ['IN', ['tag_1', 'tag_2', 'tag_3', 'tag_4']]
            , 'RCS34' : ['ANALOG', ['tag_7', 'tag_6', 'tag_23', 'tag_22']]
            , 'RCS14' : ['ANALOG', ['tag_1', 'tag_2']]
            , 'STN5' : ['ANALOG', ['tag_88', 'tag_89']]
            , 'RCS35' : ['ANALOG', ['tag_2', 'tag_5', 'tag_19', 'tag_1', 'tag_18']]
        }

# 상한, 하한 적용 데이터 선언
# rcs_list = ['STN', 'STN', 'RCS3', 'RCS1', 'RCS1', 'RCS1', 'ETC', 'ANA']
rcs_list = ['STN', 'RCS3']
# group_list = ['AO', 'ANALOG', 'ANALOG', 'ELEC', 'ANALOG', 'AI', 'AI', 'IN']
group_list = ['ANALOG', 'ANALOG']
tag_list = [
    ['tag_40', 'tag_41', 'tag_42', 'tag_43', 'tag_44', 'tag_45', 'tag_88', 'tag_89']
    , ['tag_10', 'tag_11', 'tag_12', 'tag_13', 'tag_14', 'tag_15', 'tag_16', 'tag_17'
       , 'tag_19', 'tag_2', 
#       'tag_22', 'tag_23', 'tag_24', 'tag_25', 
       'tag_26', 'tag_27', 'tag_28', 'tag_29'
       , 'tag_30', 'tag_31', 'tag_32', 'tag_33', 
#       'tag_6', 'tag_7', 'tag_8', 'tag_9'
        ]
]

# 무산소조, SBR조, 호기조 NH4-N, NO3-N 기계세척으로 인한 0 출력 데이터 선언
change_data = {
    'RCS3' : ['ANALOG', ['tag_10', 'tag_11', 'tag_12', 'tag_13', 'tag_14', 'tag_15', 'tag_16', 'tag_17', 'tag_26', 'tag_27', 'tag_28', 'tag_29', 'tag_30', 'tag_31', 'tag_32', 'tag_33', 'tag_34', 'tag_35']]
}


# Define constants for min and max values
MIN_MAX_VALUES = {
    "DO": (0.1, 20.0),
    "MLSS": (1092.0, 5500.0),
    "ANX_NH4N": (0.1, 22.2),
    "ANX_NO3N": (0.1, 4.0),
    "SBR_NH4N": (0.1, 40.0),
    "SBR_NO3N": (0.1, 20.0),
    "ARB_NH4N": (0.1, 40.0),
    "ARB_NO3N": (0.1, 20.0),
#    "TOC": (14.4, 300),
#    "SS": (18.5, 400),
#    "TN": (3.4, 100),
#    "TP": (0.6, 20)
}

# 데이터 상한, 하한 기준 선언
minmaxValue = [
    [
        ["ARB_TANK_DO_301A", *MIN_MAX_VALUES["DO"]],
        ["ARB_TANK_DO_301B", *MIN_MAX_VALUES["DO"]],
        ["ARB_TANK_DO_302A", *MIN_MAX_VALUES["DO"]],
        ["ARB_TANK_DO_302B", *MIN_MAX_VALUES["DO"]],
        ["ARB_TANK_DO_302C", *MIN_MAX_VALUES["DO"]],
        ["ARB_TANK_DO_302D", *MIN_MAX_VALUES["DO"]],
        ["AARB_MLSS_FST", *MIN_MAX_VALUES["MLSS"]],
        ["AARB_MLSS_SCD", *MIN_MAX_VALUES["MLSS"]]
    ],
    [
        ["SBR_TANK_NH4N_FST_A", *MIN_MAX_VALUES["SBR_NH4N"]],
        ["SBR_TANK_NO3N_FST_A", *MIN_MAX_VALUES["SBR_NO3N"]],
        ["SBR_TANK_NH4N_FST_B", *MIN_MAX_VALUES["SBR_NH4N"]],
        ["SBR_TANK_NO3N_FST_B", *MIN_MAX_VALUES["SBR_NO3N"]],
        ["ANX_TANK_NH4N_FST", *MIN_MAX_VALUES["ANX_NH4N"]],
        ["ANX_TANK_NO3N_FST", *MIN_MAX_VALUES["ANX_NO3N"]],
        ["ARB_TANK_NH4N_FST", *MIN_MAX_VALUES["ARB_NH4N"]],
        ["ARB_TANK_NO3N_FST", *MIN_MAX_VALUES["ARB_NO3N"]],
        ["ARB_TANK_MLSS_SCD", *MIN_MAX_VALUES["MLSS"]],
        ["ARB_TANK_MLSS_FST", *MIN_MAX_VALUES["MLSS"]],
#        ["INFLOW_TOC_DNSTY_SCD", *MIN_MAX_VALUES["TOC"]],
#        ["INFLOW_SS_DNSTY_SCD", *MIN_MAX_VALUES["SS"]],
#        ["INFLOW_TN_DNSTY_SCD", *MIN_MAX_VALUES["TN"]],
#        ["INFLOW_TP_DNSTY_SCD", *MIN_MAX_VALUES["TP"]],
        ["SBR_TANK_NH4N_FST_C", *MIN_MAX_VALUES["SBR_NH4N"]],
        ["SBR_TANK_NO3N_FST_C", *MIN_MAX_VALUES["SBR_NO3N"]],
        ["SBR_TANK_NH4N_FST_D", *MIN_MAX_VALUES["SBR_NH4N"]],
        ["SBR_TANK_NO3N_FST_D", *MIN_MAX_VALUES["SBR_NO3N"]],
        ["ANX_TANK_NH4N_SCD", *MIN_MAX_VALUES["ANX_NH4N"]],
        ["ANX_TANK_NO3N_SCD", *MIN_MAX_VALUES["ANX_NO3N"]],
        ["ARB_TANK_NH4N_SCD", *MIN_MAX_VALUES["ARB_NH4N"]],
        ["ARB_TANK_NO3N_SCD", *MIN_MAX_VALUES["ARB_NO3N"]],
#        ["INFLOW_TOC_DNSTY_FST", *MIN_MAX_VALUES["TOC"]],
#        ["INFLOW_SS_DNSTY_FST", *MIN_MAX_VALUES["SS"]],
#        ["INFLOW_TN_DNSTY_FST", *MIN_MAX_VALUES["TN"]],
#        ["INFLOW_TP_DNSTY_FST", *MIN_MAX_VALUES["TP"]]
    ]
]

# minmaxValue = [ [['ARB_TANK_DO_301A', 0.1, 20.0], ['ARB_TANK_DO_301B', 0.1, 20.0]
#                  , ['ARB_TANK_DO_302A', 0.1, 20.0], ['ARB_TANK_DO_302B', 0.1, 20.0], ['ARB_TANK_DO_302C', 0.1, 20.0], ['ARB_TANK_DO_302D', 0.1, 20.0]
#                  , ['AARB_MLSS_FST', 1092.0, 5500.0], ['AARB_MLSS_SCD', 1092.0, 5500.0]]
#                 , [['SBR_TANK_NH4N_FST_A', 0.1, 20.0],['SBR_TANK_NO3N_FST_A', 0.1, 20.0],['SBR_TANK_NH4N_FST_B', 0.1, 20.0],['SBR_TANK_NO3N_FST_B', 0.1, 20.0]
#                    ,['ANX_TANK_NH4N_FST', 0.1, 22.2],['ANX_TANK_NO3N_FST', 0.1, 4.0],['ARB_TANK_NH4N_FST', 0.1, 20.0],['ARB_TANK_NO3N_FST', 0.1, 20.0]
#                    ,['ARB_TANK_MLSS_SCD', 1092.0, 5500.0],['ARB_TANK_MLSS_FST', 1092.0, 5500.0]
#                    ,['INFLOW_TOC_DNSTY_SCD', 14.4, 216.0],['INFLOW_SS_DNSTY_SCD', 18.5, 277.5],['INFLOW_TN_DNSTY_SCD', 3.4, 51.0],['INFLOW_TP_DNSTY_SCD', 0.6, 9.0]
#                    ,['SBR_TANK_NH4N_FST_C', 0.1, 20.0],['SBR_TANK_NO3N_FST_C', 0.1, 20.0],['SBR_TANK_NH4N_FST_D', 0.1, 20.0],['SBR_TANK_NO3N_FST_D', 0.1, 20.0]
#                    ,['ANX_TANK_NH4N_SCD', 0.1, 22.2],['ANX_TANK_NO3N_SCD', 0.1, 4.0],['ARB_TANK_NH4N_SCD', 0.1, 20.0],['ARB_TANK_NO3N_SCD', 0.1, 20.0]
#                    ,['INFLOW_TOC_DNSTY_FST', 14.4, 216.0],['INFLOW_SS_DNSTY_FST', 18.5, 277.5],['INFLOW_TN_DNSTY_FST', 3.4, 51.0],['INFLOW_TP_DNSTY_FST', 0.6, 9.0]]
#             ]


# 이상치 유무 체크 - 'outerlierChkData'테이블 저장 여부 
outerlierChk = False

#-----------------------------------------------------------------------------------------------------------------------

# 데이터 Insert
def insert_process(tabel_name,df_final):
    engine = create_engine('postgresql://postgres:Passw0rd!!@192.168.3.222:5432/EH_SEWER')
    try:
        # 데이터프레임을 DB 테이블에 삽입
        df_final.to_sql(tabel_name, engine, if_exists='append', index=False)
    finally:
        # 연결 종료
        engine.dispose()

#log남기기
# 로그 파일 경로 및 이름 생성 (현재 날짜 포함)
log_dir = 'EH_outlier_log'
log_file_name = os.path.join(log_dir, 'EH_outlier_log.log')

# 로그 디렉토리 존재 여부 확인 및 생성
os.makedirs(log_dir, exist_ok=True)  # exist_ok=True로 설정하면 디렉토리가 이미 존재해도 에러를 발생시키지 않음

logger = logging.getLogger(name='')  # RootLogger
logger.setLevel(logging.DEBUG)

max_file_size = 5 * 1024 * 1024  # 5MB
backup_count = 5  # 최대 5개의 백업 파일

file_handler = RotatingFileHandler(log_file_name,maxBytes=max_file_size, backupCount=backup_count) ## 파일 핸들러 생성
formatter = logging.Formatter('|%(asctime)s||%(name)s||%(levelname)s|\n%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter) ## 텍스트 포맷 설정
logger.addHandler(file_handler) ## 핸들러 등록

#-----------------------------------------------------------------------------------------------------------------------

# 데이터 가져오기
def fetch_data(sql_query):
    engine = create_engine('postgresql://postgres:Passw0rd!!@192.168.3.222:5432/EH_SEWER')
    try:
        logging.info('Start fetching data from database')
        df = pd.read_sql_query(sql_query, engine)
        return df
    except Exception as e:
        logging.error('Error while fetching data from database: %s', e)
        logging.debug('SQL Query: %s', sql_query)
        return None
    finally:
        engine.dispose()

# 이상치 체크 및 처리
def check_outliers(df):
    global outerlierChk
    df003 = df.copy()
    df004 = df.copy()

    for key, value in select_tag.items():
        rcs_id_filter = (df['rcs_id'] == key[:-1]) & (df['data_typ'] == value[0])
        tmp = df.loc[rcs_id_filter, value[1]]
        columns = tmp.columns

        for i in range(len(columns)):
            column = columns[i]
            x = tmp.iloc[-1, i]
            data = tmp.iloc[3:, i]
            median = data.median()
            mad = round(((data - median).abs()).median(), 2)
            mean = tmp.iloc[:5, i].mean()

            df003_filter = df003.loc[rcs_id_filter].index[-1]
            if mad == 0:
                if (tmp.iloc[1:5, i].mean() * 4) < x:
                    df003.loc[df003_filter, column] = np.nan
                    df004.loc[df003_filter, column] = mean
                    outerlierChk = True
            else:
                zscore = 0.6745 * (x - median) / mad
                threshold = determine_threshold(key)
                if abs(zscore) > threshold:
                    df003.loc[df003_filter, column] = np.nan
                    df004.loc[df003_filter, column] = mean
                    outerlierChk = True

    return df003, df004

# 기준값 결정 함수
def determine_threshold(key):
    if key in ['STN1', 'RCS31']:
        return 10
    elif key in ['RCS32']:
        return 200
    elif key in ['RCS33', 'RCS13', 'ANA3']:
        return 25
    elif key in ['RCS34', 'RCS14']:
        return 50
    else:
        return 40
        
        
# 무산소조, SBR조, 호기조 NH4-N, NO3-N 기계세척으로 인한 0 출력 시 이전데이터 유지
def retain_previous_value(df):
    chg_df = df.copy()
    # print(chg_df)
    for key, value in change_data.items():
        for col in value[1]:
            mask = (chg_df['rcs_id'] == key) & (chg_df['data_typ'] == value[0])
            filtered_data = chg_df.loc[mask, col]
            # 마지막 값 확인
            if filtered_data.iloc[-1] == 0.0:
                # 마지막 0보다 큰 값의 인덱스 찾기
                last_valid_index = filtered_data[filtered_data > 0].last_valid_index()
                
                if last_valid_index is not None:
                    last_valid_value = filtered_data.loc[last_valid_index]  # 마지막 유효 인덱스의 값
                    
                    # 마지막 0인 값을 이전 값으로 대체
                    # mask를 사용하여 직접적으로 수정
                    chg_df.loc[mask, col] = chg_df.loc[mask, col].replace(0.0, last_valid_value, regex=True)
            else:
                # 계측기 청소로 인한 연속적인 0 출력으로 현재 값이 이상치로 판단될 수 있음.
                # 해서 현재값 이전 2개의 데이터들이 연속으로 0을 출력하면 이전 값들 중 0보다 큰 데이터로 변환하여 이상치 판단
                # 현재 값 이전 5개 데이터 모두 0일 경우에는 0 데이터 대체 X
                # 현재값 이전 데이터들
                except_now = filtered_data.iloc[:-1]
                # 전체에서 0이 연속적으로 두 개 이상인지 확인
                consecutive_zeros = (except_now == 0.0).astype(int).rolling(window=2).sum()
                if (consecutive_zeros > 1).any():

                    # 연속된 0의 인덱스 찾기
                    zero_indices = except_now[except_now == 0.0].index.tolist()
                    # chg_df에서 zero_indices에 해당하는 날짜를 가져오기
                    query_dates_list = chg_df.loc[chg_df.index.intersection(zero_indices), 'meas_dtm'].tolist()

                    # query_dates_list에서 날짜를 문자열로 변환
                    query_dates_str = ", ".join([f"'{date.strftime('%Y-%m-%d %H:%M:%S')}'" for date in query_dates_list])
                    
                        # SQL 쿼리로 데이터 가져오기
                    sql_query = f"""
                        SELECT *
                        FROM ehswco_meas_mstr04
                        WHERE TRIM(rcs_id) = TRIM('{key}') AND TRIM(data_typ) = TRIM('{value[0]}')
                        AND meas_dtm IN ({query_dates_str})
                        ORDER BY meas_dtm;
                    """
                    
                    replace_df = fetch_data(sql_query)
                     # chg_df와 replace_df를 병합하여 대체
                    merged_df = chg_df.merge(replace_df, on=['meas_dtm', 'rcs_id', 'data_typ'], suffixes=('', '_replace'), how='left')

                    # 0인 데이터 대체
                    chg_df.loc[mask & (chg_df[col] == 0.0), col] = merged_df.loc[mask & (chg_df[col] == 0.0), f'{col}_replace']

    return chg_df
    

# 메인 처리 흐름
def main():

    logging.info('Start processing data')
    start = time.time()  # 시작 시간 저장

    sql_query = """
        SELECT *
        FROM ehswco_meas_mstr01
        WHERE meas_dtm IN (
            SELECT DISTINCT meas_dtm
            FROM ehswco_meas_mstr01
            ORDER BY meas_dtm DESC
            LIMIT 6
        )
        ORDER BY meas_dtm;
        """

    r_df = fetch_data(sql_query)

    if r_df is None:
        return
    # 컬럼 이름 지정
    tags = ['tag_' + str(i) for i in range(1, len(r_df.columns) - 2)]
    r_df.columns = ['meas_dtm', 'rcs_id', 'data_typ'] + tags
    r_df[tags] = r_df[tags].astype(float)
    r_df.reset_index(inplace=True, drop=True)

    #무산소조, SBR조, 호기조 NH4-N, NO3-N 
    df = retain_previous_value(r_df)

     # 상한하한 적용
    for i in range(len(rcs_list)):
        for j in range(len(tag_list[i])):
            df.loc[(df['rcs_id'] == rcs_list[i]) & (df['data_typ'] == group_list[i]), tag_list[i][j]] = \
                df.loc[(df['rcs_id'] == rcs_list[i]) & (df['data_typ'] == group_list[i]), tag_list[i][j]].clip(lower=minmaxValue[i][j][1], upper=minmaxValue[i][j][2])
    
    df003, df004 = check_outliers(df)

    now = df['meas_dtm'].max()
    try:
        logging.info('Start saving data to database ehswco_meas_mstr03')
        insert_process('ehswco_meas_mstr03', df003.loc[df003['meas_dtm'] == now])
        #insert_process('ehswco003', df003.loc[df003['meas_dtm'] == now])
    except Exception as e:
        logging.error('Error while saving data to database ehswco_meas_mstr03: %s', e)

    try:
        logging.info('Start saving data to database ehswco_meas_mstr04')
        insert_process('ehswco_meas_mstr04', df004.loc[df004['meas_dtm'] == now])
        #insert_process('ehswco004', df004.loc[df004['meas_dtm'] == now])
    except Exception as e:
        logging.error('Error while saving data to database ehswco_meas_mstr04: %s', e)

    if outerlierChk:
        try:
            logging.info('Start saving data to database outerlierChk')
            outerlierChkData = pd.DataFrame({'meas_dtm': [now]})
            insert_process('outerlierChkData', outerlierChkData)
        except Exception as e:
            logging.error('Error while saving data to database outerlierChk: %s', e)

    logging.info('Total process time: %s seconds', time.time() - start)
    logging.info('--------------------------------------------------------------------------------------------------')

if __name__ == "__main__":
    main()