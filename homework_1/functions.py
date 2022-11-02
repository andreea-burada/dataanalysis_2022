import os
import pandas
import json
import math

excel_path = os.getcwd()
excel_path += "\\dataIN\\CoduriRomania.xlsx"
csv_path = os.getcwd()
csv_path += "\\dataIN\\Ethnicity.csv"


'''
input: path - excel location
output: cities_sheet, counties_sheet, regions_sheet
'''
def read_excel(path=excel_path):
    
    cities_sheet = pandas.read_excel(path, sheet_name='Localitati')
    counties_sheet = pandas.read_excel(path, sheet_name='Judete', index_col=0)
    regions_sheet = pandas.read_excel(path, sheet_name='Regiuni', index_col=0)

    return cities_sheet, counties_sheet, regions_sheet


'''
input: path - csv location
output: ethnicities_csv - dict
'''
def read_csv(path=csv_path):
    ethnicities_csv = pandas.read_csv(path)
    
    return ethnicities_csv


'''
input: dict_2_save - dict to be exported, filename - name of json file
output: -
'''
def save_to_json(dict_2_save, filename):
    path = os.getcwd()
    path += '\\dataOUT\\' + filename
    json_from_dict = json.dumps(dict_2_save, indent=4, sort_keys=True)
    file = open(path, 'w')
    file.write(json_from_dict)
    file.close()


'''
input: eth_csv - csv of ethnicities; code_info_dict - dictionary of code and respective county, region and macroregion
output: code_info_dict - dictionary of codes and respective county, region and macroregion
'''
def map_code_to_info(eth_csv, cities_sheet, counties_sheet, regions_sheet):
    code_info_dict = {}
    # iterate over the csv
    for index, row in eth_csv.iterrows():

        # find county, region and macro-region for code
        current_code = row['Code']
        code_dict = {current_code: []}

        # find county
        current_county = cities_sheet.iloc[index]['County']
        code_dict[current_code].append(current_county)

        # find region
        current_region = counties_sheet.loc[current_county]['Regiune']
        code_dict[current_code].append(current_region)

        # find macro-region
        current_macroregion = regions_sheet.loc[current_region]['MacroRegiune']
        code_dict[current_code].append(current_macroregion)

        code_info_dict.update(code_dict)

    return code_info_dict


'''
input: eth_csv - csv of ethnicities; code_info_dict - dictionary of code and respective county, region and macroregion
output: ethnicities_per_county_dict - ce ne cere
'''
def ethnicities_per_county(eth_csv, code_info_dict):
    ethnicities_per_county_dict = {}
    for index, row in eth_csv.iterrows():
        # find county of code
        code = row['Code']
        # 0 - county
        county = code_info_dict[code][0]
        current_row_dict = dict(row[2:])
        # check if key exists
        if county not in ethnicities_per_county_dict:
            ethnicities_per_county_dict[county] = current_row_dict
        else:
            # if key exists, sum up values
            for key, value in current_row_dict.items():
               ethnicities_per_county_dict[county][key] += value


    return ethnicities_per_county_dict


'''
input: eth_csv - csv of ethnicities; code_info_dict - dictionary of code and respective county, region and macroregion
output: ethnicities_per_cregion_dict - ce ne cere
'''
def ethnicities_per_region(eth_csv, code_info_dict):
    ethnicities_per_region_dict = {}
    for index, row in eth_csv.iterrows():
        # find county of code
        code = row['Code']
        # 1 - region
        region = code_info_dict[code][1]
        current_row_dict = dict(row[2:])
        # check if key exists
        if region not in ethnicities_per_region_dict:
            ethnicities_per_region_dict[region] = current_row_dict
        else:
            # if key exists, sum up values
            for key, value in current_row_dict.items():
               ethnicities_per_region_dict[region][key] += value


    return ethnicities_per_region_dict


'''
input: eth_csv - csv of ethnicities; code_info_dict - dictionary of code and respective county, region and macroregion
output: ethnicities_per_macroregion_dict - ce ne cere
'''
def ethnicities_per_macroregion(eth_csv, code_info_dict):
    ethnicities_per_macroregion_dict = {}
    for index, row in eth_csv.iterrows():
        # find county of code
        code = row['Code']
        # 2 - macroregion
        macroregion = str(code_info_dict[code][2])
        current_row_dict = dict(row[2:])
        # check if key exists
        if macroregion not in ethnicities_per_macroregion_dict:
            ethnicities_per_macroregion_dict[macroregion] = current_row_dict
        else:
            # if key exists, sum up values
            for key, value in current_row_dict.items():
               ethnicities_per_macroregion_dict[macroregion][key] += value


    return ethnicities_per_macroregion_dict


'''
input: ethnicities_per_counties - dict of ethnicities sorted by county, ethnicity - ethnicity for which we calculate the index
output: dissimilarity_index - ce ne cere
'''
def dissimilarity_index(ethnicities_per_county, ethnicity):
    # T - total population of all counties
    T = 0
    for key, value in ethnicities_per_county.items():
        for k, v in value.items():
            T += v
    
    # dict_T - total population per county
    dict_T = {}
    for key, value in ethnicities_per_county.items():
        total = 0
        for k, v in value.items():
            total += v
        dict_T.update({key: total})
    
    # dict_Tx - total population per ethnicity
    dict_Tx = {}
    for key, value in ethnicities_per_county.items():
        for k, v in value.items():
            if k not in dict_Tx:
                dict_Tx.update({k: v})
            else:
                dict_Tx[k] += v

    # ethnicities list
    ethnicities = list(ethnicities_per_county[list(ethnicities_per_county.keys())[0]].keys())

    # counties list
    counties = list(ethnicities_per_county.keys())

    if ethnicity not in ethnicities:
        raise Exception('Invalid ethnicity')

    D = 0
    for i in range(len(counties)):
        current_county = counties[i]
        xi = ethnicities_per_county[current_county][ethnicity]
        ri = dict_T[current_county] - xi
        Tx = dict_Tx[ethnicity]
        Tr = T - Tx

        D += abs((xi / Tx) - (ri / Tr))

    return 0.5 * D


'''
input: ethnicities_per_counties - dict of ethnicities sorted by county, ethnicity - ethnicity for which we calculate the index
output: Shannon_Weaver - ce ne cere
'''
def Shannon_Weaver(ethnicities_per_county, ethnicity):
     # dict_T - total population per county
    dict_T = {}
    for key, value in ethnicities_per_county.items():
        total = 0
        for k, v in value.items():
            total += v
        dict_T.update({key: total})

    # ethnicities list
    ethnicities = list(ethnicities_per_county[list(ethnicities_per_county.keys())[0]].keys())

    # counties list
    counties = list(ethnicities_per_county.keys())

    if ethnicity not in ethnicities:
        raise Exception('Invalid ethnicity')

    H = 0
    for i in range(len(counties)):
        current_county = counties[i]
        xi = ethnicities_per_county[current_county][ethnicity]
        if xi != 0:
            ri = dict_T[current_county] - xi
            pi = xi / (ri + xi)

            H += pi * math.log(pi, 2)

    return -abs(H)
