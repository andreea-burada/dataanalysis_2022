import pandas
import os
import json

path = os.getcwd()
path += "\\dataIN\\CoduriRomania.xlsx"
csv_path = os.getcwd()
csv_path += "\\dataIN\\Ethnicity.csv"

'''
input: path - excel location
output: cities_sheet, counties_sheet, regions_sheet
'''
def read_excel(path=path):
    
    cities_sheet = pandas.read_excel(path, sheet_name='Localitati')
    counties_sheet = pandas.read_excel(path, sheet_name='Judete', index_col=0)
    regions_sheet = pandas.read_excel(path, sheet_name='Regiuni', index_col=0)

    # excel_dict = pandas.read_excel(path, sheet_name=None, index_col=0)
    # print(type(excel_dict))
    # print(excel_dict)

    # for key, value in excel_dict.items():
    #     print(key, value)
    # return cities_sheet, counties_sheet, regions_sheet

    # for test
    # print(cities_sheet)
    # print(counties_sheet)
    # print(regions_sheet)
    
    # traverse cities_sheet
    # i = 0
    # for index, row in cities_sheet.iterrows():
    #     i += 1
    #     print(row['Code'])
    #     if i == 20:
    #         break

    cities_sheet.groupby

    return cities_sheet, counties_sheet, regions_sheet


def read_csv(path=csv_path):
    ethnicities_csv = pandas.read_csv(path)
    
    return ethnicities_csv


def save_to_json(dict_2_save, filename):
    path = os.getcwd()
    path += '\\dataOUT\\' + filename
    json_from_dict = json.dumps(dict_2_save, indent=4, sort_keys=True)
    file = open(path, 'w')
    file.write(json_from_dict)
    file.close()


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
        # 1 - macroregion
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


# test code
s1, s2, s3 = read_excel()
eth_csv = read_csv()
marele_dict = map_code_to_info(eth_csv, s1, s2, s3)
eth_county = ethnicities_per_county(eth_csv, marele_dict)
eth_region = ethnicities_per_region(eth_csv, marele_dict)
eth_macroregion = ethnicities_per_macroregion(eth_csv, marele_dict)

save_to_json(eth_county, filename="ethnicities_per_county.json")
save_to_json(eth_region, filename="ethnicities_per_region.json")
save_to_json(eth_macroregion, filename="ethnicities_per_macroregion.json")
