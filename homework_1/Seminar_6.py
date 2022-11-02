import functions as func
import os
import numpy

# 1. Calculate and save the population by ethnicity at the level of counties, regions and macroregions

excel_path = os.getcwd()
excel_path += '\\dataIN\\CoduriRomania.xlsx'

csv_path = os.getcwd()
csv_path += '\\dataIN\\Ethnicity.csv'

# read input files

# read excel file
# and get the data as panda dataframes
counties_sheet, region_sheet, macroregion_sheet = func.read_excel(excel_path)

# read csv file
ethnicities_csv = func.read_csv(csv_path)

# generate dictionary of codes and corresponding county, region and macroregion
code_map_dict = func.map_code_to_info(ethnicities_csv, counties_sheet, region_sheet, macroregion_sheet)

# generate dictionaries of ethnicities at level of counties, regions and macroregions
ethnicities_per_county_dict = func.ethnicities_per_county(ethnicities_csv, code_map_dict)
ethnicities_per_region_dict = func.ethnicities_per_region(ethnicities_csv, code_map_dict)
ethnicities_per_macroregion_dict = func.ethnicities_per_macroregion(ethnicities_csv, code_map_dict)

# export dicts to jsons and save data
func.save_to_json(ethnicities_per_county_dict, 'Ethnicities_per_County.json')
func.save_to_json(ethnicities_per_region_dict, 'Ethnicities_per_Region.json')
func.save_to_json(ethnicities_per_macroregion_dict, 'Ethnicities_per_MacroRegion.json')



# 2. To calculate and save the indications of ethnic segregation at the counties level

# get list of ethnicities
ethnicities = list(ethnicities_per_county_dict[list(ethnicities_per_county_dict.keys())[0]].keys())

# The dissimilarity index:
for eth in ethnicities:
    dissimilarity_index = func.dissimilarity_index(ethnicities_per_county_dict, eth)
    print('Dissimilarity index of ' + eth + ' is ' + str(numpy.round(dissimilarity_index, 3)))

print('\n')

# Shannon-Weaver index:
for eth in ethnicities:
    shannon_weaver = func.Shannon_Weaver(ethnicities_per_county_dict, eth)
    print('Shannon-Weaver index of ' + eth + ' is ' + str(numpy.round(shannon_weaver, 3)))