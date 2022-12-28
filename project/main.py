import os
import json
import pandas
import numpy as np
import factor_analyzer as fa

import utils
import EFA.EFA as efa
import graphics as graph

'''
Load config.json
'''
f = open('config.json')
config = json.load(f)
f.close()
#print(config)

'''
get data from Excel file
there are 7 sheets that need to be merged
primary-education; lower-secondary-education; upper-secondary-education; bachelor-education; master's-education; risk-of-poverty-prec; unemployment-rate
'''
primary_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="primary-education", index_col=0)
lower_secondary_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="lower-secondary-education", index_col=0)
upper_secondary_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="upper-secondary-education", index_col=0)
bachelor_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="bachelor-education", index_col=0)
masters_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="master's-education", index_col=0)
poverty_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="risk-of-poverty-prec", index_col=0)
unemployment_rate_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="unemployment-rate", index_col=0)

# merge all sheets into one table
table_df = primary_ed_df.merge(lower_secondary_ed_df, left_on='Country', right_on='Country')
table_df = table_df.merge(upper_secondary_ed_df, left_on='Country', right_on='Country')
table_df = table_df.merge(bachelor_ed_df, left_on='Country', right_on='Country')
table_df = table_df.merge(masters_ed_df, left_on='Country', right_on='Country')
table_df = table_df.merge(poverty_df, left_on='Country', right_on='Country')
table_df = table_df.merge(unemployment_rate_df, left_on='Country', right_on='Country')

variable_names = table_df.columns.values
variables_list = list(variable_names)
#print(variables_list)
observation_names = table_df.index.values
countries_list = list(observation_names)
#print(countries_list)

# get matrix from DataFrame
table_values = table_df.values

# replace NaN values if there are any
X_matrix = utils.replaceNaN(table_values)
EFA_model = efa.EFA(X_matrix)
X_standardized = EFA_model.get_X_standardized()

# export standardized matrix to CSV file
X_standardized_df = pandas.DataFrame(data=X_standardized, index=observation_names, columns=variable_names)
X_standardized_df.to_csv(path_or_buf=os.path.join(config["RESULTS_PATH"], 'X_standardized.csv'))

# group data by European region
regions_df = pandas.read_excel(io=config["REGIONS_PATH"], index_col=0)
table_by_regions_df = table_df.merge(regions_df, left_on='Country', right_on='Country')
# group data by region - sum education enorllment numbers, average for percentage
group_by_regions = table_by_regions_df[['Region'] + variables_list[:-2]].groupby(by='Region').agg(func=sum).merge(
    right=(table_by_regions_df[['Region'] + variables_list[-2:]].groupby(by='Region').agg('mean')), left_on='Region', right_on='Region')
#print(group_by_regions)
group_by_regions.to_csv(path_or_buf=os.path.join(config["RESULTS_PATH"], 'data_by_european_region.csv'))


# Factor existence estimation - Bartlett sphericity test
sphericity_Bartlett = fa.calculate_bartlett_sphericity(X_standardized_df)
print(sphericity_Bartlett)
# testing hypothesis
if sphericity_Bartlett[0] > sphericity_Bartlett[1]:
    print('H₁: There is at least one common factor.')
else:
    print('H₀: There are no common factors.')
    exit(-1)
    
    
# Factor existence estimation - KMO (Kaiser-Meyer-Olkin) index
kmo_index = fa.calculate_kmo(X_standardized_df)
#print(kmo_index)
kmo_vector = kmo_index[0]
#print(type(kmo_vector)); print(kmo_vector.shape)
kmo_matrix = kmo_vector[:, np.newaxis]
#print(kmo_matrix); print(kmo_matrix.shape)
kmo_matrix_df = pandas.DataFrame(data=kmo_matrix, columns=['KMO_indices'], index=variable_names)
graph.correlogram(matrix=kmo_matrix_df, dec=5, title='The correlogram of Kaiser-Meyer-Olkin indices')

if kmo_index[1] >= 0.5:
    print('H₁: There is at least one common factor.')
else:
    print('H₀: There are no common factors.')
    exit(-2)

# scatter plot of standardized matrix
graph.scatterPlot(matrix=X_standardized_df, title="Scatter Plot of Standardized Matrix of Observation")


# Observe the variables in the space of factor 1 and 2
# extracting the significant factors
no_factors = 1
chi2TabMin = 1
for k in range(1, variable_names.shape[0]):
    fa_model = fa.FactorAnalyzer(n_factors=k)
    fa_model.fit(X=X_standardized_df)
    common_factors = fa_model.loadings_  # factorii comuni - factorii de corelatie
    #print(common_factors)
    specific_factors = fa_model.get_uniquenesses()
    #print(specific_factors)

    chi2Calc, chi2Tab = EFA_model.BartlettTest(common_factors, specific_factors)
    #print(chi2Calc, chi2Tab)
    #aefModel.calculTestBartlett(factoriComuni, factoriSpecifici)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break

    if chi2Tab < chi2TabMin:
        chi2TabMin = chi2Tab
        no_factors = k

print("No. factors: ", no_factors)

# Crearea modelului cu numarul de factori semnificativi
fa_Fit_model = fa.FactorAnalyzer(n_factors=no_factors)
fa_Fit_model.fit(X_standardized_df)

FA_factor_loadings = fa_Fit_model.loadings_
factor_names = ['F' + str(j + 1) for j in range(0, no_factors)]
FA_factor_loadings_df = pandas.DataFrame(data=FA_factor_loadings, columns=factor_names, index=variable_names)
graph.correlogram(matrix=FA_factor_loadings_df, title="Correlogram of correlation factors from FactorAnalyzer")

PCA_eigenvalues = EFA_model.get_eigenvalues()
# eigenvalues graph from PCA
graph.principal_components(eigenvalues=PCA_eigenvalues, title="Variance explained by PCA principal components")


# the representation quality of observations on the factors axis
calculated_observations = EFA_model.get_calculated_observations()
calculated_observations_df = pandas.DataFrame(data=calculated_observations, columns=['F' + str(j + 1) for j in range(variable_names.shape[0])], index=observation_names)
graph.correlogram(matrix=calculated_observations_df, title="Representation Quality of Observations on the Factors Axis")

graph.show()