import os
import json
import pandas
import numpy as np
import factor_analyzer as fa
from sklearn.preprocessing import StandardScaler

import utils
import PCA.PCA as pca  # Principal Component Analysis
import EFA.EFA as efa  # Exploratory Factor Analysis
import graphics as graph

'''
Load config.json
'''
f = open('config.json')
config = json.load(f)
f.close()
# print(config)



'''
    Data extraction from EXCEL
'''
# there are 7 sheets that need to be merged
# primary-education; lower-secondary-education; upper-secondary-education; bachelor-education; master's-education; risk-of-poverty-prec; unemployment-rate

primary_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="primary-education", index_col=0)
lower_secondary_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="lower-secondary-education", index_col=0)
upper_secondary_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="upper-secondary-education", index_col=0)
bachelor_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="bachelor-education", index_col=0)
masters_ed_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="master's-education", index_col=0)
poverty_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="risk-of-poverty-prec", index_col=0)
unemployment_rate_df = pandas.read_excel(io=config["DATA_PATH"], sheet_name="unemployment-rate", index_col=0)



'''
    DataFrame processing
'''

# merge all sheets into one table
table_df = primary_ed_df.merge(lower_secondary_ed_df, left_on='Country', right_on='Country')
table_df = table_df.merge(upper_secondary_ed_df, left_on='Country', right_on='Country')
table_df = table_df.merge(bachelor_ed_df, left_on='Country', right_on='Country')
table_df = table_df.merge(masters_ed_df, left_on='Country', right_on='Country')
table_df = table_df.merge(poverty_df, left_on='Country', right_on='Country')
table_df = table_df.merge(unemployment_rate_df, left_on='Country', right_on='Country')

variable_names = table_df.columns.values
variables_list = list(variable_names)
# print(variables_list)
observation_names = table_df.index.values
countries_list = list(observation_names)
# print(countries_list)

# export to xlsx
table_df.to_excel(os.path.join(config["RESULTS_PATH"], 'data_merged.xlsx'))

# get matrix from DataFrame
table_values = table_df.values

# replace NaN values if there are any
X_matrix = table_values
# X_matrix = utils.replaceNaN(X_matrix)
X_df = pandas.DataFrame(data=X_matrix, index=observation_names, columns=variable_names)
X_df.to_csv(path_or_buf=os.path.join(config["RESULTS_PATH"], 'X.csv'))

# standardize X
X_standardized = utils.standardize(X_matrix)

# export standardized matrix to CSV file
X_standardized_df = pandas.DataFrame(data=X_standardized, index=observation_names, columns=variable_names)
X_standardized_df.to_csv(path_or_buf=os.path.join(config["RESULTS_PATH"], 'X_standardized.csv'))

# scatter plot of standardized matrix
graph.scatterPlot(matrix=X_standardized_df,
                  title="Scatter Plot of Standardized Matrix of Observation - No. of Primary Education Enrollments")
graph.show()

# group data by European region
regions_df = pandas.read_excel(io=config["REGIONS_PATH"], index_col=0)
# label each country by region
table_by_regions_df = table_df.merge(regions_df, left_on='Country', right_on='Country')
# export to csv
table_by_regions_df.to_csv(path_or_buf=os.path.join(config["RESULTS_PATH"], 'data_by_european_region_not_aggreg.csv'))
# group data by region - sum education enorllment numbers, average for percentage
group_by_regions_df = table_by_regions_df[['Region'] + variables_list[:-2]].groupby(by='Region').agg(func=sum).merge(
    right=(table_by_regions_df[['Region'] + variables_list[-2:]].groupby(by='Region').agg('mean')), left_on='Region',
    right_on='Region')
# print(group_by_regions)
group_by_regions_df.to_csv(path_or_buf=os.path.join(config["RESULTS_PATH"], 'data_by_european_region.csv'))



'''
    Principal Component Analysis
'''
# create PDA model
PCA_model = pca.PCA(X_matrix)
print(PCA_model.get_X_standardized())

# save the co-variance matrix into a CSV file
cov_matrix = PCA_model.get_covariance_matrix()
cov_df = pandas.DataFrame(data=cov_matrix, columns=variable_names, index=variable_names)
cov_df.to_csv(path_or_buf=os.path.join(config["RESULTS_PATH"], 'covariance_matrix_PCA.csv'))
graph.correlogram(matrix=cov_df, dec=1, title='Correlogram of Variance-Covariance Matrix')

# create the graph of explained variance by the principal components
alpha = PCA_model.get_eigenvalues()  # alpha = eigenvalues
graph.principal_components(eigenvalues=alpha, title='Explained Variance by the Principal Components')

# get factor loadings
Rxc = PCA_model.get_Rxc()
Rxc_df = pandas.DataFrame(data=Rxc, columns=['C' + str(i + 1) for i in range(variable_names.shape[0])],
                          index=variable_names)
# save to CSV
Rxc_df.to_csv(path_or_buf=os.path.join(config["RESULTS_PATH"], 'PCA_factor_loadings.csv'))
# create the correlogram of factor loadings
graph.correlogram(matrix=Rxc_df, dec=1, title='Correlogram of Factor Loadings')

# quality of observations
calc_obs = PCA_model.get_calculated_observations()
calc_obs_df = pandas.DataFrame(data=calc_obs, index=observation_names, columns=['C' + str(i + 1) for i in range(variable_names.shape[0])])
graph.correlogram(matrix=calc_obs_df, title='Correlogram of Quality of Observations on the PC Axis')

# common
common = PCA_model.get_common()
common_df = pandas.DataFrame(data=common, index=variable_names, columns=['C' + str(i + 1) for i in range(variable_names.shape[0])])
graph.correlogram(matrix=common_df, title='Community Correlogram')

# scores
scores = PCA_model.get_scores()
scores_df = pandas.DataFrame(data=scores, index=observation_names, columns=['C' + str(i + 1) for i in range(variable_names.shape[0])])
graph.correlogram(matrix=scores_df, title='Score Matrix')

# correlation circle - variables
graph.correlation_circle(matrix=Rxc_df, title='Correlation Circle - Between Variables and C1, C2')

# correlation circle - observations
max_score = np.max(scores)
min_score = np.min(scores)
graph.correlation_circle(matrix=scores_df, valMin=min_score, valMax=max_score, title='Observation Distribution by C1, C2')

graph.show()



'''
    Exploratory Factor Analysis
'''

EFA_model = efa.EFA(X_matrix)
# get X standardized using sklearn
scalars = StandardScaler()
X_standardized_EFA = scalars.fit_transform(X_matrix)

# save to CSV
X_standardized_EFA_df = pandas.DataFrame(data=X_standardized_EFA, index=observation_names, columns=variable_names)
X_standardized_EFA_df.to_csv(path_or_buf=os.path.join(config["RESULTS_PATH"], 'X_standardized_EFA.csv'))

# Factor existence estimation - Bartlett sphericity test
sphericity_Bartlett = fa.calculate_bartlett_sphericity(X_standardized_EFA_df)
print(sphericity_Bartlett)
# testing hypothesis
if sphericity_Bartlett[0] > sphericity_Bartlett[1]:
    print('H₁: There is at least one common factor.')
else:
    print('H₀: There are no common factors.')
    exit(-1)

# Factor existence estimation - KMO (Kaiser-Meyer-Olkin) index
kmo_index = fa.calculate_kmo(X_standardized_EFA_df)
# print(kmo_index)
kmo_vector = kmo_index[0]
# print(type(kmo_vector)); print(kmo_vector.shape)
kmo_matrix = kmo_vector[:, np.newaxis]
# print(kmo_matrix); print(kmo_matrix.shape)
kmo_matrix_df = pandas.DataFrame(data=kmo_matrix, columns=['KMO_indices'], index=variable_names)
graph.correlogram(matrix=kmo_matrix_df, dec=5, title='The Correlogram of Kaiser-Meyer-Olkin indices')
graph.show()

if kmo_index[1] >= 0.5:
    print('H₁: There is at least one common factor.')
else:
    print('H₀: There are no common factors.')
    exit(-2)

# Observe the variables in the space of factor 1 and 2
# extracting the significant factors
no_factors = 1
chi2TabMin = 1
for k in range(1, variable_names.shape[0]):
    fa_model = fa.FactorAnalyzer(n_factors=k)
    fa_model.fit(X=X_standardized_EFA_df)
    common_factors = fa_model.loadings_  # factorii comuni - factorii de corelatie
    # print(common_factors)
    specific_factors = fa_model.get_uniquenesses()
    # print(specific_factors)

    chi2Calc, chi2Tab = EFA_model.BartlettTest(common_factors, specific_factors)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break

    if chi2Tab < chi2TabMin:
        chi2TabMin = chi2Tab
        no_factors = k

print("No. factors: ", no_factors)

# # Crearea modelului cu numarul de factori semnificativi
# fa_Fit_model = fa.FactorAnalyzer(n_factors=no_factors)
# fa_Fit_model.fit(X_standardized_df)
#
# FA_factor_loadings = fa_Fit_model.loadings_
# factor_names = ['F' + str(j + 1) for j in range(0, no_factors)]
# FA_factor_loadings_df = pandas.DataFrame(data=FA_factor_loadings, columns=factor_names, index=variable_names)
# graph.correlogram(matrix=FA_factor_loadings_df, title="Correlogram of correlation factors from FactorAnalyzer")
#
# PCA_eigenvalues = EFA_model.get_eigenvalues()
# # eigenvalues graph from PCA
# graph.principal_components(eigenvalues=PCA_eigenvalues, title="Variance explained by PCA principal components")
#
#
# # the representation quality of observations on the factors axis
# calculated_observations = EFA_model.get_calculated_observations()
# calculated_observations_df = pandas.DataFrame(data=calculated_observations, columns=['F' + str(j + 1) for j in range(variable_names.shape[0])], index=observation_names)
# graph.correlogram(matrix=calculated_observations_df, title="Representation Quality of Observations on the Factors Axis")
#
# graph.show()
