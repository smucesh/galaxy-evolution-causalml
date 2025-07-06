import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time

# Configuration
treatment_name = 'nearest_neighbour_density'  # subhalo_velocity, nearest_neighbour, nearest_neighbour_cut, subhalo_mass
model_name = 'causal_model_halo_mass'  # causal_model, no_model, traditional_model, stellar_mass, subhalo_mass
time_point = 'current'  # final, current

# Loading data
df_data = pd.read_hdf('data/data.h5', 'snapshots')
print(len(df_data))

df_data['weight'] = ''

df_data = df_data.unstack(level=1)
print(len(df_data))

# Analysis
levels = 100
quantiles = np.linspace(0.01, 0.99, levels)

samples = 1000
snapshots = np.flip(df_data.columns.levels[1].values)
iterables = [np.arange(levels), snapshots, np.arange(samples)]
print(iterables)

df_results = pd.DataFrame(columns=['outcome'], index=pd.MultiIndex.from_product(iterables, names=["levels", "snapshot", "sample"]))
df_results = df_results.unstack(level=2)

treatment = np.zeros((len(df_data), len(snapshots)))

for i, snapshot in enumerate(snapshots):
    treatment[:, i] = df_data.loc[:, (treatment_name, min(snapshots)):(treatment_name, snapshot)].mean(axis=1)

treatment = np.fliplr(treatment)
treatment = np.quantile(treatment, q=quantiles, axis=0).reshape(-1, 1).flatten()
df_results['treatment'] = treatment
df_results.set_index('treatment', append=True, inplace=True)

# Bootstrapping
if model_name == 'subhalo_mass':
    var_treatment = 'subhalo_mass'
else:
    var_treatment = treatment_name
var_outcome = 'star_formation_rate'

start = time.time()
for sample in range(samples):

    # print(sample)

    # Bootstrap sample dataframe
    df_data_sample = df_data.sample(frac=1., replace=True, random_state=sample)

    # Load weights
    if model_name != 'no_model':
        weights = np.load(f'../results/{treatment_name}/{model_name}/weights/{sample}.npy')
        df_data_sample['weight'] = weights

    # print(df_data_sample['weight'])

    for i, snapshot in enumerate(snapshots):

        # print(i, snapshot)

        df_data_sample_copy = df_data_sample.copy()

        # Forming final weights and treatment
        if model_name == 'no_model':
            pass
        elif model_name == 'naive_model_stellar_mass':  # change for different models
            df_data_sample_copy['weight', 'final'] = df_data_sample_copy['weight', snapshot]
        else:
            df_data_sample_copy['weight', 'final'] = df_data_sample_copy.loc[:,
                                                     ('weight', min(snapshots)):('weight', snapshot)].prod(axis=1)

        df_data_sample_copy[var_treatment, 'final'] = df_data_sample_copy.loc[:,
                                                      (var_treatment, min(snapshots)):(var_treatment, snapshot)].mean(axis=1)
        # print(df_data_sample['weight', 'final'])
        # print(df_data_sample[var_treatment, 'final'])

        if model_name != 'no_model':

            # Trimming weights
            lower_quantile = df_data_sample_copy['weight', 'final'].quantile(.01)
            upper_quantile = df_data_sample_copy['weight', 'final'].quantile(.99)

            # Only trim weights if they exist for a snapshot
            if upper_quantile != lower_quantile:
                df_data_sample_copy = df_data_sample_copy[(df_data_sample_copy['weight', 'final'] > lower_quantile) & (
                            df_data_sample_copy['weight', 'final'] < upper_quantile)]

        # Defining treatment
        treatment = df_data.loc[:, (var_treatment, min(snapshots)):(var_treatment, snapshot)].mean(axis=1)
        treatment = np.quantile(treatment, q=quantiles).reshape(-1, 1)

        # Outcome model
        X = df_data_sample_copy[[(var_treatment, 'final')]]
        if time_point == 'final':
            y = df_data_sample_copy[var_outcome, 99]
        else:
            y = df_data_sample_copy[var_outcome, snapshot]

        # print(X.shape, y.shape)

        reg = RandomForestRegressor(n_estimators=100, n_jobs=-1, min_samples_leaf=200)
        if model_name == 'no_model':
            reg.fit(X, y)
        else:
            reg.fit(X, y, sample_weight=df_data_sample_copy['weight', 'final'])

        # Predicting outcome
        outcome = reg.predict(treatment)
        df_results['outcome', sample].loc[:, snapshot] = outcome

finish = time.time()
print(finish - start)

df_results = df_results.stack()
print(len(df_results))

df_results = df_results.sort_index(level=[0, 1], ascending=[True, False])
print(len(df_results))

df_results.to_hdf(f'results/{model_name}/joint_effect_outcome_{time_point}.h5', 'snapshots_all', format='fixed', data_columns=True)

