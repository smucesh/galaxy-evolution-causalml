import numpy as np
import pandas as pd

model = 'causal_model_halo_mass'

# Loading data
df_data = pd.read_hdf('data/data.h5', 'snapshots')
print(len(df_data))

df_data['weight'] = ''
df_data = df_data.unstack(level=1)
print(len(df_data))

df_weights = pd.read_hdf(f'results/{model}/weights.h5', 'snapshots_all')
df_weights = df_weights.unstack(level=1)
df_data['weight'] = df_weights.to_numpy()

snapshots = np.flip(df_data.columns.levels[1].values)
iterables = [snapshots]
print(iterables)

df_balance = pd.DataFrame(columns=['aacc_original', 'aacc_weighted'], index=pd.MultiIndex.from_product(iterables, names=['snapshot']))

samples = 1000
iterables = [np.arange(samples), snapshots]
print(iterables)

# Configuration
var_confounder = 'halo_mass'
var_treatment = 'nearest_neighbour_density'
lag = len(snapshots)
correlation = 'kendall'

for i, snapshot in enumerate(snapshots):

    print(i)

    if i == len(snapshots) - 1:
        pass

    else:
        # Create copy of the original dataset for trimming
        df_data_copy = df_data.copy()

        # Create empty dataframe to store correlations for each snapshot
        df_correlations = pd.DataFrame(columns=[var_confounder, var_treatment], index=pd.MultiIndex.from_product(iterables, names=['sample', 'snapshot']))
        df_correlations = df_correlations.unstack(level=1)

        # Trimming weights
        lower_quantile = df_data_copy['weight', snapshot].quantile(.01)
        upper_quantile = df_data_copy['weight', snapshot].quantile(.99)
        print(lower_quantile, upper_quantile)

        # df_data_copy = df_data_copy[(df_data_copy['weight', snapshot]>lower_quantile) & (df_data_copy['weight', snapshot]<upper_quantile)]

        for sample in range(samples):
            # Sample data with weighting
            df_data_sample = df_data_copy.sample(frac=1, replace=True, random_state=sample, weights=df_data_copy['weight', snapshot])

            confounder_history = df_data_sample.loc[:, (var_confounder, snapshots[lag-1]):(var_confounder, snapshots[i+1])]
            treatment_history = df_data_sample.loc[:, (var_treatment, snapshots[lag-1]):(var_treatment, snapshots[i+1])]
            treatment = df_data_sample[var_treatment, snapshot]

            df_correlations.loc[sample][var_confounder] = confounder_history.corrwith(treatment, method=correlation)
            df_correlations.loc[sample][var_treatment] = treatment_history.corrwith(treatment, method=correlation)

        # Weighted data
        dj_weighted = df_correlations.mean()  # Mean correlation of each confounder with treatment
        #zj_weighted = 0.5 * np.log((1 + dj_weighted) / (1 - dj_weighted))  #  Fisher transformation
        #aacc_weighted = zj_weighted.abs().mean()  # Average absolute correlation coefficient
        aacc_weighted = dj_weighted.abs().mean() # Average absolute correlation coefficient

        # Original data (should be trimmed data or original data?)
        confounder_history = df_data.loc[:, (var_confounder, snapshots[lag-1]):(var_confounder, snapshots[i+1])]
        treatment_history = df_data.loc[:, (var_treatment, snapshots[lag-1]):(var_treatment, snapshots[i+1])]
        treatment = df_data[var_treatment, snapshot]

        dj_original = pd.concat([confounder_history.corrwith(treatment, method=correlation), treatment_history.corrwith(treatment, method=correlation)], axis=0)
        #zj_original = 0.5 * np.log((1 + dj_original) / (1 - dj_original))
        #aacc_original = zj_original.abs().mean()
        aacc_original = dj_original.abs().mean()

        df_balance['aacc_original'].loc[snapshot] = aacc_original
        df_balance['aacc_weighted'].loc[snapshot] = aacc_weighted

df_balance.to_hdf(f'results/{model}/balance_marginal_effect_no_trimming.h5', 'snapshots_all')