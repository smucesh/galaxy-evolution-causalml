import numpy as np
import pandas as pd
import scipy.stats
from sklearn.ensemble import RandomForestRegressor
import time

# Loading data
df_data = pd.read_hdf('data/data.h5', 'snapshots')
print(len(df_data))

df_data = df_data.unstack(level=1)
print(len(df_data))

index = df_data.index.values
snapshots = np.flip(df_data.columns.levels[1].values)
iterables = [index, snapshots]
print(iterables)

# Configuration
samples = 1000
model = 'causal_model_halo_mass'
var_confounder = 'halo_mass' # stellar mass for traditional model
var_treatment = 'nearest_neighbour_density'
var_outcome = 'star_formation_rate'
lag = len(snapshots)

# Bootstrapping
start = time.time()
for sample in range(samples):
    
    print(sample)

    # Generate bootstrap sample
    df_data_sample = df_data.sample(frac=1., replace=True, random_state=sample)
    
    # Creating empty weights array
    df_weights = pd.DataFrame(columns=['weight'], index=pd.MultiIndex.from_product(iterables, names=["id", "snapshot"]))
    df_weights = df_weights.unstack(level=1)
    
    for i, snapshot in enumerate(snapshots):
        
        print(snapshot)

        # For Causal Model
        if i == len(snapshots) - 1:
            df_weights['weight', snapshot] = 1

        else:
            # Data
            confounder_history = df_data_sample.loc[:, (var_confounder, snapshots[lag-1]):(var_confounder, snapshots[i+1])]
            treatment_history = df_data_sample.loc[:, (var_treatment, snapshots[lag-1]):(var_treatment, snapshots[i+1])]
            treatment = df_data_sample[var_treatment, snapshot]
            
            # Propensity Score Model
            X = np.column_stack((confounder_history.values, treatment_history.values))
            y = treatment.values
            #print(X.shape, y.shape)

            reg = RandomForestRegressor(n_estimators=100, n_jobs=-1, min_samples_leaf=5)
            reg.fit(X, y)
            y_pred = reg.predict(X)
            propensity_score = scipy.stats.norm(y_pred, np.std(y-y_pred)).pdf(y)

            # Numerator Model
            X = treatment_history
            y = treatment.values
            #print(X.shape, y.shape)

            reg = RandomForestRegressor(n_estimators=100, n_jobs=-1,  min_samples_leaf=5)
            reg.fit(X, y)
            y_pred = reg.predict(X)
            marginal_score = scipy.stats.norm(y_pred, np.std(y-y_pred)).pdf(y)

            # Weight
            df_weights['weight', snapshot] = marginal_score/propensity_score

        '''         
        # For Traditional Model
        # Data
	    confounder = df_data_sample[var_confounder, snapshot]
        treatment = df_data_sample[var_treatment, snapshot]
        #print(confounder)
        #print(treatment)

        # Propensity Score Model
        X = confounder.values.reshape(-1, 1)
        y = treatment.values
        #print(X.shape, y.shape)

        reg = RandomForestRegressor(n_estimators=100, n_jobs=-1, min_samples_leaf=5)
        reg.fit(X, y)
        y_pred = reg.predict(X)
        propensity_score = scipy.stats.norm(y_pred, np.std(y-y_pred)).pdf(y)

        # Numerator Model
        kde = scipy.stats.gaussian_kde(treatment)
        marginal_score = kde.evaluate(treatment)

        # Weight
        df_weights['weight', snapshot] = marginal_score/propensity_score
        '''

    weights = df_weights.to_numpy()
    np.save(f'results/{model}/weights/{sample}.npy', weights)
            
finish = time.time()
print(finish-start)

