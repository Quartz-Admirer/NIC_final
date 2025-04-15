import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import load_data
import boids


feature_cols = ["close", "ma_close",
                    "boids_mean_x", "boids_mean_y", "boids_mean_vx", "boids_mean_vy",
                    "boids_std_x", "boids_std_y", "boids_std_vx", "boids_std_vy"]
target_col = "future_close"

limits=[1000,5000,10000,20000]
numsboids=[50,100,200,400]
dimensions=[100,200,400,800]
speeds=[5,10,20]
radiuses=[50,100,150,200]

limits=[20000]
numsboids=[100,200,400]
dimensions=[100,200,400,800]
speeds=[5,10,20]
radiuses=[50,100,150,200]

for limit in limits:
    for num_boids in numsboids:
        for dimension in dimensions:
            for max_speed in speeds:
                for perception_radius in radiuses:
                    processed_path=f'{limit}_{num_boids}_{dimension}_{max_speed}_{perception_radius}.csv'
                    df = load_data.load_and_preprocess(symbol="BTCUSDT", interval="1h", limit=limit, ma_window=(limit*5)//100)
                    boids_df = boids.generate_boids_features(num_days=len(df),
                                                                num_boids=num_boids,
                                                                width=dimension,
                                                                height=dimension,
                                                                max_speed=max_speed,
                                                                perception_radius=perception_radius)
                    df_boids = pd.concat([df.reset_index(drop=True), boids_df.reset_index(drop=True)], axis=1)

                    df_boids.dropna(subset=feature_cols + [target_col], inplace=True)

                    scaler_X = MinMaxScaler()
                    scaler_y = MinMaxScaler()

                    df_boids[feature_cols] = scaler_X.fit_transform(df_boids[feature_cols])
                    df_boids[[target_col]] = scaler_y.fit_transform(df_boids[[target_col]])
                    target_min = df[target_col].min()
                    target_max = df[target_col].max()

                    with open(f'{limit}_{num_boids}_{dimension}_{max_speed}_{perception_radius}.json', "w") as f:
                        json.dump({"min": target_min, "max": target_max}, f)
                    df_boids[feature_cols + [target_col]].to_csv(processed_path, index=False)
                    print(f'1-st iteration completed: {limit}_{num_boids}_{dimension}_{max_speed}_{perception_radius}')



