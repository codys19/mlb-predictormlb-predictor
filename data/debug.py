import pandas as pd

games = pd.read_csv("raw/game_results.csv")

print("Shape:", games.shape)
print("\nColumns:", list(games.columns))
print("\nFirst 3 rows:")
print(games.head(3).to_string())
print("\nhome_team_won unique values:", games["home_team_won"].unique()[:10])
print("home_away unique values:", games["home_away"].unique()[:10])
print("result unique values:", games["result"].unique()[:10])
print("\nNull counts:")
print(games[["date","home_team_won","home_away","result","runs_scored","runs_allowed"]].isnull().sum())