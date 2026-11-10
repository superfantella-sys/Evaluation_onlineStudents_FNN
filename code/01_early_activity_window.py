# Excerpt from the private training pipeline (train_capstone.py, early_activity).
# Keeps each learner's activity from their first recorded day through 14 days later.
# It is the core of the feature design. Not runnable on its own: `events` is the
# daily activity table and `keep_user_uuid` is not used here.

import pandas as pd

# Find each learner's first recorded activity date.
events["date"] = pd.to_datetime(events["date"])
first_day = events.groupby("user_uuid", as_index=False)["date"].min()
first_day = first_day.rename(columns={"date": "first_event_date"})
events = events.merge(first_day, on="user_uuid", how="left")

# Keep only activity on or before 14 days after that first date (inclusive).
events["cutoff_date"] = events["first_event_date"] + pd.Timedelta(days=14)
early = events.loc[events["date"] <= events["cutoff_date"]].copy()
