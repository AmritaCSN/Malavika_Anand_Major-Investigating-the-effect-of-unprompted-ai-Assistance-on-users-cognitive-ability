import pandas as pd


def find_date_time_diff(df):

    prompted_grouped_df = df.groupby("user")
    df = prompted_grouped_df.apply(lambda x: x.sort_values(["date", "time"]))
    # Group the DataFrame by 'user' and calculate the datetime difference
    date_diff = df.groupby(level="user")["date"].diff()
    # Fill the first NaN value with the initial datetime
    date_diff = date_diff.fillna(pd.Timedelta(0))
    df["date_diff"] = date_diff.values
    df.reset_index(drop=True, inplace=True)
    return df


def select_latest_data(df):
    df["date"] = pd.to_datetime(df["date"])

    start_date = pd.Timestamp(
        "2024-04-12"  # this is when we started collecting new data
    ).date()  # end_date = pd.Timestamp.today().date()

    return df[(df["date"].dt.date >= start_date)].copy()


def remove_incomplete(df):
    """
    will search the dataframe for people who completed all 12 pages of the quiz
    removes anyone who didn't do all 12 pages

    """

    # we can remove page 0, it's got no information for this
    df = df[df["page"] != 0]

    # remove all users who did not complete the quiz
    completed_users = df.groupby("user")["page"].max()
    completed_users = completed_users[completed_users == 12].index
    df = df[df["user"].isin(completed_users)]
    df.reset_index(drop=True, inplace=True)
    return df


def _extract_final_choice(x):
    """
    to be used in a groupby on "page" and "user"
    For each users page, does reverse search (starting at their last action)
    and selects the first valid answer (in reverse! so it's their final answer)
    """

    for action in x["action"].values[::-1]:  # reverse search through the actions
        if action in ("A", "B", "C", "D"):  # valid actions
            valid_index = x[x["action"] == action].index[
                -1
            ]  # -1 because we're reversing
            return x.loc[valid_index]


def extract_answers(df):
    """
    Filter the dataframe for only the final answers
    This will remove "continue" , "prompt", and "start" etc.
    Only final answers will be remaining.

    """

    grouped = df.groupby(["user", "page"])
    final_answers = grouped.apply(
        _extract_final_choice, include_groups=False
    ).reset_index(drop=True)
    final_answers[["user", "page"]] = (
        grouped[["user", "page"]].first().reset_index(drop=True)
    )
    return final_answers
