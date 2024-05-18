import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt


# Calculates the time difference between eacha actions of the user
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

# Takes the latest data from 12th April 2024
def select_latest_data(df):
    df["date"] = pd.to_datetime(df["date"])

    start_date = pd.Timestamp(
        "2024-04-12"  # this is when we started collecting new data
    ).date()  # end_date = pd.Timestamp.today().date()

    return df[(df["date"].dt.date >= start_date)].copy()

#  Removes those users who didn't complete the quiz
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

# Keeps the last/final option clicked by the user and removes the other options in the page
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

#  Removes the other actions like "Continue", "Start", "Prompt" and "End" and keeps only the options
def extract_answers(df):
    """
    Filter the dataframe for only the final answers
    This will remove "continue" , "prompt", and "start" etc.
    Only final answers will be remaining.

    """

    grouped = df.groupby(["user", "page"])
    final_answers = grouped.apply(
        _extract_final_choice
    ).reset_index(drop=True)
    final_answers[["user", "page"]] = (
        grouped[["user", "page"]].first().reset_index(drop=True)
    )
    return final_answers

# Extract the correct answer and add it.
def extract_correct_score(df):
    """
    Extract the correct answer and add it.
    Assumes we have a df with 'page' and 'action' columns
    It compares the answer in action, with the correct answer for that page
    """
    correct_answers_mapped = [
        "A",
        "C",
        "D",
        "D",
        "D",
        "D",
        "C",
        "B",
        "C",
        "C",
        "C",
        "B",
    ]
    df["score"] = (
        df["action"] == df["page"].map(lambda x: correct_answers_mapped[x - 1])
    ).astype(int)
    return df

# Evaluates the score by taking the sum 
def get_user_scores(df):
    return df.groupby("user")["score"].sum()


# Statistical analysis on the total score
def stats(no_as, prompted, unprompted):
    # Assuming your three pandas series are named 'series1', 'series2', and 'series3'

    no_as = get_user_scores(no_as)
    prompted = get_user_scores(prompted)
    unprompted = get_user_scores(unprompted)

    print(f"prompted df {prompted.shape[0]}")
    print(f"unprompted df {unprompted.shape[0]}")
    print(f"no as df {no_as.shape[0]}")
    print()

    f_statistic, p_value = f_oneway(no_as, prompted, unprompted)

    print("F-statistic:", f_statistic)
    print("p-value:", p_value)
    print()

    data = pd.DataFrame(
        {
            "no_as": no_as,
            "prompted": prompted,
            "unprompted": unprompted,
        }
    )

    print('Means of the score')
    res = data.mean().plot.bar().get_figure()
    # set the y limit to 5
    plt.title('Mean Value of Scores across 3 groups')
    plt.xlabel('Groups')
    plt.ylabel('Mean Value')
    res.savefig("means.png")

    data_melted = pd.melt(data)
    data_melted.dropna(axis=0, inplace=True)

    tukey_results = pairwise_tukeyhsd(data_melted["value"], data_melted["variable"])
    print(tukey_results)

# Removes those users who completed test in less than 60 seconds and more than 1500 seconds
def time_removal(df, low, high):
    times = df.groupby("user")["date_diff"].sum()
    tt = times.astype(np.int64) // 10**9  # convert to seconds
    keepers = tt[tt > low].index
    keepers_h = tt[tt < high].index
    df = df[df["user"].isin(keepers_h)].copy()
    df = df[df["user"].isin(keepers)].copy()
    return df


def cleanup_data(df, low=60, high=1500):
    """
    This function will take the raw data and clean it up.
    It will:
    - Remove incomplete data
    - Select only the latest data
    - Extract only the final answers
    - Calculate the time difference between each action
    - add the correct score column

    """

    df = select_latest_data(df)
    df = remove_incomplete(df)
    df = extract_answers(df)
    df = find_date_time_diff(df)
    df = extract_correct_score(df)
    df = time_removal(df, low, high)

    return df

# This function will stratify the data into two groups- CRT and Math for further analysis
def stratify(df):
    """
    This function will stratify the data into two groups:
    - CRT
    - MATH
       """

    crt_df = df[df["page"] <=7]
    math_df = df[df["page"] > 7]

    return crt_df, math_df


# Functions for analysis of Math questions:

def stats_for_math(df):
    """
    this function for the analysis 
    we want for the maths questions
    """
    df_incorrect = df[(df['score'] == 0)]
    df_count = df_incorrect.groupby('page')['id'].nunique().reset_index()

    return df_count

# Plot to interpret or visualise how many people answered incorrectly
def plot_for_incorrect_math(no_as_count, prompted_count, unprompted_count):
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = np.arange(len(no_as_count['page']))

    plt.bar(index, no_as_count['id'], bar_width, label='No Assistance')
    plt.bar(index + bar_width, prompted_count['id'], bar_width, label='Prompted')
    plt.bar(index + 2 * bar_width, unprompted_count['id'], bar_width, label='Unprompted')

    plt.xlabel('Page')
    plt.ylabel('Count of Users who answered the Math questions incorrectly')
    plt.title('Number of users who answered incorrectly')
    plt.xticks(index + bar_width, no_as_count['page'])
    plt.legend()
    plt.savefig("incorrect_count_math.png")
    plt.show()
    
# Evalautes math questions score alone
def get_user_scores_math(df):
    return df.groupby("user")["score"].sum()

# calculates mean value for the Math questions
def math_mean_score(no_as_math, prompted_math, unprompted_math):

    no_as = get_user_scores(no_as_math)
    prompted = get_user_scores(prompted_math)
    unprompted = get_user_scores(unprompted_math)

    print(f"prompted df {prompted.shape[0]}")
    print(f"unprompted df {unprompted.shape[0]}")
    print(f"no as df {no_as.shape[0]}")
    print()

    data = pd.DataFrame(
    {
        "no_as": no_as.mean(),
        "prompted": prompted.mean(),
        "unprompted": unprompted.mean(),
    },
    index=[0]
    )

    print(f'no assistance mean: {no_as.mean()}')
    print(f'prompted mean: {prompted.mean()}')
    print(f'unprompted mean: {unprompted.mean()}')
    print()

    plot = data.mean().plot.bar().get_figure()
    plt.title('Mean Value for the Math questions across 3 groups')
    plt.xlabel('Groups')
    plt.ylabel('Mean Value for Math questions')
    plt.ylim(0, 6)
    plot.savefig("means_math.png")


# Code to show the plot with significant difference
def plot_showing_significance():
    import matplotlib.pyplot as plt


    group_means = [4.341772151898734, 3.8840579710144927, 3.5444444444444443]
    group_labels = ['no_as', 'prompted', 'unprompted']

    fig, ax = plt.subplots()
    bar_width = 0.5
    x = np.arange(len(group_labels))
    bars = ax.bar(x, group_means, bar_width)

    # Customize the bars for the significantly different groups
    bars[0].set_color('r')  # Set the color of the "no_as" bar to red
    bars[2].set_hatch('///')  # Use a hatched pattern for the "unprompted" bar

    # Add annotation or symbol to denote significant difference
    ax.annotate('*', xy=(2, 3.9), xytext=(2.2, 4.2),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Add labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_xlabel('Groups')
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Value of Scores across 3 groups')

    plt.tight_layout()
    plt.show()


# Plot fot average time users took to answer the math questions for unprompted and no-assistance group
def plot_for_time_analysis(no_as_math, unprompted_math):
    no_as_math.loc[:, 'date_diff_seconds'] = no_as_math['date_diff'].apply(lambda x: x.total_seconds())
    unprompted_math.loc[:, 'date_diff'] = unprompted_math['date_diff'].apply(lambda x: x.total_seconds())

    no_as_math['group'] = 'No Assistance'
    unprompted_math['group'] = 'Unprompted'

    combined_df = pd.concat([no_as_math, unprompted_math])

    grouped = combined_df.groupby(['page', 'group'])['date_diff_seconds'].mean().reset_index()

    pivot_table = grouped.pivot_table(index='page', columns='group', values='date_diff_seconds', aggfunc='mean')

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_table.plot(kind='bar', ax=ax, rot=0, legend=True)

    # Set plot title and labels
    ax.set_title('Average Time Spent on Each Page by Users from Both Groups')
    ax.set_xlabel('Page')
    ax.set_ylabel('Average Time Difference (seconds)')

    # Add legend
    ax.legend(title='Group', loc='upper right')

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    plt.show()