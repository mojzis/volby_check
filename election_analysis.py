import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import requests
    import zipfile
    import io
    return io, mo, pd, requests, zipfile


@app.cell
def __(mo):
    mo.md(
        """
        # Czech Election Data Analysis 2025

        Analyzing voting patterns to identify counting commissions where top parties received 0 votes.
        """
    )
    return


@app.cell
def __(io, pd, requests, zipfile):
    # Download the election data
    url = "https://www.volby.cz/opendata/ps2025/csv_od/pst4p.zip"

    print("Downloading election data...")
    response = requests.get(url)
    response.raise_for_status()

    # Unzip and load the data
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    csv_filename = zip_file.namelist()[0]

    print(f"Loading {csv_filename}...")
    with zip_file.open(csv_filename) as f:
        df = pd.read_csv(f, encoding='cp1250', delimiter=';')

    print(f"Data loaded: {len(df)} rows")
    df
    return csv_filename, df, response, url, zip_file


@app.cell
def __(df):
    # Show basic statistics
    print(f"Number of counting commissions: {df['ID_OKRSKY'].nunique()}")
    print(f"Number of parties: {df['KSTRANA'].nunique()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    df.head(10)
    return


@app.cell
def __(df, pd):
    # Calculate total votes per party across all commissions
    party_totals = df.groupby('KSTRANA')['POC_HLASU'].sum().sort_values(ascending=False)

    # Get total number of votes cast
    total_votes = party_totals.sum()

    # Calculate percentage for each party
    party_percentages = (party_totals / total_votes * 100).round(2)

    # Combine into a summary DataFrame
    party_summary = pd.DataFrame({
        'Total_Votes': party_totals,
        'Percentage': party_percentages
    })

    print("Top parties by vote share:")
    party_summary.head(10)
    return party_percentages, party_summary, party_totals, total_votes


@app.cell
def __(df, mo, party_summary):
    mo.md(
        f"""
        ## Identifying Top Parties

        We'll focus on the top 6-7 parties with the most votes:
        """
    )

    # Get top 7 parties
    top_parties = party_summary.head(7).index.tolist()
    print(f"Top 7 parties: {top_parties}")
    top_parties
    return (top_parties,)


@app.cell
def __(df, mo, pd, top_parties):
    mo.md(
        """
        ## Analysis: Commissions Where Top Parties Got 0 Votes

        For each of the top parties, finding counting commissions where they received 0 votes.
        """
    )

    # For each top party, find commissions where they got 0 votes
    zero_vote_analysis = []

    for party in top_parties:
        party_data = df[df['KSTRANA'] == party]

        # Find commissions where this party got 0 votes
        zero_vote_commissions = party_data[party_data['POC_HLASU'] == 0]['ID_OKRSKY'].unique()

        # For these commissions, calculate the total votes cast
        # (we need to sum all parties' votes for these commissions)
        for commission in zero_vote_commissions:
            commission_data = df[df['ID_OKRSKY'] == commission]
            total_votes_in_commission = commission_data['POC_HLASU'].sum()

            zero_vote_analysis.append({
                'Party': party,
                'Commission_ID': commission,
                'Total_Votes_In_Commission': total_votes_in_commission
            })

    zero_votes_df = pd.DataFrame(zero_vote_analysis)

    print(f"\nFound {len(zero_votes_df)} instances where top parties got 0 votes")
    zero_votes_df
    return (
        commission,
        commission_data,
        party,
        party_data,
        total_votes_in_commission,
        zero_vote_analysis,
        zero_vote_commissions,
        zero_votes_df,
    )


@app.cell
def __(mo, party_summary, zero_votes_df):
    mo.md(
        """
        ## Statistical Summary

        Comparing the size of commissions where parties got 0 votes
        """
    )

    # Group by party to see statistics
    summary_by_party = zero_votes_df.groupby('Party').agg({
        'Commission_ID': 'count',
        'Total_Votes_In_Commission': ['mean', 'median', 'min', 'max', 'std']
    }).round(2)

    summary_by_party.columns = ['Count_of_Zero_Vote_Commissions',
                                  'Mean_Commission_Size',
                                  'Median_Commission_Size',
                                  'Min_Commission_Size',
                                  'Max_Commission_Size',
                                  'Std_Commission_Size']

    # Add party percentage for context
    summary_by_party['Party_Percentage'] = summary_by_party.index.map(
        lambda x: party_summary.loc[x, 'Percentage']
    )

    print("Statistics of commissions where each top party got 0 votes:")
    summary_by_party
    return (summary_by_party,)


@app.cell
def __(df, mo):
    mo.md(
        """
        ## Overall Commission Size Distribution

        For context, let's look at the distribution of all commission sizes
        """
    )

    # Calculate total votes per commission
    commission_sizes = df.groupby('ID_OKRSKY')['POC_HLASU'].sum()

    print(f"Commission size statistics:")
    print(f"  Mean: {commission_sizes.mean():.2f}")
    print(f"  Median: {commission_sizes.median():.2f}")
    print(f"  Min: {commission_sizes.min():.2f}")
    print(f"  Max: {commission_sizes.max():.2f}")
    print(f"  Std Dev: {commission_sizes.std():.2f}")

    commission_sizes.describe()
    return (commission_sizes,)


@app.cell
def __(commission_sizes, mo, zero_votes_df):
    mo.md(
        """
        ## Key Findings

        Comparing whether commissions where parties got 0 votes are smaller or larger than average
        """
    )

    overall_mean = commission_sizes.mean()
    overall_median = commission_sizes.median()

    zero_vote_mean = zero_votes_df['Total_Votes_In_Commission'].mean()
    zero_vote_median = zero_votes_df['Total_Votes_In_Commission'].median()

    print(f"Average commission size (all): {overall_mean:.2f} votes")
    print(f"Median commission size (all): {overall_median:.2f} votes")
    print()
    print(f"Average commission size (where top parties got 0): {zero_vote_mean:.2f} votes")
    print(f"Median commission size (where top parties got 0): {zero_vote_median:.2f} votes")
    print()

    if zero_vote_mean < overall_mean:
        print(f"✓ Commissions where parties got 0 votes are SMALLER on average ({zero_vote_mean:.2f} vs {overall_mean:.2f})")
        print(f"  Difference: {overall_mean - zero_vote_mean:.2f} votes ({(1 - zero_vote_mean/overall_mean)*100:.1f}% smaller)")
    else:
        print(f"✗ Commissions where parties got 0 votes are LARGER on average ({zero_vote_mean:.2f} vs {overall_mean:.2f})")
        print(f"  Difference: {zero_vote_mean - overall_mean:.2f} votes ({(zero_vote_mean/overall_mean - 1)*100:.1f}% larger)")
    return overall_mean, overall_median, zero_vote_mean, zero_vote_median


if __name__ == "__main__":
    app.run()
