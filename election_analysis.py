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
    from pathlib import Path
    import plotly.express as px
    import plotly.graph_objects as go
    return Path, go, io, mo, pd, px, requests, zipfile


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
def __(Path, io, pd, requests, zipfile):
    # Cache file path
    parquet_file = Path("election_data.parquet")

    # Check if cached data exists
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    else:
        # Download the election data
        url = "https://www.volby.cz/opendata/ps2025/csv_od/pst4p.zip"
        response = requests.get(url)
        response.raise_for_status()

        # Unzip and load the data
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        csv_filename = zip_file.namelist()[0]

        with zip_file.open(csv_filename) as f:
            df = pd.read_csv(f)

        # Save to parquet for future use
        df.to_parquet(parquet_file)

    df
    return df, parquet_file


@app.cell
def __(df):
    # Show basic statistics
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

    party_summary.head(10)
    return party_percentages, party_summary, party_totals, total_votes


@app.cell
def __(mo):
    mo.md(
        """
        ## Identifying Top Parties

        We'll focus on the top 6-7 parties with the most votes.
        """
    )
    return


@app.cell
def __(party_summary):
    # Get top 7 parties
    top_parties = party_summary.head(7).index.tolist()
    top_parties
    return (top_parties,)


@app.cell
def __(mo):
    mo.md(
        """
        ## Analysis: Commissions Where Top Parties Got 0 Votes

        For each of the top parties, finding counting commissions where they received 0 votes.
        """
    )
    return


@app.cell
def __(df, pd, top_parties):
    # Get all unique commissions and top parties
    all_commissions = df['ID_OKRSKY'].unique()

    # Create a DataFrame with all possible combinations of top parties and commissions
    all_combinations = pd.MultiIndex.from_product(
        [top_parties, all_commissions],
        names=['Party', 'Commission_ID']
    ).to_frame(index=False)

    # Get actual combinations present in the data
    actual_combinations = df[df['KSTRANA'].isin(top_parties)][['KSTRANA', 'ID_OKRSKY']].copy()
    actual_combinations.columns = ['Party', 'Commission_ID']
    actual_combinations['Present'] = True

    # Merge to find missing combinations (where parties got 0 votes)
    combined = all_combinations.merge(actual_combinations, on=['Party', 'Commission_ID'], how='left')
    zero_votes = combined[combined['Present'].isna()][['Party', 'Commission_ID']].copy()

    # Calculate total votes per commission (once, for all commissions)
    commission_totals = df.groupby('ID_OKRSKY')['POC_HLASU'].sum().reset_index()
    commission_totals.columns = ['Commission_ID', 'Total_Votes_In_Commission']

    # Merge to get the total votes for each commission where a top party is missing
    zero_votes_df = zero_votes.merge(commission_totals, on='Commission_ID')

    zero_votes_df
    return all_combinations, zero_votes_df


@app.cell
def __(df, mo, party_summary, top_parties, zero_votes_df):
    mo.md(
        """
        ## Per-Party Detailed Analysis

        Comprehensive statistics for each top party
        """
    )

    # Calculate average votes per commission for each party (only commissions where they appeared)
    party_avg_votes = df[df['KSTRANA'].isin(top_parties)].groupby('KSTRANA')['POC_HLASU'].mean()

    # Group by party to see zero-vote commission statistics
    zero_vote_stats = zero_votes_df.groupby('Party').agg({
        'Commission_ID': 'count',
        'Total_Votes_In_Commission': ['mean', 'median', 'min', 'max', 'std']
    }).round(2)

    zero_vote_stats.columns = ['Count_of_Zero_Vote_Commissions',
                                  'Mean_Commission_Size',
                                  'Median_Commission_Size',
                                  'Min_Commission_Size',
                                  'Max_Commission_Size',
                                  'Std_Commission_Size']

    # Add overall average votes per commission for each party
    zero_vote_stats['Avg_Votes_Per_Commission'] = zero_vote_stats.index.map(
        lambda x: party_avg_votes.get(x, 0)
    ).round(2)

    # Add party percentage for context
    zero_vote_stats['Party_Percentage'] = zero_vote_stats.index.map(
        lambda x: party_summary.loc[x, 'Percentage']
    )

    # Reorder columns for better readability
    zero_vote_stats = zero_vote_stats[[
        'Party_Percentage',
        'Avg_Votes_Per_Commission',
        'Count_of_Zero_Vote_Commissions',
        'Max_Commission_Size',
        'Mean_Commission_Size',
        'Median_Commission_Size',
        'Min_Commission_Size',
        'Std_Commission_Size'
    ]]

    zero_vote_stats
    return party_avg_votes, zero_vote_stats


@app.cell
def __(mo, zero_votes_df):
    mo.md(
        """
        ## Biggest Commissions with 0 Votes per Party

        For each party, showing the largest commission where they received 0 votes
        """
    )

    # Find the biggest commission with 0 votes for each party
    biggest_zero_vote_commissions = zero_votes_df.loc[
        zero_votes_df.groupby('Party')['Total_Votes_In_Commission'].idxmax()
    ][['Party', 'Commission_ID', 'Total_Votes_In_Commission']].sort_values(
        'Total_Votes_In_Commission', ascending=False
    )

    biggest_zero_vote_commissions
    return (biggest_zero_vote_commissions,)


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


@app.cell
def __(mo):
    mo.md(
        """
        # Interactive Visualizations

        Explore the data through interactive charts
        """
    )
    return


@app.cell
def __(mo, party_avg_votes, px, zero_vote_stats):
    mo.md("### Average Votes per Commission by Party")

    # Create a bar chart showing average votes per commission
    avg_votes_chart = px.bar(
        x=party_avg_votes.index.astype(str),
        y=party_avg_votes.values,
        labels={'x': 'Party Code', 'y': 'Average Votes per Commission'},
        title='Average Votes per Commission for Top Parties',
        text=party_avg_votes.values.round(1)
    )
    avg_votes_chart.update_traces(textposition='outside')
    avg_votes_chart.update_layout(showlegend=False, height=500, xaxis_type='category')

    mo.ui.plotly(avg_votes_chart)
    return (avg_votes_chart,)


@app.cell
def __(mo, pd, px, zero_votes_df):
    mo.md("### Distribution of Commission Sizes with 0 Votes per Party")

    # Create a box plot showing the distribution of commission sizes where parties got 0 votes
    # Convert Party to string to ensure categorical treatment
    box_data = zero_votes_df.copy()
    box_data['Party'] = box_data['Party'].astype(str)

    box_chart = px.box(
        box_data,
        x='Party',
        y='Total_Votes_In_Commission',
        title='Distribution of Commission Sizes Where Parties Received 0 Votes',
        labels={'Party': 'Party Code', 'Total_Votes_In_Commission': 'Commission Size (Total Votes)'},
        points='outliers'
    )
    box_chart.update_layout(height=600, xaxis_type='category')

    mo.ui.plotly(box_chart)
    return box_chart, box_data


@app.cell
def __(mo, px, zero_votes_df):
    mo.md("### Histogram: Commission Sizes with 0 Votes by Party")

    # Create histogram showing distribution of commission sizes for each party
    # Convert Party to string to ensure categorical treatment
    hist_data = zero_votes_df.copy()
    hist_data['Party'] = hist_data['Party'].astype(str)

    histogram_chart = px.histogram(
        hist_data,
        x='Total_Votes_In_Commission',
        color='Party',
        title='Distribution of Commission Sizes Where Parties Received 0 Votes',
        labels={'Total_Votes_In_Commission': 'Commission Size (Total Votes)', 'Party': 'Party Code'},
        nbins=50,
        opacity=0.7
    )
    histogram_chart.update_layout(
        barmode='overlay',
        height=600,
        xaxis_title='Commission Size (Total Votes)',
        yaxis_title='Count'
    )

    mo.ui.plotly(histogram_chart)
    return hist_data, histogram_chart


@app.cell
def __(go, mo, zero_vote_stats):
    mo.md("### Comparison: Party Performance vs Zero-Vote Commissions")

    # Create a scatter plot comparing party percentage vs zero-vote commission characteristics
    scatter_chart = go.Figure()

    scatter_chart.add_trace(go.Scatter(
        x=zero_vote_stats['Party_Percentage'],
        y=zero_vote_stats['Count_of_Zero_Vote_Commissions'],
        mode='markers+text',
        text=zero_vote_stats.index.astype(str),
        textposition='top center',
        marker=dict(
            size=zero_vote_stats['Max_Commission_Size'] / 10,
            color=zero_vote_stats['Mean_Commission_Size'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Mean Commission<br>Size (votes)')
        ),
        name='Parties'
    ))

    scatter_chart.update_layout(
        title='Party Support vs Number of Zero-Vote Commissions',
        xaxis_title='Party Vote Percentage (%)',
        yaxis_title='Number of Commissions with 0 Votes',
        height=600,
        hovermode='closest'
    )

    scatter_chart.update_traces(
        hovertemplate='<b>%{text}</b><br>' +
                      'Vote %: %{x:.2f}%<br>' +
                      'Zero-vote commissions: %{y}<br>' +
                      '<extra></extra>'
    )

    mo.ui.plotly(scatter_chart)
    return (scatter_chart,)


@app.cell
def __(biggest_zero_vote_commissions, go, mo):
    mo.md("### Biggest Commissions with 0 Votes by Party")

    # Create a horizontal bar chart showing the biggest commission with 0 votes for each party
    # Convert Party to string to ensure categorical treatment
    biggest_data = biggest_zero_vote_commissions.copy()
    biggest_data['Party'] = biggest_data['Party'].astype(str)

    biggest_commission_chart = go.Figure(go.Bar(
        x=biggest_data['Total_Votes_In_Commission'],
        y=biggest_data['Party'],
        orientation='h',
        text=biggest_data['Total_Votes_In_Commission'],
        textposition='outside',
        marker=dict(
            color=biggest_data['Total_Votes_In_Commission'],
            colorscale='Reds',
            showscale=False
        ),
        hovertemplate='<b>%{y}</b><br>' +
                      'Commission ID: ' + biggest_data['Commission_ID'].astype(str) + '<br>' +
                      'Total Votes: %{x}<br>' +
                      '<extra></extra>'
    ))

    biggest_commission_chart.update_layout(
        title='Largest Commission Where Each Party Received 0 Votes',
        xaxis_title='Commission Size (Total Votes)',
        yaxis_title='Party Code',
        yaxis_type='category',
        height=500
    )

    mo.ui.plotly(biggest_commission_chart)
    return biggest_commission_chart, biggest_data


if __name__ == "__main__":
    app.run()
