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
    from scipy import stats
    import numpy as np
    from plotly.subplots import make_subplots
    return (
        Path,
        go,
        io,
        make_subplots,
        mo,
        np,
        pd,
        px,
        requests,
        stats,
        zipfile,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # Suspicious Zero-Vote Analysis with Geographic Context

        ## Statistical Analysis with Municipality Size

        This analysis enhances the probability-based detection by adding geographic context:
        - Municipality (OBEC) size information
        - Party performance by municipality size category
        - Detection of suspicious patterns in areas where parties should be strong
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
        # Download the election data with headers to avoid 403
        url = "https://www.volby.cz/opendata/ps2025/csv_od/pst4p.zip"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Unzip and load the data
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        csv_filename = zip_file.namelist()[0]

        with zip_file.open(csv_filename) as f:
            df = pd.read_csv(f)

        # Save to parquet for future use
        df.to_parquet(parquet_file)

    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")
    df
    return df, parquet_file


@app.cell
def __(df, mo):
    mo.md(
        """
        ## Municipality (OBEC) Analysis

        Exploring the size distribution of municipalities and calculating
        voting patterns by municipality size.
        """
    )

    # Calculate municipality sizes based on total votes
    municipality_sizes = df.groupby('OBEC').agg({
        'POC_HLASU': 'sum',
        'ID_OKRSKY': 'nunique'
    }).reset_index()
    municipality_sizes.columns = ['OBEC', 'Total_Votes', 'Num_Commissions']

    # Categorize municipalities by size
    # Using quantiles to create balanced categories
    municipality_sizes['Size_Category'] = pd.qcut(
        municipality_sizes['Total_Votes'],
        q=5,
        labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'],
        duplicates='drop'
    )

    print("Municipality size distribution:")
    print(municipality_sizes.groupby('Size_Category', observed=False)['OBEC'].count())
    print("\nVotes by category:")
    print(municipality_sizes.groupby('Size_Category', observed=False)['Total_Votes'].agg(['min', 'max', 'mean']))

    municipality_sizes
    return (municipality_sizes,)


@app.cell
def __(df, municipality_sizes, pd):
    # Merge municipality size info back to main dataframe
    df_with_size = df.merge(
        municipality_sizes[['OBEC', 'Total_Votes', 'Size_Category']],
        on='OBEC',
        suffixes=('', '_Municipality')
    )

    # Calculate party performance by municipality size
    party_by_size = df_with_size.groupby(['KSTRANA', 'Size_Category'], observed=False)['POC_HLASU'].agg([
        'sum', 'count', 'mean'
    ]).reset_index()
    party_by_size.columns = ['Party', 'Size_Category', 'Total_Votes', 'Num_Appearances', 'Avg_Votes']

    # Calculate party's vote share in each category
    category_totals = party_by_size.groupby('Size_Category', observed=False)['Total_Votes'].transform('sum')
    party_by_size['Vote_Share_In_Category'] = party_by_size['Total_Votes'] / category_totals * 100

    party_by_size
    return df_with_size, party_by_size


@app.cell
def __(df_with_size, pd):
    # Calculate total votes per party across all commissions
    party_totals = df_with_size.groupby('KSTRANA')['POC_HLASU'].sum().sort_values(ascending=False)

    # Get total number of votes cast
    total_votes = party_totals.sum()

    # Calculate percentage for each party
    party_percentages = (party_totals / total_votes * 100).round(4)

    # Calculate probability (for binomial distribution)
    party_probabilities = party_totals / total_votes

    # Combine into a summary DataFrame
    party_summary = pd.DataFrame({
        'Total_Votes': party_totals,
        'Percentage': party_percentages,
        'Probability': party_probabilities
    })

    party_summary.head(10)
    return (
        party_percentages,
        party_probabilities,
        party_summary,
        party_totals,
        total_votes,
    )


@app.cell
def __(party_summary):
    # Get top 7 parties
    top_parties = party_summary.head(7).index.tolist()
    top_parties
    return (top_parties,)


@app.cell
def __(df_with_size, municipality_sizes, pd, party_by_size, top_parties):
    # Get all unique commissions and top parties
    all_commissions = df_with_size['ID_OKRSKY'].unique()

    # Create a DataFrame with all possible combinations of top parties and commissions
    all_combinations = pd.MultiIndex.from_product(
        [top_parties, all_commissions],
        names=['Party', 'Commission_ID']
    ).to_frame(index=False)

    # Get actual combinations present in the data
    actual_combinations = df_with_size[df_with_size['KSTRANA'].isin(top_parties)][
        ['KSTRANA', 'ID_OKRSKY']
    ].copy()
    actual_combinations.columns = ['Party', 'Commission_ID']
    actual_combinations['Present'] = True

    # Merge to find missing combinations (where parties got 0 votes)
    combined = all_combinations.merge(actual_combinations, on=['Party', 'Commission_ID'], how='left')
    zero_votes = combined[combined['Present'].isna()][['Party', 'Commission_ID']].copy()

    # Calculate commission info with OBEC and size category
    commission_info = df_with_size.groupby('ID_OKRSKY').agg({
        'POC_HLASU': 'sum',
        'OBEC': 'first',
        'Size_Category': 'first',
        'Total_Votes': 'first'
    }).reset_index()
    commission_info.columns = [
        'Commission_ID',
        'Total_Votes_In_Commission',
        'OBEC',
        'Municipality_Size_Category',
        'Municipality_Total_Votes'
    ]

    # Merge to get the full info for each commission where a top party is missing
    zero_votes_df = zero_votes.merge(commission_info, on='Commission_ID')

    # For each party-size category, calculate their expected probability
    # This accounts for the fact that parties perform differently in different size municipalities
    party_prob_by_size = party_by_size.copy()
    party_prob_by_size['Probability_In_Category'] = (
        party_prob_by_size['Vote_Share_In_Category'] / 100
    )

    # Merge party probabilities by size category
    zero_votes_df = zero_votes_df.merge(
        party_prob_by_size[['Party', 'Size_Category', 'Probability_In_Category', 'Vote_Share_In_Category']],
        left_on=['Party', 'Municipality_Size_Category'],
        right_on=['Party', 'Size_Category'],
        how='left'
    )

    zero_votes_df
    return (
        all_combinations,
        commission_info,
        party_prob_by_size,
        zero_votes_df,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ## Enhanced Probability Calculation

        Now using **size-adjusted probabilities**:

        **P(0 votes) = (1 - p_category)^n**

        Where:
        - p_category = party's vote probability in municipalities of this size
        - n = total votes in the commission

        This is more accurate than using overall national probability, as it accounts
        for the fact that parties perform differently in large cities vs small towns.
        """
    )
    return


@app.cell
def __(party_summary, pd, zero_votes_df):
    # For each zero-vote case, calculate the probability
    prob_analysis = zero_votes_df.copy()

    # Add overall party probability and percentage
    prob_analysis['Party_Probability_Overall'] = prob_analysis['Party'].map(
        lambda x: party_summary.loc[x, 'Probability']
    )
    prob_analysis['Party_Percentage'] = prob_analysis['Party'].map(
        lambda x: party_summary.loc[x, 'Percentage']
    )

    # Calculate probability using BOTH overall and category-specific probabilities
    # Overall probability
    prob_analysis['Probability_of_Zero_Overall'] = (
        1 - prob_analysis['Party_Probability_Overall']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Category-adjusted probability (more accurate)
    prob_analysis['Probability_of_Zero_Adjusted'] = (
        1 - prob_analysis['Probability_In_Category']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Use adjusted where available, otherwise fall back to overall
    prob_analysis['Probability_of_Zero'] = prob_analysis['Probability_of_Zero_Adjusted'].fillna(
        prob_analysis['Probability_of_Zero_Overall']
    )

    # Convert to percentage
    prob_analysis['Probability_of_Zero_Percent'] = prob_analysis['Probability_of_Zero'] * 100

    # Flag suspicious cases (less than 1% probability)
    prob_analysis['Is_Suspicious'] = prob_analysis['Probability_of_Zero'] < 0.01

    # Flag HIGHLY suspicious cases (less than 0.1%)
    prob_analysis['Is_Highly_Suspicious'] = prob_analysis['Probability_of_Zero'] < 0.001

    prob_analysis
    return (prob_analysis,)


@app.cell
def __(mo, prob_analysis):
    mo.md(
        """
        ## Top 3 Biggest Zero-Vote Commissions Per Party

        With municipality size context
        """
    )

    # Get top 3 biggest zero-vote commissions per party
    top3_per_party = (
        prob_analysis
        .sort_values(['Party', 'Total_Votes_In_Commission'], ascending=[True, False])
        .groupby('Party')
        .head(3)
        .sort_values(['Party_Percentage', 'Total_Votes_In_Commission'], ascending=[False, False])
    )

    # Create a display dataframe with relevant columns
    top3_display = top3_per_party[[
        'Party',
        'Party_Percentage',
        'Commission_ID',
        'OBEC',
        'Municipality_Size_Category',
        'Municipality_Total_Votes',
        'Total_Votes_In_Commission',
        'Vote_Share_In_Category',
        'Probability_of_Zero',
        'Probability_of_Zero_Percent',
        'Is_Suspicious',
        'Is_Highly_Suspicious'
    ]].copy()

    # Format the probability column for better readability
    top3_display['Probability_Display'] = top3_display['Probability_of_Zero'].apply(
        lambda x: f"{x:.2e}" if x < 0.0001 else f"{x*100:.4f}%"
    )

    top3_display
    return top3_display, top3_per_party


@app.cell
def __(mo, top3_per_party):
    mo.md(
        """
        ## Summary of Suspicious Cases by Municipality Size

        Breakdown by municipality size category
        """
    )

    suspicious_cases = top3_per_party[top3_per_party['Is_Suspicious']].copy()

    print(f"Found {len(suspicious_cases)} suspicious cases out of {len(top3_per_party)} top-3 cases")
    print(f"Suspicious rate: {len(suspicious_cases)/len(top3_per_party)*100:.1f}%\n")

    print("Breakdown by municipality size:")
    size_breakdown = suspicious_cases.groupby('Municipality_Size_Category', observed=False).agg({
        'Commission_ID': 'count',
        'Total_Votes_In_Commission': ['mean', 'min', 'max'],
        'Probability_of_Zero': 'mean'
    })
    print(size_breakdown)
    print()

    # Show most suspicious cases
    most_suspicious = suspicious_cases.sort_values('Probability_of_Zero').head(10)[[
        'Party',
        'Party_Percentage',
        'OBEC',
        'Municipality_Size_Category',
        'Total_Votes_In_Commission',
        'Vote_Share_In_Category',
        'Probability_of_Zero'
    ]]

    most_suspicious
    return most_suspicious, size_breakdown, suspicious_cases


@app.cell
def __(mo):
    mo.md(
        """
        # Interactive Visualizations

        Enhanced charts with municipality size context
        """
    )
    return


@app.cell
def __(mo, party_by_size, party_summary, px, top_parties):
    mo.md("### Party Performance by Municipality Size")

    # Filter to top parties
    party_by_size_top = party_by_size[party_by_size['Party'].isin(top_parties)].copy()

    # Add overall percentage for ordering
    party_by_size_top['Overall_Pct'] = party_by_size_top['Party'].map(
        lambda x: party_summary.loc[x, 'Percentage']
    )
    party_by_size_top = party_by_size_top.sort_values('Overall_Pct', ascending=False)

    # Create labels
    party_by_size_top['Party_Label'] = party_by_size_top.apply(
        lambda row: f"{row['Party']} ({row['Overall_Pct']:.1f}%)", axis=1
    )

    # Create grouped bar chart
    perf_fig = px.bar(
        party_by_size_top,
        x='Size_Category',
        y='Vote_Share_In_Category',
        color='Party_Label',
        barmode='group',
        title='Party Vote Share by Municipality Size Category',
        labels={
            'Size_Category': 'Municipality Size',
            'Vote_Share_In_Category': 'Vote Share (%)',
            'Party_Label': 'Party'
        },
        category_orders={
            'Size_Category': ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
        }
    )

    perf_fig.update_layout(height=600, xaxis_type='category')

    mo.ui.plotly(perf_fig)
    return party_by_size_top, perf_fig


@app.cell
def __(go, mo, municipality_sizes, prob_analysis, px, top_parties):
    mo.md("### Geographic Distribution: Suspicious Zeros by Municipality Size")

    # Create scatter plot showing all zero-vote cases
    geo_data = prob_analysis[prob_analysis['Party'].isin(top_parties)].copy()
    geo_data['Party'] = geo_data['Party'].astype(str)

    # Create suspiciousness categories
    geo_data['Suspiciousness'] = pd.cut(
        geo_data['Probability_of_Zero'],
        bins=[0, 0.001, 0.01, 0.1, 1.0],
        labels=['Highly Suspicious (<0.1%)', 'Suspicious (0.1-1%)', 'Unlikely (1-10%)', 'Possible (>10%)']
    )

    geo_scatter = px.scatter(
        geo_data,
        x='Municipality_Total_Votes',
        y='Total_Votes_In_Commission',
        color='Suspiciousness',
        size='Probability_of_Zero_Percent',
        hover_data=['Party', 'OBEC', 'Municipality_Size_Category'],
        title='Zero-Vote Commissions: Municipality Size vs Commission Size',
        labels={
            'Municipality_Total_Votes': 'Municipality Total Votes',
            'Total_Votes_In_Commission': 'Commission Size',
            'Suspiciousness': 'Suspiciousness Level'
        },
        color_discrete_map={
            'Highly Suspicious (<0.1%)': 'darkred',
            'Suspicious (0.1-1%)': 'red',
            'Unlikely (1-10%)': 'orange',
            'Possible (>10%)': 'lightblue'
        },
        log_x=True
    )

    geo_scatter.update_layout(height=600)

    mo.ui.plotly(geo_scatter)
    return geo_data, geo_scatter


@app.cell
def __(go, mo, np, top3_per_party):
    mo.md("### Enhanced Probability Heatmap with Municipality Context")

    # Create a more detailed table for the heatmap
    heatmap_data = top3_per_party.copy()

    # Add party labels with percentages
    heatmap_data['Party_Label'] = heatmap_data.apply(
        lambda row: f"{row['Party']} ({row['Party_Percentage']:.1f}%)", axis=1
    )

    # Add rank within each party
    heatmap_data['Rank'] = heatmap_data.groupby('Party').cumcount() + 1
    heatmap_data['Rank_Label'] = 'Top ' + heatmap_data['Rank'].astype(str)

    # Sort parties by percentage
    party_order = heatmap_data.groupby('Party_Label')['Party_Percentage'].first().sort_values(ascending=False).index.tolist()

    # Create pivot table for heatmap using log scale
    heatmap_data['Log_Probability'] = -np.log10(heatmap_data['Probability_of_Zero'] + 1e-100)

    pivot_data = heatmap_data.pivot(
        index='Party_Label',
        columns='Rank_Label',
        values='Log_Probability'
    )

    # Reorder rows by party percentage
    pivot_data = pivot_data.reindex(party_order)

    # Create custom text for hover
    hover_text = []
    for party_label in pivot_data.index:
        row_text = []
        for rank in ['Top 1', 'Top 2', 'Top 3']:
            row_data = heatmap_data[
                (heatmap_data['Party_Label'] == party_label) &
                (heatmap_data['Rank_Label'] == rank)
            ]
            if len(row_data) > 0:
                row = row_data.iloc[0]
                text = (
                    f"Party: {party_label}<br>"
                    f"OBEC: {row['OBEC']}<br>"
                    f"Municipality: {row['Municipality_Size_Category']}<br>"
                    f"Commission: {row['Commission_ID']}<br>"
                    f"Commission Size: {row['Total_Votes_In_Commission']} votes<br>"
                    f"Party's vote share in {row['Municipality_Size_Category']}: {row['Vote_Share_In_Category']:.2f}%<br>"
                    f"P(0 votes): {row['Probability_of_Zero']:.2e}<br>"
                    f"Suspicious: {'YES' if row['Is_Suspicious'] else 'No'}"
                )
            else:
                text = "N/A"
            row_text.append(text)
        hover_text.append(row_text)

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=['Top 1 (Biggest)', 'Top 2', 'Top 3'],
        y=pivot_data.index,
        colorscale='Reds',
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title='-log₁₀(P)',
            tickmode='linear',
            tick0=0,
            dtick=10
        )
    ))

    heatmap_fig.update_layout(
        title='Probability of Zero Votes: -log₁₀(P) Scale with Municipality Context<br>(Higher = More Suspicious)',
        xaxis_title='Commission Rank by Size',
        yaxis_title='Party (Vote %)',
        height=500,
        yaxis_type='category',
        xaxis_type='category'
    )

    mo.ui.plotly(heatmap_fig)
    return (
        heatmap_data,
        heatmap_fig,
        hover_text,
        party_order,
        pivot_data,
    )


@app.cell
def __(go, mo, make_subplots, top3_per_party):
    mo.md("### Comparison: Overall vs Size-Adjusted Probabilities")

    # Compare the two probability calculations
    comparison_data = top3_per_party[
        top3_per_party['Probability_of_Zero_Overall'].notna() &
        top3_per_party['Probability_of_Zero_Adjusted'].notna()
    ].copy()

    comparison_data['Party_Label'] = comparison_data.apply(
        lambda row: f"{row['Party']} ({row['Party_Percentage']:.1f}%)", axis=1
    )

    # Calculate difference
    comparison_data['Probability_Ratio'] = (
        comparison_data['Probability_of_Zero_Adjusted'] /
        comparison_data['Probability_of_Zero_Overall']
    )

    # Create subplot with two charts
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Overall Probability', 'Size-Adjusted Probability'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    # Chart 1: Overall
    fig.add_trace(
        go.Scatter(
            x=comparison_data['Total_Votes_In_Commission'],
            y=comparison_data['Probability_of_Zero_Overall'],
            mode='markers',
            name='Overall',
            marker=dict(
                size=10,
                color=comparison_data['Party_Percentage'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Party %', x=0.46)
            ),
            text=comparison_data['Party'],
            hovertemplate='Party: %{text}<br>Commission Size: %{x}<br>P(0): %{y:.2e}<extra></extra>'
        ),
        row=1, col=1
    )

    # Chart 2: Adjusted
    fig.add_trace(
        go.Scatter(
            x=comparison_data['Total_Votes_In_Commission'],
            y=comparison_data['Probability_of_Zero_Adjusted'],
            mode='markers',
            name='Adjusted',
            marker=dict(
                size=10,
                color=comparison_data['Party_Percentage'],
                colorscale='Viridis',
                showscale=False
            ),
            text=comparison_data['Party'],
            hovertemplate='Party: %{text}<br>Commission Size: %{x}<br>P(0): %{y:.2e}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text='Commission Size', row=1, col=1)
    fig.update_xaxes(title_text='Commission Size', row=1, col=2)
    fig.update_yaxes(title_text='Probability of 0 Votes', type='log', row=1, col=1)
    fig.update_yaxes(title_text='Probability of 0 Votes', type='log', row=1, col=2)

    fig.update_layout(
        title_text='Comparing Probability Calculations',
        height=500,
        showlegend=False
    )

    mo.ui.plotly(fig)
    return comparison_data, fig


@app.cell
def __(mo, px, suspicious_cases):
    mo.md("### Suspicious Cases: Distribution by Municipality Size")

    # Create distribution of suspicious cases by size category
    susp_by_size = suspicious_cases.groupby('Municipality_Size_Category', observed=False).size().reset_index(name='Count')

    size_dist_fig = px.bar(
        susp_by_size,
        x='Municipality_Size_Category',
        y='Count',
        title='Number of Suspicious Zero-Vote Cases by Municipality Size',
        labels={
            'Municipality_Size_Category': 'Municipality Size',
            'Count': 'Number of Suspicious Cases'
        },
        category_orders={
            'Municipality_Size_Category': ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
        },
        text='Count'
    )

    size_dist_fig.update_traces(textposition='outside')
    size_dist_fig.update_layout(height=500, xaxis_type='category')

    mo.ui.plotly(size_dist_fig)
    return size_dist_fig, susp_by_size


@app.cell
def __(mo):
    mo.md(
        """
        ## Key Insights with Geographic Context

        ### What the Enhanced Analysis Shows:

        1. **Size-Adjusted Probabilities**: Parties perform differently in cities vs towns.
           A zero in a large city for an urban party is MORE suspicious than using
           overall national probability would suggest.

        2. **Geographic Patterns**: If suspicious zeros cluster in specific municipality
           sizes, it may indicate systematic issues rather than random anomalies.

        3. **Expected vs Reality**: The comparison charts show where actual results
           deviate most from what we'd expect given the party's performance in
           similar-sized municipalities.

        ### Red Flags to Look For:

        - **Urban party getting 0 in large cities**: Highly suspicious
        - **Rural party getting 0 in small towns**: Highly suspicious
        - **Clusters of suspicious zeros** in same size category
        - **Very low probabilities** (< 0.1%): Nearly impossible by chance

        ### Interpretation:

        - **Municipality size matters**: Context is crucial for assessing suspiciousness
        - **Size-adjusted probabilities** are more accurate than overall national stats
        - **Pattern detection**: Look for systematic issues, not just individual cases
        """
    )
    return


if __name__ == "__main__":
    app.run()
