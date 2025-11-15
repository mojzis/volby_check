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
    mo.md("# Zero-Vote Analysis with Geographic Context")
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
    mo.md("## Municipality (OBEC) Analysis")

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
    print(municipality_sizes.groupby('Size_Category')['OBEC'].count())
    print("\nVotes by category:")
    print(municipality_sizes.groupby('Size_Category')['Total_Votes'].agg(['min', 'max', 'mean']))

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
    party_by_size = df_with_size.groupby(['KSTRANA', 'Size_Category'])['POC_HLASU'].agg([
        'sum', 'count', 'mean'
    ]).reset_index()
    party_by_size.columns = ['Party', 'Size_Category', 'Total_Votes', 'Num_Appearances', 'Avg_Votes']

    # Calculate party's vote share in each category
    category_totals = party_by_size.groupby('Size_Category')['Total_Votes'].transform('sum')
    party_by_size['Vote_Share_In_Category'] = party_by_size['Total_Votes'] / category_totals * 100

    party_by_size
    return df_with_size, party_by_size


@app.cell
def __(df_with_size, pd):
    # Calculate party performance by OBEC (municipality-specific)
    party_by_obec = df_with_size.groupby(['KSTRANA', 'OBEC'])['POC_HLASU'].agg([
        'sum', 'count'
    ]).reset_index()
    party_by_obec.columns = ['Party', 'OBEC', 'Total_Votes', 'Num_Appearances']

    # Calculate total votes per OBEC to get party's share in each municipality
    obec_totals = df_with_size.groupby('OBEC')['POC_HLASU'].sum().reset_index()
    obec_totals.columns = ['OBEC', 'OBEC_Total_Votes']

    party_by_obec = party_by_obec.merge(obec_totals, on='OBEC')
    party_by_obec['Vote_Share_In_OBEC'] = party_by_obec['Total_Votes'] / party_by_obec['OBEC_Total_Votes'] * 100
    party_by_obec['Probability_In_OBEC'] = party_by_obec['Vote_Share_In_OBEC'] / 100

    party_by_obec
    return (party_by_obec,)


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
def __(df_with_size, municipality_sizes, pd, party_by_obec, party_by_size, top_parties):
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

    # Merge OBEC-specific party probabilities (most accurate)
    zero_votes_df = zero_votes_df.merge(
        party_by_obec[['Party', 'OBEC', 'Vote_Share_In_OBEC', 'Probability_In_OBEC']],
        on=['Party', 'OBEC'],
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
    mo.md("## Three-Tier Probability Calculation")
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

    # Calculate probability using three tiers: Overall → Size-Category → OBEC-specific
    # Tier 1: Overall probability (baseline)
    prob_analysis['Probability_of_Zero_Overall'] = (
        1 - prob_analysis['Party_Probability_Overall']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Tier 2: Size-category adjusted probability (better)
    prob_analysis['Probability_of_Zero_Size_Adjusted'] = (
        1 - prob_analysis['Probability_In_Category']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Tier 3: OBEC-specific adjusted probability (most accurate)
    prob_analysis['Probability_of_Zero_OBEC_Adjusted'] = (
        1 - prob_analysis['Probability_In_OBEC']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Use the most specific probability available with fallback hierarchy:
    # OBEC-specific → Size-Category → Overall
    prob_analysis['Probability_of_Zero'] = (
        prob_analysis['Probability_of_Zero_OBEC_Adjusted']
        .fillna(prob_analysis['Probability_of_Zero_Size_Adjusted'])
        .fillna(prob_analysis['Probability_of_Zero_Overall'])
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
    mo.md("## Top 3 Biggest Zero-Vote Commissions Per Party")

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
        'Vote_Share_In_OBEC',
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
    mo.md("## Summary of Suspicious Cases")

    suspicious_cases = top3_per_party[top3_per_party['Is_Suspicious']].copy()

    print(f"Found {len(suspicious_cases)} suspicious cases out of {len(top3_per_party)} top-3 cases")
    print(f"Suspicious rate: {len(suspicious_cases)/len(top3_per_party)*100:.1f}%\n")

    print("Breakdown by municipality size:")
    size_breakdown = suspicious_cases.groupby('Municipality_Size_Category').agg({
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
        'Vote_Share_In_OBEC',
        'Probability_of_Zero'
    ]]

    most_suspicious
    return most_suspicious, size_breakdown, suspicious_cases


@app.cell
def __(mo):
    mo.md("# Interactive Visualizations")
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
def __(go, mo, municipality_sizes, np, prob_analysis, px, top_parties):
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

    # Create inverted size metric: more suspicious (lower probability) = larger bubble
    # Using -log10(probability) so that very small probabilities give large sizes
    geo_data['Bubble_Size'] = -np.log10(geo_data['Probability_of_Zero'] + 1e-100)

    geo_scatter = px.scatter(
        geo_data,
        x='Municipality_Total_Votes',
        y='Total_Votes_In_Commission',
        color='Suspiciousness',
        size='Bubble_Size',
        hover_data=['Party', 'OBEC', 'Municipality_Size_Category', 'Vote_Share_In_OBEC', 'Probability_of_Zero_Percent'],
        title='Zero-Vote Commissions: Municipality Size vs Commission Size (OBEC-Adjusted)',
        labels={
            'Municipality_Total_Votes': 'Municipality Total Votes',
            'Total_Votes_In_Commission': 'Commission Size',
            'Suspiciousness': 'Suspiciousness Level',
            'Vote_Share_In_OBEC': 'Party % in OBEC'
        },
        color_discrete_map={
            'Highly Suspicious (<0.1%)': 'darkred',
            'Suspicious (0.1-1%)': 'red',
            'Unlikely (1-10%)': 'orange',
            'Possible (>10%)': 'lightblue'
        },
        size_max=30,  # Maximum bubble size
        log_x=True
    )

    # Set minimum bubble size
    geo_scatter.update_traces(marker=dict(sizemin=5))
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
                obec_share = row.get('Vote_Share_In_OBEC', None)
                obec_share_text = f"{obec_share:.2f}%" if pd.notna(obec_share) else "N/A"
                text = (
                    f"Party: {party_label}<br>"
                    f"OBEC: {row['OBEC']}<br>"
                    f"Municipality: {row['Municipality_Size_Category']}<br>"
                    f"Commission: {row['Commission_ID']}<br>"
                    f"Commission Size: {row['Total_Votes_In_Commission']} votes<br>"
                    f"Party's share in this OBEC: {obec_share_text}<br>"
                    f"Party's share in {row['Municipality_Size_Category']}: {row['Vote_Share_In_Category']:.2f}%<br>"
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
    mo.md("### Comparison: Overall vs Size-Adjusted vs OBEC-Adjusted Probabilities")

    # Compare the three probability calculations
    comparison_data = top3_per_party[
        top3_per_party['Probability_of_Zero_Overall'].notna()
    ].copy()

    comparison_data['Party_Label'] = comparison_data.apply(
        lambda row: f"{row['Party']} ({row['Party_Percentage']:.1f}%)", axis=1
    )

    # Create subplot with three charts
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Overall Probability', 'Size-Adjusted', 'OBEC-Adjusted (Most Accurate)'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
        horizontal_spacing=0.08
    )

    # Chart 1: Overall
    fig.add_trace(
        go.Scatter(
            x=comparison_data['Total_Votes_In_Commission'],
            y=comparison_data['Probability_of_Zero_Overall'],
            mode='markers',
            name='Overall',
            marker=dict(
                size=8,
                color=comparison_data['Party_Percentage'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Party %', x=1.02)
            ),
            text=comparison_data['Party'],
            hovertemplate='Party: %{text}<br>Commission Size: %{x}<br>P(0): %{y:.2e}<extra></extra>'
        ),
        row=1, col=1
    )

    # Chart 2: Size-Adjusted
    fig.add_trace(
        go.Scatter(
            x=comparison_data['Total_Votes_In_Commission'],
            y=comparison_data['Probability_of_Zero_Size_Adjusted'],
            mode='markers',
            name='Size-Adjusted',
            marker=dict(
                size=8,
                color=comparison_data['Party_Percentage'],
                colorscale='Viridis',
                showscale=False
            ),
            text=comparison_data['Party'],
            hovertemplate='Party: %{text}<br>Commission Size: %{x}<br>P(0): %{y:.2e}<extra></extra>'
        ),
        row=1, col=2
    )

    # Chart 3: OBEC-Adjusted
    fig.add_trace(
        go.Scatter(
            x=comparison_data['Total_Votes_In_Commission'],
            y=comparison_data['Probability_of_Zero_OBEC_Adjusted'],
            mode='markers',
            name='OBEC-Adjusted',
            marker=dict(
                size=8,
                color=comparison_data['Party_Percentage'],
                colorscale='Viridis',
                showscale=False
            ),
            text=comparison_data['Party'],
            hovertemplate='Party: %{text}<br>Commission Size: %{x}<br>P(0): %{y:.2e}<extra></extra>'
        ),
        row=1, col=3
    )

    fig.update_xaxes(title_text='Commission Size', row=1, col=1)
    fig.update_xaxes(title_text='Commission Size', row=1, col=2)
    fig.update_xaxes(title_text='Commission Size', row=1, col=3)
    fig.update_yaxes(title_text='Probability of 0 Votes', type='log', row=1, col=1)
    fig.update_yaxes(title_text='Probability of 0 Votes', type='log', row=1, col=2)
    fig.update_yaxes(title_text='Probability of 0 Votes', type='log', row=1, col=3)

    fig.update_layout(
        title_text='Comparing Three Tiers of Probability Calculations',
        height=500,
        showlegend=False
    )

    mo.ui.plotly(fig)
    return comparison_data, fig


@app.cell
def __(mo, px, suspicious_cases):
    mo.md("### Suspicious Cases: Distribution by Municipality Size")

    # Create distribution of suspicious cases by size category
    susp_by_size = suspicious_cases.groupby('Municipality_Size_Category').size().reset_index(name='Count')

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


if __name__ == "__main__":
    app.run()
