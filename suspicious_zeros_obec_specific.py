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
        # OBEC-Specific Zero-Vote Analysis

        ## The Most Accurate Probability Model

        This analysis uses **municipality-specific (OBEC) probabilities** - the most granular and accurate approach:

        ### Three Levels of Accuracy:
        1. **Overall National Probability**: Party's overall vote share across the country
        2. **Size-Category Adjusted**: Party's performance in similar-sized municipalities
        3. **OBEC-Specific** (this notebook): Party's actual performance in the exact municipality

        ### Why OBEC-Specific is Best:
        - A party might be very unpopular in Prague (capital) but strong in rural areas
        - Getting 0 votes in one Prague commission when the party gets <1% there overall is NOT suspicious
        - But getting 0 votes in a small town where they usually get 15% is HIGHLY suspicious
        - This captures local political preferences that size categories miss
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
        ## Municipality (OBEC) Characterization

        First, let's understand the municipalities and categorize them by size.
        """
    )

    # Calculate municipality sizes based on total votes
    municipality_sizes = df.groupby('OBEC').agg({
        'POC_HLASU': 'sum',
        'ID_OKRSKY': 'nunique'
    }).reset_index()
    municipality_sizes.columns = ['OBEC', 'Total_Votes', 'Num_Commissions']

    # Categorize municipalities by size
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

    df_with_size
    return (df_with_size,)


@app.cell
def __(df_with_size, pd):
    # Calculate party performance by OBEC (municipality-specific)
    party_by_obec = df_with_size.groupby(['KSTRANA', 'OBEC'])['POC_HLASU'].agg([
        'sum', 'count'
    ]).reset_index()
    party_by_obec.columns = ['Party', 'OBEC', 'Total_Votes', 'Num_Commissions']

    # Calculate total votes per OBEC to get party's share in each municipality
    obec_totals = df_with_size.groupby('OBEC')['POC_HLASU'].sum().reset_index()
    obec_totals.columns = ['OBEC', 'OBEC_Total_Votes']

    party_by_obec = party_by_obec.merge(obec_totals, on='OBEC')
    party_by_obec['Vote_Share_In_OBEC'] = party_by_obec['Total_Votes'] / party_by_obec['OBEC_Total_Votes'] * 100
    party_by_obec['Probability_In_OBEC'] = party_by_obec['Vote_Share_In_OBEC'] / 100

    print(f"Calculated party performance in {party_by_obec['OBEC'].nunique()} municipalities")
    print(f"Total party-municipality combinations: {len(party_by_obec)}")

    party_by_obec.head(20)
    return (party_by_obec,)


@app.cell
def __(df_with_size, pd):
    # Calculate overall party statistics
    party_totals = df_with_size.groupby('KSTRANA')['POC_HLASU'].sum().sort_values(ascending=False)
    total_votes = party_totals.sum()
    party_percentages = (party_totals / total_votes * 100).round(4)
    party_probabilities = party_totals / total_votes

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
def __(df_with_size, mo, party_by_obec, pd, top_parties):
    mo.md(
        """
        ## Finding Zero-Vote Cases with OBEC Context

        Identifying commissions where top parties got 0 votes, and calculating
        what we'd expect based on their performance in that specific municipality.
        """
    )

    # Get all unique commissions and top parties
    all_commissions = df_with_size['ID_OKRSKY'].unique()

    # Create all possible combinations
    all_combinations = pd.MultiIndex.from_product(
        [top_parties, all_commissions],
        names=['Party', 'Commission_ID']
    ).to_frame(index=False)

    # Get actual combinations
    actual_combinations = df_with_size[df_with_size['KSTRANA'].isin(top_parties)][
        ['KSTRANA', 'ID_OKRSKY']
    ].copy()
    actual_combinations.columns = ['Party', 'Commission_ID']
    actual_combinations['Present'] = True

    # Find missing combinations (zero votes)
    combined = all_combinations.merge(actual_combinations, on=['Party', 'Commission_ID'], how='left')
    zero_votes = combined[combined['Present'].isna()][['Party', 'Commission_ID']].copy()

    # Get commission info with OBEC
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

    # Merge to get full info
    zero_votes_df = zero_votes.merge(commission_info, on='Commission_ID')

    # Merge OBEC-specific party probabilities
    zero_votes_df = zero_votes_df.merge(
        party_by_obec[['Party', 'OBEC', 'Vote_Share_In_OBEC', 'Probability_In_OBEC']],
        on=['Party', 'OBEC'],
        how='left'
    )

    print(f"Found {len(zero_votes_df)} zero-vote cases for top parties")
    print(f"Cases with OBEC-specific data: {zero_votes_df['Probability_In_OBEC'].notna().sum()}")
    print(f"Cases WITHOUT OBEC data (party never appeared in that municipality): {zero_votes_df['Probability_In_OBEC'].isna().sum()}")

    zero_votes_df
    return (
        all_combinations,
        commission_info,
        zero_votes,
        zero_votes_df,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ## OBEC-Specific Probability Calculation

        **P(0 votes) = (1 - p_obec)^n**

        Where:
        - **p_obec** = party's vote probability in this exact municipality
        - **n** = total votes in the commission

        ### Key Insight:
        If a party NEVER appeared in a municipality (p_obec is NaN), that means they got 0 votes
        across ALL commissions there. This is not suspicious - it indicates genuine lack of support.

        We'll focus on cases where the party IS present in the municipality overall,
        but mysteriously got 0 in one specific commission.
        """
    )
    return


@app.cell
def __(party_summary, pd, zero_votes_df):
    # Calculate probabilities
    prob_analysis = zero_votes_df.copy()

    # Add overall party stats
    prob_analysis['Party_Probability_Overall'] = prob_analysis['Party'].map(
        lambda x: party_summary.loc[x, 'Probability']
    )
    prob_analysis['Party_Percentage'] = prob_analysis['Party'].map(
        lambda x: party_summary.loc[x, 'Percentage']
    )

    # Calculate overall probability (baseline)
    prob_analysis['Probability_of_Zero_Overall'] = (
        1 - prob_analysis['Party_Probability_Overall']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Calculate OBEC-specific probability (most accurate!)
    prob_analysis['Probability_of_Zero_OBEC'] = (
        1 - prob_analysis['Probability_In_OBEC']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Use OBEC-specific where available, otherwise use overall
    # Note: if OBEC probability is NaN, it means party never appeared in that municipality
    prob_analysis['Probability_of_Zero'] = prob_analysis['Probability_of_Zero_OBEC'].fillna(
        prob_analysis['Probability_of_Zero_Overall']
    )

    # Flag whether we used OBEC-specific or overall probability
    prob_analysis['Used_OBEC_Probability'] = prob_analysis['Probability_In_OBEC'].notna()

    # Convert to percentage
    prob_analysis['Probability_of_Zero_Percent'] = prob_analysis['Probability_of_Zero'] * 100

    # Flag suspicious cases (less than 1% probability) - only if we have OBEC data
    prob_analysis['Is_Suspicious'] = (
        (prob_analysis['Probability_of_Zero'] < 0.01) &
        prob_analysis['Used_OBEC_Probability']
    )

    # Flag HIGHLY suspicious cases (less than 0.1%)
    prob_analysis['Is_Highly_Suspicious'] = (
        (prob_analysis['Probability_of_Zero'] < 0.001) &
        prob_analysis['Used_OBEC_Probability']
    )

    print(f"Total zero-vote cases: {len(prob_analysis)}")
    print(f"Cases using OBEC-specific probability: {prob_analysis['Used_OBEC_Probability'].sum()}")
    print(f"Suspicious cases (< 1%): {prob_analysis['Is_Suspicious'].sum()}")
    print(f"Highly suspicious (< 0.1%): {prob_analysis['Is_Highly_Suspicious'].sum()}")

    prob_analysis
    return (prob_analysis,)


@app.cell
def __(mo, prob_analysis):
    mo.md(
        """
        ## Most Suspicious Cases with OBEC Context

        These are commissions where:
        1. The party IS active in the municipality (has votes in other commissions)
        2. But got 0 votes in this specific commission
        3. The probability of this happening by chance is < 1%
        """
    )

    # Show most suspicious cases with OBEC data
    most_suspicious = prob_analysis[
        prob_analysis['Used_OBEC_Probability']
    ].sort_values('Probability_of_Zero').head(20)[[
        'Party',
        'Party_Percentage',
        'OBEC',
        'Municipality_Size_Category',
        'Commission_ID',
        'Total_Votes_In_Commission',
        'Vote_Share_In_OBEC',
        'Probability_of_Zero_OBEC',
        'Probability_of_Zero_Overall',
        'Is_Suspicious',
        'Is_Highly_Suspicious'
    ]].copy()

    most_suspicious['Improvement_Factor'] = (
        most_suspicious['Probability_of_Zero_Overall'] /
        most_suspicious['Probability_of_Zero_OBEC']
    )

    most_suspicious
    return (most_suspicious,)


@app.cell
def __(mo, prob_analysis):
    mo.md("### Cases Where Party Never Appeared in Municipality")

    # These are NOT suspicious - party has no support there
    never_appeared = prob_analysis[
        ~prob_analysis['Used_OBEC_Probability']
    ].sort_values('Total_Votes_In_Commission', ascending=False).head(20)[[
        'Party',
        'Party_Percentage',
        'OBEC',
        'Municipality_Size_Category',
        'Commission_ID',
        'Total_Votes_In_Commission',
        'Probability_of_Zero_Overall'
    ]]

    mo.md(f"""
    Found **{len(prob_analysis[~prob_analysis['Used_OBEC_Probability']])}** cases where the party
    got 0 votes in the commission AND never appeared anywhere in that municipality.

    These are NOT suspicious - they indicate genuine lack of local support.
    """)

    never_appeared
    return (never_appeared,)


@app.cell
def __(mo):
    mo.md(
        """
        # Visualizations: OBEC-Specific Analysis

        Enhanced charts showing the accuracy improvement from OBEC-specific probabilities.
        """
    )
    return


@app.cell
def __(go, mo, np, prob_analysis, px):
    mo.md("### Geographic Distribution: Suspicious Zeros by Municipality")

    # Filter to cases with OBEC-specific data
    geo_data = prob_analysis[prob_analysis['Used_OBEC_Probability']].copy()
    geo_data['Party'] = geo_data['Party'].astype(str)

    # Create suspiciousness categories
    geo_data['Suspiciousness'] = pd.cut(
        geo_data['Probability_of_Zero'],
        bins=[0, 0.001, 0.01, 0.1, 1.0],
        labels=['Highly Suspicious (<0.1%)', 'Suspicious (0.1-1%)', 'Unlikely (1-10%)', 'Possible (>10%)']
    )

    # Create inverted size metric: more suspicious (lower probability) = larger bubble
    geo_data['Bubble_Size'] = -np.log10(geo_data['Probability_of_Zero'] + 1e-100)

    geo_scatter = px.scatter(
        geo_data,
        x='Municipality_Total_Votes',
        y='Total_Votes_In_Commission',
        color='Suspiciousness',
        size='Bubble_Size',
        hover_data=['Party', 'OBEC', 'Municipality_Size_Category', 'Vote_Share_In_OBEC', 'Probability_of_Zero_Percent'],
        title='Zero-Vote Commissions: OBEC-Specific Analysis (Municipality Size vs Commission Size)',
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
        size_max=30,
        log_x=True
    )

    geo_scatter.update_traces(marker=dict(sizemin=5))
    geo_scatter.update_layout(height=600)

    mo.ui.plotly(geo_scatter)
    return geo_data, geo_scatter


@app.cell
def __(go, make_subplots, mo, prob_analysis):
    mo.md("### Comparison: Overall vs OBEC-Specific Probabilities")

    # Compare the two probability calculations
    comparison_data = prob_analysis[
        prob_analysis['Used_OBEC_Probability']
    ].copy()

    comparison_data['Party_Label'] = comparison_data.apply(
        lambda row: f"{row['Party']} ({row['Party_Percentage']:.1f}%)", axis=1
    )

    # Calculate how much more accurate OBEC-specific is
    comparison_data['Probability_Ratio'] = (
        comparison_data['Probability_of_Zero_OBEC'] /
        comparison_data['Probability_of_Zero_Overall']
    )

    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Overall National Probability', 'OBEC-Specific Probability'),
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
                size=8,
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

    # Chart 2: OBEC-Specific
    fig.add_trace(
        go.Scatter(
            x=comparison_data['Total_Votes_In_Commission'],
            y=comparison_data['Probability_of_Zero_OBEC'],
            mode='markers',
            name='OBEC-Specific',
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

    fig.update_xaxes(title_text='Commission Size', row=1, col=1)
    fig.update_xaxes(title_text='Commission Size', row=1, col=2)
    fig.update_yaxes(title_text='Probability of 0 Votes', type='log', row=1, col=1)
    fig.update_yaxes(title_text='Probability of 0 Votes', type='log', row=1, col=2)

    fig.update_layout(
        title_text='Comparing Overall vs OBEC-Specific Probability Calculations',
        height=500,
        showlegend=False
    )

    mo.ui.plotly(fig)
    return comparison_data, fig


@app.cell
def __(mo, prob_analysis, px):
    mo.md("### Improvement Factor: How Much More Accurate is OBEC-Specific?")

    # Calculate improvement factor
    improvement_data = prob_analysis[
        prob_analysis['Used_OBEC_Probability']
    ].copy()

    improvement_data['Improvement_Factor'] = (
        improvement_data['Probability_of_Zero_Overall'] /
        improvement_data['Probability_of_Zero_OBEC']
    )

    # Filter out extreme outliers for visualization
    improvement_data_filtered = improvement_data[
        improvement_data['Improvement_Factor'] < 1000
    ]

    improvement_fig = px.histogram(
        improvement_data_filtered,
        x='Improvement_Factor',
        nbins=50,
        title='Distribution of Accuracy Improvement: Overall → OBEC-Specific',
        labels={
            'Improvement_Factor': 'Improvement Factor (Overall P / OBEC P)',
            'count': 'Number of Cases'
        },
        log_y=True
    )

    improvement_fig.add_vline(
        x=1,
        line_dash="dash",
        line_color="red",
        annotation_text="No improvement (ratio = 1)"
    )

    improvement_fig.update_layout(height=500)

    mo.ui.plotly(improvement_fig)
    return improvement_data, improvement_data_filtered, improvement_fig


@app.cell
def __(improvement_data, mo):
    mo.md(
        f"""
        ### Key Statistics on Improvement

        - **Median improvement factor**: {improvement_data['Improvement_Factor'].median():.2f}x
        - **Mean improvement factor**: {improvement_data['Improvement_Factor'].mean():.2f}x
        - **Cases where OBEC-specific is LESS suspicious** (ratio > 1): {(improvement_data['Improvement_Factor'] > 1).sum()}
        - **Cases where OBEC-specific is MORE suspicious** (ratio < 1): {(improvement_data['Improvement_Factor'] < 1).sum()}

        **Interpretation**:
        - Ratio > 1: Using overall probability overestimates suspiciousness (party is weaker in this municipality)
        - Ratio < 1: Using overall probability underestimates suspiciousness (party is stronger in this municipality)
        - Most cases have ratio > 1, meaning OBEC-specific probabilities correctly identify many "false alarms"
        """
    )
    return


@app.cell
def __(mo, prob_analysis, px):
    mo.md("### Party Vote Share in OBEC: Distribution")

    # Show distribution of party support levels where they got 0 in a commission
    obec_support = prob_analysis[prob_analysis['Used_OBEC_Probability']].copy()

    support_fig = px.histogram(
        obec_support,
        x='Vote_Share_In_OBEC',
        nbins=50,
        title='Distribution of Party Support in Municipality (for zero-vote commissions)',
        labels={
            'Vote_Share_In_OBEC': 'Party Vote Share in Municipality (%)',
            'count': 'Number of Zero-Vote Cases'
        },
        color='Is_Suspicious',
        barmode='overlay',
        opacity=0.7
    )

    support_fig.update_layout(height=500)

    mo.ui.plotly(support_fig)
    return obec_support, support_fig


@app.cell
def __(mo):
    mo.md(
        """
        ## Key Insights: OBEC-Specific Analysis

        ### What We Learned:

        1. **Local Context Matters Critically**
           - Many "suspicious" cases using overall probability are actually normal when considering local support
           - A party getting 0 votes where they typically get <1% is not suspicious at all

        2. **False Alarm Reduction**
           - OBEC-specific probabilities dramatically reduce false positives
           - Focuses attention on truly anomalous cases

        3. **Real Suspicious Cases Stand Out**
           - When a party gets 0 votes in a commission but normally gets 10-15% in that municipality: HIGHLY suspicious
           - These are the cases that warrant investigation

        4. **Party Never Present = Not Suspicious**
           - If a party got 0 votes across ALL commissions in a municipality, that's genuine lack of support
           - Only suspicious when they're strong elsewhere in the same municipality

        ### Recommendations:

        - **Use OBEC-specific probabilities** for the most accurate suspiciousness assessment
        - **Ignore cases** where the party never appeared in the municipality
        - **Investigate cases** where P(0) < 0.1% AND the party typically gets >5% in that municipality
        - **Look for patterns**: Multiple suspicious commissions in the same municipality is especially concerning
        """
    )
    return


if __name__ == "__main__":
    app.run()
