import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy import stats
    import numpy as np
    from plotly.subplots import make_subplots
    import election_data_loader as edl

    # Configure pandas display for HTML export (disable pager, limit rows)
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.show_dimensions', True)

    return (
        edl,
        go,
        make_subplots,
        mo,
        np,
        pd,
        px,
        stats,
    )


@app.cell
def __(mo):
    mo.md("# OBEC-Specific Zero-Vote Analysis")
    return


@app.cell
def __(edl):
    # Load election data using shared function
    df = edl.load_election_data()

    # Load municipality and party data for names
    municipalities = edl.load_municipality_data()
    parties_data = edl.load_party_data()

    print(f"Columns: {df.columns.tolist()}")
    df.head(30)
    return (df, municipalities, parties_data)


@app.cell
def __(df, edl, mo):
    mo.md("## Municipality (OBEC) Characterization")

    # Calculate municipality sizes using shared function
    municipality_sizes = edl.calculate_municipality_sizes(df, n_categories=5)

    print("\nVotes by category:")
    print(municipality_sizes.groupby('Size_Category', observed=False)['Total_Votes'].agg(['min', 'max', 'mean']))

    municipality_sizes
    return (municipality_sizes,)


@app.cell
def __(df, edl, municipality_sizes):
    # Merge municipality size info back to main dataframe using shared function
    df_with_size = edl.merge_municipality_sizes(df, municipality_sizes)

    df_with_size
    return (df_with_size,)


@app.cell
def __(df_with_size, edl):
    # Calculate party performance by OBEC using shared function
    party_by_obec = edl.calculate_party_performance_by_obec(df_with_size)

    party_by_obec.head(20)
    return (party_by_obec,)


@app.cell
def __(df_with_size, edl):
    # Calculate party statistics using shared function
    party_summary, party_totals, party_percentages, party_probabilities, total_votes = edl.calculate_party_statistics(df_with_size)

    party_summary.head(10)
    return (
        party_percentages,
        party_probabilities,
        party_summary,
        party_totals,
        total_votes,
    )


@app.cell
def __(edl, party_summary):
    # Get top 7 parties using shared function
    top_parties = edl.get_top_parties(party_summary, n=7)
    top_parties
    return (top_parties,)


@app.cell
def __(df_with_size, edl, mo, top_parties):
    mo.md("## Finding Zero-Vote Cases with OBEC Context")

    # Get commission info with OBEC and size category using shared function
    commission_info = edl.create_commission_info(df_with_size, include_size_category=True)

    # Find zero-vote cases using shared function
    zero_votes_df = edl.calculate_zero_vote_cases(df_with_size, top_parties, commission_info)

    print(f"Found {len(zero_votes_df)} zero-vote cases for top parties")

    zero_votes_df
    return (
        commission_info,
        zero_votes_df,
    )


@app.cell
def __(mo):
    mo.md("## OBEC-Specific Probability Calculation")
    return


@app.cell
def __(edl, party_by_obec, party_summary, zero_votes_df):
    # Merge OBEC probabilities and calculate using shared function
    prob_analysis = edl.merge_obec_probabilities(zero_votes_df, party_by_obec, party_summary)

    # Add suspiciousness flags using shared function
    prob_analysis = edl.add_suspiciousness_flags(prob_analysis, suspicious_threshold=0.01, highly_suspicious_threshold=0.001, require_obec_data=True)

    prob_analysis
    return (prob_analysis,)


@app.cell
def __(mo, municipalities, parties_data, pd, prob_analysis):
    mo.md("## Most Suspicious Cases (OBEC-Adjusted)")

    # Show most suspicious cases with OBEC data
    most_suspicious = prob_analysis[
        prob_analysis['Used_OBEC_Probability']
    ].sort_values('Probability_of_Zero').head(30).copy()

    # Add party names (ensure type consistency)
    parties_lookup = parties_data[['KSTRANA', 'ZKRATKAK8']].copy()
    parties_lookup.columns = ['Party', 'Party_Name']
    parties_lookup['Party'] = parties_lookup['Party'].astype(str)
    most_suspicious['Party'] = most_suspicious['Party'].astype(str)
    most_suspicious = most_suspicious.merge(parties_lookup, on='Party', how='left')
    most_suspicious['Party_Name'] = most_suspicious['Party_Name'].fillna(most_suspicious['Party'])

    # Add municipality names
    municipalities_lookup = municipalities[['OBEC', 'NAZEVOBCE']].copy()
    municipalities_lookup['OBEC'] = municipalities_lookup['OBEC'].astype(str)
    most_suspicious['OBEC'] = most_suspicious['OBEC'].astype(str)
    most_suspicious = most_suspicious.merge(municipalities_lookup, on='OBEC', how='left')
    most_suspicious['NAZEVOBCE'] = most_suspicious['NAZEVOBCE'].fillna('Unknown (' + most_suspicious['OBEC'].astype(str) + ')')

    # Select and reorder columns with names
    most_suspicious_display = most_suspicious[[
        'Party_Name',
        'Party',
        'Party_Percentage',
        'NAZEVOBCE',
        'OBEC',
        'Commission_ID',
        'Total_Votes_In_Commission',
        'Vote_Share_In_OBEC',
        'Probability_of_Zero_OBEC',
        'Is_Suspicious',
        'Is_Highly_Suspicious'
    ]].copy()

    most_suspicious_display.columns = [
        'Party_Name',
        'Party_Code',
        'National_%',
        'Municipality',
        'OBEC_Code',
        'Commission_ID',
        'Commission_Size',
        'Party_Share_in_Municipality_%',
        'Probability_of_Zero',
        'Suspicious',
        'Highly_Suspicious'
    ]

    print(f"Total suspicious cases (P < 1%): {prob_analysis['Is_Suspicious'].sum()}")
    print(f"Highly suspicious (P < 0.1%): {prob_analysis['Is_Highly_Suspicious'].sum()}")

    most_suspicious_display
    return (most_suspicious, most_suspicious_display)


@app.cell
def __(mo, prob_analysis):
    mo.md("### Cases Where Party Never Appeared in Municipality (Not Suspicious)")

    # These are NOT suspicious - party has no support there
    never_appeared_count = len(prob_analysis[~prob_analysis['Used_OBEC_Probability']])

    print(f"Cases where party got 0 votes across ALL commissions in municipality: {never_appeared_count}")
    print("(These are NOT suspicious - party has no local support)")
    return ()


@app.cell
def __(mo):
    mo.md("# Visualizations")
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
    print(f"Median improvement factor: {improvement_data['Improvement_Factor'].median():.2f}x")
    print(f"Mean improvement factor: {improvement_data['Improvement_Factor'].mean():.2f}x")
    return ()


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


if __name__ == "__main__":
    app.run()
