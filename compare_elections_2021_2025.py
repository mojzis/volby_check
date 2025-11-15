import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import election_data_loader as edl
    return edl, go, mo, pd, px


@app.cell
def __(mo):
    mo.md(
        """
        # Comparative Election Analysis: 2021 vs 2025

        This notebook compares suspicious zero-vote patterns between the 2021 and 2025
        parliamentary elections using the same probability-based methodology.

        ## Methodology

        For both elections, we:
        1. Load election data, municipality info, and party data
        2. Calculate party statistics and identify top parties
        3. Find zero-vote cases for top parties
        4. Calculate OBEC-specific probabilities
        5. Compare suspicious case rates between years
        """
    )
    return


@app.cell
def __(edl, mo):
    mo.md("## Load 2025 Election Data")

    # Load 2025 data
    df_2025 = edl.load_election_data(year=2025)
    municipalities_2025 = edl.load_municipality_data(year=2025)
    parties_2025 = edl.load_party_data(year=2025)

    print(f"2025: {len(df_2025):,} records")
    return df_2025, municipalities_2025, parties_2025


@app.cell
def __(edl, mo):
    mo.md("## Load 2021 Election Data")

    # Load 2021 data
    df_2021 = edl.load_election_data(year=2021)
    municipalities_2021 = edl.load_municipality_data(year=2021)
    parties_2021 = edl.load_party_data(year=2021)

    print(f"2021: {len(df_2021):,} records")
    return df_2021, municipalities_2021, parties_2021


@app.cell
def __(df_2021, df_2025, edl, mo):
    mo.md("## Analyze 2025 Election")

    # Calculate party statistics for 2025
    party_summary_2025, _, _, _, total_votes_2025 = edl.calculate_party_statistics(df_2025)
    top_parties_2025 = edl.get_top_parties(party_summary_2025, n=7)

    # Calculate party performance by OBEC
    party_by_obec_2025 = edl.calculate_party_performance_by_obec(df_2025)

    # Find zero-vote cases
    zero_votes_2025 = edl.calculate_zero_vote_cases(df_2025, top_parties_2025)

    # Calculate probabilities
    prob_2025 = edl.merge_obec_probabilities(zero_votes_2025, party_by_obec_2025, party_summary_2025)
    prob_2025 = edl.add_suspiciousness_flags(prob_2025, require_obec_data=True)

    print(f"2025 Summary:")
    print(f"  Total votes: {total_votes_2025:,}")
    print(f"  Zero-vote cases: {len(zero_votes_2025):,}")
    print(f"  Suspicious cases (P < 1%): {prob_2025['Is_Suspicious'].sum()}")
    print(f"  Highly suspicious (P < 0.1%): {prob_2025['Is_Highly_Suspicious'].sum()}")

    party_summary_2025.head(7)
    return (
        party_by_obec_2025,
        party_summary_2025,
        prob_2025,
        top_parties_2025,
        total_votes_2025,
        zero_votes_2025,
    )


@app.cell
def __(df_2021, edl, mo):
    mo.md("## Analyze 2021 Election")

    # Calculate party statistics for 2021
    party_summary_2021, _, _, _, total_votes_2021 = edl.calculate_party_statistics(df_2021)
    top_parties_2021 = edl.get_top_parties(party_summary_2021, n=7)

    # Calculate party performance by OBEC
    party_by_obec_2021 = edl.calculate_party_performance_by_obec(df_2021)

    # Find zero-vote cases
    zero_votes_2021 = edl.calculate_zero_vote_cases(df_2021, top_parties_2021)

    # Calculate probabilities
    prob_2021 = edl.merge_obec_probabilities(zero_votes_2021, party_by_obec_2021, party_summary_2021)
    prob_2021 = edl.add_suspiciousness_flags(prob_2021, require_obec_data=True)

    print(f"2021 Summary:")
    print(f"  Total votes: {total_votes_2021:,}")
    print(f"  Zero-vote cases: {len(zero_votes_2021):,}")
    print(f"  Suspicious cases (P < 1%): {prob_2021['Is_Suspicious'].sum()}")
    print(f"  Highly suspicious (P < 0.1%): {prob_2021['Is_Highly_Suspicious'].sum()}")

    party_summary_2021.head(7)
    return (
        party_by_obec_2021,
        party_summary_2021,
        prob_2021,
        top_parties_2021,
        total_votes_2021,
        zero_votes_2021,
    )


@app.cell
def __(mo, pd, prob_2021, prob_2025, zero_votes_2021, zero_votes_2025):
    mo.md("## Comparison Summary")

    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Metric': [
            'Total Zero-Vote Cases',
            'Cases with OBEC Data',
            'Suspicious Cases (P < 1%)',
            'Highly Suspicious (P < 0.1%)',
            'Suspicious Rate (%)',
            'Highly Suspicious Rate (%)'
        ],
        '2021': [
            len(zero_votes_2021),
            prob_2021['Used_OBEC_Probability'].sum(),
            prob_2021['Is_Suspicious'].sum(),
            prob_2021['Is_Highly_Suspicious'].sum(),
            round(prob_2021['Is_Suspicious'].sum() / prob_2021['Used_OBEC_Probability'].sum() * 100, 2),
            round(prob_2021['Is_Highly_Suspicious'].sum() / prob_2021['Used_OBEC_Probability'].sum() * 100, 2)
        ],
        '2025': [
            len(zero_votes_2025),
            prob_2025['Used_OBEC_Probability'].sum(),
            prob_2025['Is_Suspicious'].sum(),
            prob_2025['Is_Highly_Suspicious'].sum(),
            round(prob_2025['Is_Suspicious'].sum() / prob_2025['Used_OBEC_Probability'].sum() * 100, 2),
            round(prob_2025['Is_Highly_Suspicious'].sum() / prob_2025['Used_OBEC_Probability'].sum() * 100, 2)
        ]
    })

    # Calculate change
    comparison['Change'] = comparison['2025'] - comparison['2021']
    comparison['Change %'] = ((comparison['2025'] / comparison['2021'] - 1) * 100).round(1)

    comparison
    return (comparison,)


@app.cell
def __(comparison, mo, px):
    mo.md("### Suspicious Case Rates Comparison")

    # Create bar chart comparing suspicious rates
    rates_data = comparison[comparison['Metric'].str.contains('Rate')]

    fig = px.bar(
        rates_data.melt(id_vars='Metric', value_vars=['2021', '2025']),
        x='Metric',
        y='value',
        color='variable',
        barmode='group',
        title='Suspicious Case Rates: 2021 vs 2025',
        labels={'value': 'Rate (%)', 'variable': 'Year', 'Metric': ''},
        text='value'
    )

    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=500, xaxis_type='category')

    mo.ui.plotly(fig)
    return fig, rates_data


@app.cell
def __(mo, pd, prob_2021, prob_2025):
    mo.md("### Most Suspicious Cases from Each Year")

    # Get top 10 most suspicious from each year
    top_10_2021 = prob_2021[prob_2021['Used_OBEC_Probability']].nsmallest(10, 'Probability_of_Zero')
    top_10_2025 = prob_2025[prob_2025['Used_OBEC_Probability']].nsmallest(10, 'Probability_of_Zero')

    # Create comparison table
    comparison_top = pd.DataFrame({
        '2021 - Party': top_10_2021['Party'].values,
        '2021 - OBEC': top_10_2021['OBEC'].values,
        '2021 - P(0)': top_10_2021['Probability_of_Zero'].apply(lambda x: f"{x:.2e}").values,
        '2025 - Party': top_10_2025['Party'].values,
        '2025 - OBEC': top_10_2025['OBEC'].values,
        '2025 - P(0)': top_10_2025['Probability_of_Zero'].apply(lambda x: f"{x:.2e}").values,
    })

    comparison_top
    return comparison_top, top_10_2021, top_10_2025


@app.cell
def __(go, mo, prob_2021, prob_2025):
    mo.md("### Probability Distribution Comparison")

    # Create histogram comparing probability distributions
    fig_dist = go.Figure()

    # Add 2021 data
    fig_dist.add_trace(go.Histogram(
        x=-prob_2021[prob_2021['Used_OBEC_Probability']]['Probability_of_Zero'].apply(lambda x: -1 if x == 0 else pd.np.log10(x)),
        name='2021',
        opacity=0.7,
        nbinsx=50
    ))

    # Add 2025 data
    fig_dist.add_trace(go.Histogram(
        x=-prob_2025[prob_2025['Used_OBEC_Probability']]['Probability_of_Zero'].apply(lambda x: -1 if x == 0 else pd.np.log10(x)),
        name='2025',
        opacity=0.7,
        nbinsx=50
    ))

    fig_dist.update_layout(
        title='Distribution of Zero-Vote Probabilities: 2021 vs 2025',
        xaxis_title='-log₁₀(Probability) (Higher = More Suspicious)',
        yaxis_title='Number of Cases',
        barmode='overlay',
        height=500
    )

    # Add threshold lines
    fig_dist.add_vline(x=2, line_dash="dash", line_color="orange", annotation_text="1% threshold")
    fig_dist.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="0.1% threshold")

    mo.ui.plotly(fig_dist)
    return (fig_dist,)


@app.cell
def __(mo):
    mo.md(
        """
        ## Key Findings

        This comparative analysis allows us to:

        1. **Identify Trends**: Are suspicious zero-vote cases increasing or decreasing?
        2. **Methodology Validation**: Does the same analysis method produce consistent results?
        3. **Pattern Recognition**: Are the same municipalities or parties showing up in both elections?
        4. **Data Quality**: Comparison helps identify if issues are systemic or year-specific

        ### Interpretation Guidelines

        - **Higher suspicious rates** may indicate data quality issues or potential irregularities
        - **Similar patterns** across years suggest systematic factors
        - **Different patterns** may reflect genuine political changes or improved data collection
        - **Consistency check**: Top parties should generally have similar zero-vote probability distributions

        ### Next Steps

        - Examine specific municipalities that appear suspicious in both years
        - Compare party-specific patterns across elections
        - Analyze changes in commission sizes and vote distributions
        - Cross-reference with any reported issues from election officials
        """
    )
    return


if __name__ == "__main__":
    app.run()
