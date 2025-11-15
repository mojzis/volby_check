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
    import election_data_loader as edl
    return edl, go, mo, np, pd, px, stats


@app.cell
def __(mo):
    mo.md(
        """
        # Suspicious Zero-Vote Analysis

        ## Statistical Analysis of Unlikely Zero-Vote Cases

        This analysis calculates the probability that top parties would receive 0 votes
        in their largest zero-vote commissions. We use binomial probability to identify
        statistically suspicious cases where a party's zero-vote result is highly improbable.
        """
    )
    return


@app.cell
def __(edl):
    # Load election data using shared function
    df = edl.load_election_data()
    df
    return (df,)


@app.cell
def __(df, edl):
    # Calculate party statistics using shared function
    party_summary, party_totals, party_percentages, party_probabilities, total_votes = edl.calculate_party_statistics(df)

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
def __(df, edl, top_parties):
    # Find zero-vote cases using shared function
    zero_votes_df = edl.calculate_zero_vote_cases(df, top_parties)

    zero_votes_df
    return (zero_votes_df,)


@app.cell
def __(mo):
    mo.md(
        """
        ## Probability Calculation Methodology

        For each commission where a party got 0 votes, we calculate:

        **P(0 votes) = (1 - p)^n**

        Where:
        - p = party's overall vote probability (their national vote share)
        - n = total votes in the commission

        This uses the binomial distribution to calculate the probability that,
        given a party's success rate, they would receive exactly 0 votes in
        a commission of size n purely by chance.

        **Suspicious threshold**: Cases where P(0 votes) < 0.01 (less than 1% probability)
        are flagged as potentially suspicious.
        """
    )
    return


@app.cell
def __(party_summary, pd, zero_votes_df):
    # For each zero-vote case, calculate the probability
    prob_analysis = zero_votes_df.copy()

    # Add party probability
    prob_analysis['Party_Probability'] = prob_analysis['Party'].map(
        lambda x: party_summary.loc[x, 'Probability']
    )

    # Add party percentage for display
    prob_analysis['Party_Percentage'] = prob_analysis['Party'].map(
        lambda x: party_summary.loc[x, 'Percentage']
    )

    # Calculate probability of getting 0 votes: P(X=0) = (1-p)^n
    prob_analysis['Probability_of_Zero'] = (
        1 - prob_analysis['Party_Probability']
    ) ** prob_analysis['Total_Votes_In_Commission']

    # Convert to percentage and scientific notation for readability
    prob_analysis['Probability_of_Zero_Percent'] = prob_analysis['Probability_of_Zero'] * 100

    # Flag suspicious cases (less than 1% probability)
    prob_analysis['Is_Suspicious'] = prob_analysis['Probability_of_Zero'] < 0.01

    prob_analysis
    return (prob_analysis,)


@app.cell
def __(mo, prob_analysis):
    mo.md(
        """
        ## Top 3 Biggest Zero-Vote Commissions Per Party

        Focusing on the largest commissions where each party received 0 votes,
        as these are the most statistically interesting cases.
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
        'Total_Votes_In_Commission',
        'Probability_of_Zero',
        'Probability_of_Zero_Percent',
        'Is_Suspicious'
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
        ## Summary of Suspicious Cases

        Cases where the probability of a party getting 0 votes is less than 1%
        """
    )

    suspicious_cases = top3_per_party[top3_per_party['Is_Suspicious']].copy()

    suspicious_summary = suspicious_cases[[
        'Party',
        'Party_Percentage',
        'Commission_ID',
        'Total_Votes_In_Commission',
        'Probability_of_Zero',
        'Probability_of_Zero_Percent'
    ]].sort_values('Probability_of_Zero')

    print(f"Found {len(suspicious_cases)} suspicious cases out of {len(top3_per_party)} top-3 cases")
    print(f"Suspicious rate: {len(suspicious_cases)/len(top3_per_party)*100:.1f}%")

    suspicious_summary
    return suspicious_cases, suspicious_summary


@app.cell
def __(mo):
    mo.md(
        """
        # Interactive Visualizations

        Charts showing the probability analysis and suspicious cases
        """
    )
    return


@app.cell
def __(go, mo, np, party_summary, top3_per_party):
    mo.md("### Probability Heatmap: Top 3 Zero-Vote Commissions Per Party")

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

    # Create pivot table for heatmap
    # Use log scale for probabilities (they can be very small)
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
                    f"Commission: {row['Commission_ID']}<br>"
                    f"Commission Size: {row['Total_Votes_In_Commission']} votes<br>"
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
        title='Probability of Zero Votes: -log₁₀(P) Scale<br>(Higher = More Suspicious)',
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
def __(go, mo, top3_per_party):
    mo.md("### Bubble Chart: Commission Size vs Probability of Zero")

    # Create bubble chart
    bubble_data = top3_per_party.copy()
    bubble_data['Party_Label'] = bubble_data.apply(
        lambda row: f"{row['Party']} ({row['Party_Percentage']:.1f}%)", axis=1
    )

    # Add rank for sizing
    bubble_data['Rank'] = bubble_data.groupby('Party').cumcount() + 1

    # Create color based on suspiciousness
    bubble_data['Color'] = bubble_data['Is_Suspicious'].map({
        True: 'Suspicious (P < 1%)',
        False: 'Not Suspicious'
    })

    bubble_fig = go.Figure()

    # Add suspicious points
    suspicious = bubble_data[bubble_data['Is_Suspicious']]
    bubble_fig.add_trace(go.Scatter(
        x=suspicious['Total_Votes_In_Commission'],
        y=suspicious['Probability_of_Zero_Percent'],
        mode='markers+text',
        name='Suspicious (P < 1%)',
        text=suspicious['Party'].astype(str),
        textposition='top center',
        marker=dict(
            size=15,
            color='red',
            line=dict(width=2, color='darkred'),
            symbol='circle'
        ),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Commission Size: %{x} votes<br>' +
            'P(0 votes): %{y:.4f}%<br>' +
            '<extra></extra>'
        )
    ))

    # Add non-suspicious points
    not_suspicious = bubble_data[~bubble_data['Is_Suspicious']]
    bubble_fig.add_trace(go.Scatter(
        x=not_suspicious['Total_Votes_In_Commission'],
        y=not_suspicious['Probability_of_Zero_Percent'],
        mode='markers+text',
        name='Not Suspicious',
        text=not_suspicious['Party'].astype(str),
        textposition='top center',
        marker=dict(
            size=10,
            color='lightblue',
            line=dict(width=1, color='blue'),
            symbol='circle'
        ),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Commission Size: %{x} votes<br>' +
            'P(0 votes): %{y:.4f}%<br>' +
            '<extra></extra>'
        )
    ))

    # Add horizontal line at 1% threshold
    bubble_fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="orange",
        annotation_text="1% Threshold",
        annotation_position="right"
    )

    bubble_fig.update_layout(
        title='Probability Analysis: Commission Size vs P(0 Votes)',
        xaxis_title='Commission Size (Total Votes)',
        yaxis_title='Probability of Getting 0 Votes (%)',
        yaxis_type='log',
        height=600,
        showlegend=True,
        hovermode='closest'
    )

    mo.ui.plotly(bubble_fig)
    return bubble_data, bubble_fig, not_suspicious, suspicious


@app.cell
def __(mo, party_summary, px, top3_per_party):
    mo.md("### Bar Chart: Expected vs Actual Votes in Top Zero-Vote Commissions")

    # Calculate expected votes for each case
    bar_data = top3_per_party.copy()
    bar_data['Expected_Votes'] = (
        bar_data['Party_Probability'] * bar_data['Total_Votes_In_Commission']
    )
    bar_data['Actual_Votes'] = 0  # By definition in this analysis

    bar_data['Party_Label'] = bar_data.apply(
        lambda row: f"{row['Party']} ({row['Party_Percentage']:.1f}%)", axis=1
    )

    bar_data['Commission_Label'] = (
        bar_data['Party_Label'] + ' - ' +
        bar_data['Commission_ID'].astype(str) +
        ' (' + bar_data['Total_Votes_In_Commission'].astype(str) + ' votes)'
    )

    # Sort by party percentage then commission size
    bar_data = bar_data.sort_values(['Party_Percentage', 'Total_Votes_In_Commission'],
                                     ascending=[False, False])

    # Create grouped bar chart
    bar_fig = px.bar(
        bar_data,
        x='Commission_Label',
        y='Expected_Votes',
        title='Expected Votes vs Actual (0) in Top 3 Zero-Vote Commissions Per Party',
        labels={'Commission_Label': 'Party - Commission (Size)',
                'Expected_Votes': 'Expected Number of Votes'},
        color='Is_Suspicious',
        color_discrete_map={True: 'red', False: 'lightblue'},
        text='Expected_Votes'
    )

    bar_fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    bar_fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        showlegend=True,
        legend_title='Suspicious',
        xaxis_type='category'
    )

    mo.ui.plotly(bar_fig)
    return bar_data, bar_fig


@app.cell
def __(go, mo, top3_per_party):
    mo.md("### Waterfall Chart: Suspiciousness Score by Party")

    # Calculate a suspiciousness score for each party
    # Score = sum of -log10(probability) for their top 3 cases
    suspicion_scores = (
        top3_per_party
        .groupby('Party')
        .apply(lambda g: pd.Series({
            'Suspicion_Score': -np.log10(g['Probability_of_Zero'] + 1e-100).sum(),
            'Party_Percentage': g['Party_Percentage'].iloc[0],
            'Num_Suspicious': g['Is_Suspicious'].sum(),
            'Max_Commission_Size': g['Total_Votes_In_Commission'].max()
        }), include_groups=False)
        .reset_index()
    )

    suspicion_scores['Party_Label'] = suspicion_scores.apply(
        lambda row: f"{row['Party']} ({row['Party_Percentage']:.1f}%)", axis=1
    )

    # Sort by suspicion score
    suspicion_scores = suspicion_scores.sort_values('Suspicion_Score', ascending=False)

    # Create bar chart
    suspicion_fig = go.Figure(go.Bar(
        x=suspicion_scores['Party_Label'],
        y=suspicion_scores['Suspicion_Score'],
        marker_color=suspicion_scores['Num_Suspicious'],
        marker_colorscale='Reds',
        marker_showscale=True,
        marker_colorbar=dict(title='# Suspicious<br>Cases'),
        text=suspicion_scores['Suspicion_Score'].round(1),
        textposition='outside',
        hovertemplate=(
            '<b>%{x}</b><br>' +
            'Suspicion Score: %{y:.1f}<br>' +
            'Suspicious Cases: %{marker.color}<br>' +
            '<extra></extra>'
        )
    ))

    suspicion_fig.update_layout(
        title='Overall Suspiciousness Score by Party<br>(Sum of -log₁₀(P) for Top 3 Cases)',
        xaxis_title='Party (Vote %)',
        yaxis_title='Suspiciousness Score (Higher = More Suspicious)',
        height=500,
        xaxis_type='category'
    )

    mo.ui.plotly(suspicion_fig)
    return np, suspicion_fig, suspicion_scores


@app.cell
def __(mo):
    mo.md(
        """
        ## Key Insights

        ### What the Analysis Shows:

        1. **Statistical Improbability**: Cases with P < 1% are statistically suspicious -
           there's less than a 1% chance the party would naturally receive 0 votes in
           a commission of that size given their overall performance.

        2. **Party Support Matters**: Larger parties (higher vote %) are more suspicious
           when they get 0 votes in large commissions, as their expected votes are higher.

        3. **Commission Size**: Larger commissions make zero votes more suspicious -
           it's one thing to get 0 in a 50-vote commission, quite another in a 500-vote one.

        ### Interpretation:

        - **Red flags** (P < 0.1%): Highly suspicious, warrants investigation
        - **Yellow flags** (0.1% < P < 1%): Moderately suspicious, worth reviewing
        - **Green** (P > 1%): Could reasonably happen by chance
        """
    )
    return


if __name__ == "__main__":
    app.run()
