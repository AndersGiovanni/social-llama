import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


llama_df = pd.read_csv("results_llamas-total.csv")
# gemma_df = pd.read_csv("results_gemma.csv")


for col in llama_df.columns[
    2:
]:  # Starting from the third column, as the first two are 'task' and 'zero-shot'
    llama_df[f"diff_{col}"] = llama_df[col] - llama_df["zero-shot"]

# Dropping original columns to keep only the absolute differences
llama_df_abs_diff = llama_df.drop(llama_df.columns[1:8], axis=1)

# Define your groups and their tasks
groups = {
    "Humor and Sarcasm": ["hahackathon#is_humor", "sarc", "tweet_irony"],
    "Offensiveness": [
        "contextual-abuse#IdentityDirectedAbuse",
        "contextual-abuse#PersonDirectedAbuse",
        "hasbiasedimplication",
        "hateoffensive",
        "implicit-hate#explicit_hate",
        "implicit-hate#implicit_hate",
        "implicit-hate#stereotypical_hate",
        "intentyn",
        "tweet_offensive",
    ],
    "Sentiment and Emotion": [
        "crowdflower",
        "dailydialog",
        "empathy#distress_bin",
        "tweet_emotion",
    ],
    "Social Factors": [
        "complaints",
        "hayati_politeness",
        "questionintimacy",
        "stanfordpoliteness",
    ],
    "Trustworthiness": ["hypo-l", "rumor#rumor_bool", "two-to-lie#receiver_truth"],
}

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

llama_df_2 = llama_df.loc[:, ~llama_df.columns.str.startswith("diff")]

llama_df_2

# plt.show()

# Assuming llama_df_2 and groups are already defined as per previous examples


# Function to calculate group averages based on selected methods
def calculate_group_averages(selected_methods):
    group_averages = {}
    for group, tasks in groups.items():
        group_df = llama_df_2[llama_df_2["task"].isin(tasks)]
        group_averages[group] = group_df[selected_methods].mean()
    group_averages_df = pd.DataFrame(group_averages)
    return group_averages_df.T


# Radar chart creation function adapted for Streamlit and Plotly
# @st.cache(allow_output_mutation=True)
def make_radar_chart(group_averages_df):
    fig = go.Figure()
    cmap = cm.get_cmap("tab20b")
    angles = list(group_averages_df.index)
    angles.append(angles[0])

    for i, col in enumerate(group_averages_df.columns):
        data = group_averages_df[col].values.tolist()
        data.append(data[0])  # Complete the loop
        fig.add_trace(
            go.Scatterpolar(
                r=data,
                theta=angles,
                mode="lines+markers",
                line_color="rgba" + str(cmap(i / len(group_averages_df.columns))),
                # fill='toself',
                name=col,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
                # Make text labels larger and bold
                tickmode="array",
                tickfont=dict(size=12, color="black"),
            )
        ),
        showlegend=True,
    )
    return fig


# Streamlit app
st.title("Radar Chart of Group Averages")
# Streamlit app adjustments for wider layout

# Let users select methods
methods = llama_df_2.columns[1:]  # Exclude 'task' column
# Convert the pandas Index object to a list for the 'default' parameter
selected_methods = st.multiselect(
    "Select methods to display:", options=methods, default=methods.tolist()
)

if selected_methods:
    # Calculate group averages based on selected methods
    group_averages_df = calculate_group_averages(selected_methods)

    # Create and display the radar chart
    fig = make_radar_chart(group_averages_df)
    st.plotly_chart(fig)
else:
    st.write("Please select at least one method to display.")
