import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_beautiful_concept_treemap(csv_file):
    df = pd.read_csv(csv_file)

    # Normalize CTR for coloring
    df["ctr_norm"] = (df["ctr"] - df["ctr"].min()) / (df["ctr"].max() - df["ctr"].min())

    # Hover text
    df["hover"] = (
        "<b style='font-size:16px'>" + df["concept"] + "</b><br><br>" +
        "<b>Exposure:</b> " + df["exposure_pct"].round(2).astype(str) + "%<br>" +
        "<b>CTR:</b> " + df["ctr"].round(2).astype(str) + "%<br>" +
        "<b>RPC:</b> $" + df["rpc"].round(2).astype(str) + "<br><br>" +
        "<b>Top Keywords:</b><br>• " +
        df["matched_keywords"].apply(lambda x: "<br>• ".join(x.split("; ")[:5]))
    )

    # Beautiful modern color palette (Soft but expressive)
    modern_palette = [
        "#FF6B6B",  # red
        "#FFA94D",  # orange
        "#FFD43B",  # yellow
        "#82C91E",  # green
        "#2F9E44"   # deep green
    ]

    fig = px.treemap(
        df,
        path=["concept"],
        values="exposure_pct",
        color="ctr_norm",
        hover_data={"hover": True},
        color_continuous_scale=modern_palette,
        maxdepth=1
    )

    # BEAUTIFICATION
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        texttemplate="<b>%{label}</b>",
        textfont=dict(
            size=14,
            family="Segoe UI Semibold",
        ),
        tiling=dict(
            pad=4,           # spacing between tiles
            packing="squarify"
        ),
        marker=dict(
            line=dict(width=1.5, color="rgba(255,255,255,0.65)"),  # glassy outline
        )
    )

    fig.update_layout(
        title={
            "text": (
                "<span style='font-size:28px; font-weight:600; font-family:Segoe UI'>"
                "Concept Exposure Treemap</span><br>"
                "<span style='font-size:16px; color:#666; font-family:Segoe UI'>"
                "Tile Size = Exposure % &nbsp;|&nbsp; Tile Color = CTR Performance"
                "</span>"
            ),
            "x": 0.5,
            "xanchor": "center"
        },
        margin=dict(t=100, l=20, r=20, b=20),
        paper_bgcolor="#f2f4f8",
        plot_bgcolor="#f2f4f8",

        coloraxis_colorbar=dict(
            title=dict(
                text="CTR",
                font=dict(size=13, family="Segoe UI Semibold")
            ),
            lenmode="fraction",
            len=0.8,
            thickness=18,
            tickfont=dict(size=12, family="Segoe UI")
        ),

        width=1400,
        height=900
    )

    return fig


# MAIN
if __name__ == "__main__":
    input_csv = "concept_output.csv"
    print(f"Generating beautiful treemap for {input_csv}...")

    fig = create_beautiful_concept_treemap(input_csv)
    fig.show()
    fig.write_html("concept_exposure_treemap_beautiful.html")

    print("✨ Beautiful treemap saved to 'concept_exposure_treemap_beautiful.html'")
