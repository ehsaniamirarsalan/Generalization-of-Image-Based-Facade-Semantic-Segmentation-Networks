import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os


def thesis_evaluation_visualizations(output_dir="visualization_outputs"):
    # Create dictionary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # lets define some colors
    natural_colors = ['#e63946', '#43aa8b', '#577590']  # i like it dark, why not :)

    # Background and text colors for dark theme
    dark_background = '#1f2937'  # Dark blue-gray
    grid_color = '#4b5563'  # Medium gray
    text_color = '#e5e7eb'  # Light gray

    # Define data for visualizations - this has to be taken from the evaluation script where you put in your confmat numpy array

    # Overall metrics data
    overall_metrics = pd.DataFrame([
        {'name': 'Mask R-CNN', 'Overall Pixel Accuracy': 0.5012, 'Mean IoU': 0.2564, 'Mean F1 Score': 0.5128},
        {'name': 'DeepLabV3+', 'Overall Pixel Accuracy': 0.6274, 'Mean IoU': 0.2042, 'Mean F1 Score': 0.4083},
        {'name': 'U-Net', 'Overall Pixel Accuracy': 0.5029, 'Mean IoU': 0.1188, 'Mean F1 Score': 0.2375}
    ])

    # Class-wise IoU data
    class_iou_data = pd.DataFrame([
        {'class': 'facade', 'Mask R-CNN': 0.1676, 'DeepLabV3+': 0.3353, 'U-Net': 0.2994},
        {'class': 'window', 'Mask R-CNN': 0.4298, 'DeepLabV3+': 0.3848, 'U-Net': 0.2696},
        {'class': 'door', 'Mask R-CNN': 0.0000, 'DeepLabV3+': 0.0073, 'U-Net': 0.0868},
        {'class': 'cornice', 'Mask R-CNN': 0.3559, 'DeepLabV3+': 0.2498, 'U-Net': 0.0584},
        {'class': 'sill', 'Mask R-CNN': 0.3313, 'DeepLabV3+': 0.0944, 'U-Net': 0.0415},
        {'class': 'balcony', 'Mask R-CNN': 0.3292, 'DeepLabV3+': 0.2190, 'U-Net': 0.0358},
        {'class': 'blind', 'Mask R-CNN': 0.2653, 'DeepLabV3+': 0.1754, 'U-Net': 0.0609},
        {'class': 'deco', 'Mask R-CNN': 0.1237, 'DeepLabV3+': 0.0875, 'U-Net': 0.0342},
        {'class': 'molding', 'Mask R-CNN': 0.3261, 'DeepLabV3+': 0.3218, 'U-Net': 0.2155},
        {'class': 'pillar', 'Mask R-CNN': 0.2313, 'DeepLabV3+': 0.1361, 'U-Net': 0.0164},
        {'class': 'shop', 'Mask R-CNN': 0.3134, 'DeepLabV3+': 0.1207, 'U-Net': 0.0216}
    ])

    # Class-wise F1 data
    class_f1_data = pd.DataFrame([
        {'class': 'facade', 'Mask R-CNN': 0.3351, 'DeepLabV3+': 0.6705, 'U-Net': 0.5987},
        {'class': 'window', 'Mask R-CNN': 0.8596, 'DeepLabV3+': 0.7697, 'U-Net': 0.5392},
        {'class': 'door', 'Mask R-CNN': 0.0000, 'DeepLabV3+': 0.0145, 'U-Net': 0.1736},
        {'class': 'cornice', 'Mask R-CNN': 0.7118, 'DeepLabV3+': 0.4996, 'U-Net': 0.1167},
        {'class': 'sill', 'Mask R-CNN': 0.6626, 'DeepLabV3+': 0.1888, 'U-Net': 0.0831},
        {'class': 'balcony', 'Mask R-CNN': 0.6584, 'DeepLabV3+': 0.4380, 'U-Net': 0.0717},
        {'class': 'blind', 'Mask R-CNN': 0.5306, 'DeepLabV3+': 0.3508, 'U-Net': 0.1217},
        {'class': 'deco', 'Mask R-CNN': 0.2474, 'DeepLabV3+': 0.1750, 'U-Net': 0.0684},
        {'class': 'molding', 'Mask R-CNN': 0.6523, 'DeepLabV3+': 0.6436, 'U-Net': 0.4310},
        {'class': 'pillar', 'Mask R-CNN': 0.4626, 'DeepLabV3+': 0.2722, 'U-Net': 0.0328},
        {'class': 'shop', 'Mask R-CNN': 0.6267, 'DeepLabV3+': 0.2415, 'U-Net': 0.0432}
    ])

    # Average performance data
    avg_performance_data = pd.DataFrame([
        {'network': 'Mask R-CNN', 'Avg Pixel Acc': 0.4529, 'Avg IoU': 0.2612,
         'Avg Precision': 0.6597, 'Avg Recall': 0.4529, 'Avg F1': 0.5225},
        {'network': 'DeepLabV3+', 'Avg Pixel Acc': 0.3226, 'Avg IoU': 0.1938,
         'Avg Precision': 0.7488, 'Avg Recall': 0.3226, 'Avg F1': 0.3877},
        {'network': 'U-Net', 'Avg Pixel Acc': 0.1902, 'Avg IoU': 0.1036,
         'Avg Precision': 0.3497, 'Avg Recall': 0.1902, 'Avg F1': 0.2073}
    ])

    # Radar chart data (scaled to percentage for better visualization)
    radar_data = class_iou_data.copy()
    radar_data['Mask R-CNN'] = radar_data['Mask R-CNN'] * 100
    radar_data['DeepLabV3+'] = radar_data['DeepLabV3+'] * 100
    radar_data['U-Net'] = radar_data['U-Net'] * 100

    # Generate and save visualizations with dark natural theme
    create_overall_metrics_chart(overall_metrics, output_dir, natural_colors, dark_background, grid_color, text_color)
    create_radar_chart(radar_data, output_dir, natural_colors, dark_background, grid_color, text_color)
    create_class_iou_chart(class_iou_data, output_dir, natural_colors, dark_background, grid_color, text_color)
    create_class_f1_chart(class_f1_data, output_dir, natural_colors, dark_background, grid_color, text_color)
    create_average_performance_chart(avg_performance_data, output_dir, natural_colors, dark_background, grid_color,
                                     text_color)
    create_performance_line_chart(class_iou_data, class_f1_data, output_dir, natural_colors, dark_background,
                                  grid_color, text_color)

    print(f"All visualizations have been saved to {output_dir}")


def create_overall_metrics_chart(data, output_dir, color_scheme, bg_color, grid_color, text_color):
    # letz Melt the dataframe for easier plotting
    melted_data = pd.melt(data, id_vars=['name'],
                          value_vars=['Overall Pixel Accuracy', 'Mean IoU', 'Mean F1 Score'],
                          var_name='Metric', value_name='Value')

    # Create the grouped bar chart
    fig = px.bar(melted_data, x='name', y='Value', color='Metric', barmode='group',
                 title='Overall Network Performance',
                 labels={'name': 'Network', 'Value': 'Score'},
                 color_discrete_sequence=color_scheme)

    # Adjust layout for dark theme
    fig.update_layout(
        font=dict(size=14, color=text_color),
        xaxis_title='Network',
        yaxis_title='Score',
        legend_title='Metric',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title_font_color=text_color
    )

    # Update grid and axis lines
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)

    # Save the figure
    fig.write_html(f"{output_dir}/overall_metrics_chart.html")
    fig.write_image(f"{output_dir}/overall_metrics_chart.png", width=1000, height=600, scale=2)

    return fig


def create_radar_chart(data, output_dir, color_scheme, bg_color, grid_color, text_color):
    # Create the radar chart
    fig = go.Figure()

    # Add traces for each network with natural colors
    fig.add_trace(go.Scatterpolar(
        r=data['Mask R-CNN'],
        theta=data['class'],
        fill='toself',
        name='Mask R-CNN',
        line_color=color_scheme[0],
        fillcolor=f'rgba({int(color_scheme[0][1:3], 16)}, {int(color_scheme[0][3:5], 16)}, {int(color_scheme[0][5:7], 16)}, 0.5)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=data['DeepLabV3+'],
        theta=data['class'],
        fill='toself',
        name='DeepLabV3+',
        line_color=color_scheme[1],
        fillcolor=f'rgba({int(color_scheme[1][1:3], 16)}, {int(color_scheme[1][3:5], 16)}, {int(color_scheme[1][5:7], 16)}, 0.5)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=data['U-Net'],
        theta=data['class'],
        fill='toself',
        name='U-Net',
        line_color=color_scheme[2],
        fillcolor=f'rgba({int(color_scheme[2][1:3], 16)}, {int(color_scheme[2][3:5], 16)}, {int(color_scheme[2][5:7], 16)}, 0.5)'
    ))

    # Update layout for dark theme
    fig.update_layout(
        title='IoU Comparison Across Classes (Radar Chart)',
        font=dict(size=14, color=text_color),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 50],  # Max value is 50%
                color=text_color,
                gridcolor=grid_color
            ),
            angularaxis=dict(
                color=text_color,
                gridcolor=grid_color
            ),
            bgcolor=bg_color
        ),
        showlegend=True,
        legend=dict(font=dict(color=text_color)),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title_font_color=text_color
    )

    # Save the figure
    fig.write_html(f"{output_dir}/radar_chart.html")
    fig.write_image(f"{output_dir}/radar_chart.png", width=1000, height=800, scale=2)

    return fig


def create_class_iou_chart(data, output_dir, color_scheme, bg_color, grid_color, text_color):
    # Melt the dataframe for easier plotting
    melted_data = pd.melt(data, id_vars=['class'],
                          value_vars=['Mask R-CNN', 'DeepLabV3+', 'U-Net'],
                          var_name='Network', value_name='IoU')

    # Create the grouped bar chart with natural colors
    fig = px.bar(melted_data, x='class', y='IoU', color='Network', barmode='group',
                 title='IoU Values by Class',
                 labels={'class': 'Class', 'IoU': 'IoU Score'},
                 color_discrete_sequence=color_scheme)

    # Adjust layout for dark theme
    fig.update_layout(
        font=dict(size=14, color=text_color),
        xaxis_title='Class',
        yaxis_title='IoU Score',
        legend_title='Network',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title_font_color=text_color
    )

    # Update grid and axis lines
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color, tickangle=-45)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)

    # Save the figure
    fig.write_html(f"{output_dir}/class_iou_chart.html")
    fig.write_image(f"{output_dir}/class_iou_chart.png", width=1200, height=700, scale=2)

    return fig


def create_class_f1_chart(data, output_dir, color_scheme, bg_color, grid_color, text_color):
    # Melt the dataframe for easier plotting
    melted_data = pd.melt(data, id_vars=['class'],
                          value_vars=['Mask R-CNN', 'DeepLabV3+', 'U-Net'],
                          var_name='Network', value_name='F1')

    # Create the grouped bar chart with natural colors
    fig = px.bar(melted_data, x='class', y='F1', color='Network', barmode='group',
                 title='F1 Scores by Class',
                 labels={'class': 'Class', 'F1': 'F1 Score'},
                 color_discrete_sequence=color_scheme)

    # Adjust layout for dark theme
    fig.update_layout(
        font=dict(size=14, color=text_color),
        xaxis_title='Class',
        yaxis_title='F1 Score',
        legend_title='Network',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title_font_color=text_color
    )

    # Update grid and axis lines
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color, tickangle=-45)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)

    # Save the figure
    fig.write_html(f"{output_dir}/class_f1_chart.html")
    fig.write_image(f"{output_dir}/class_f1_chart.png", width=1200, height=700, scale=2)

    return fig


def create_average_performance_chart(data, output_dir, color_scheme, bg_color, grid_color, text_color):
    # Define metric colors based on the natural color scheme
    metric_colors = [color_scheme[0], color_scheme[1], color_scheme[2],
                     f'#{int(color_scheme[0][1:3], 16) // 2:02x}{int(color_scheme[1][3:5], 16) // 2:02x}{int(color_scheme[2][5:7], 16) // 2:02x}',
                     f'#{int(color_scheme[2][1:3], 16) // 2:02x}{int(color_scheme[0][3:5], 16) // 2:02x}{int(color_scheme[1][5:7], 16) // 2:02x}']

    # Melt the dataframe for easier plotting
    melted_data = pd.melt(data, id_vars=['network'],
                          value_vars=['Avg Pixel Acc', 'Avg IoU', 'Avg Precision', 'Avg Recall', 'Avg F1'],
                          var_name='Metric', value_name='Value')

    # Create the grouped bar chart with appropriate colors
    fig = px.bar(melted_data, x='network', y='Value', color='Metric', barmode='group',
                 title='Average Performance Across Classes',
                 labels={'network': 'Network', 'Value': 'Average Score'},
                 color_discrete_sequence=metric_colors)

    # Adjust layout for dark theme
    fig.update_layout(
        font=dict(size=14, color=text_color),
        xaxis_title='Network',
        yaxis_title='Average Score',
        legend_title='Metric',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title_font_color=text_color
    )

    # Update grid and axis lines
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)

    # Save the figure
    fig.write_html(f"{output_dir}/average_performance_chart.html")
    fig.write_image(f"{output_dir}/average_performance_chart.png", width=1000, height=600, scale=2)

    return fig


def create_performance_line_chart(iou_data, f1_data, output_dir, color_scheme, bg_color, grid_color, text_color):
    # Create the line chart with natural colors
    fig = go.Figure()

    # Add traces for each network's IoU
    fig.add_trace(go.Scatter(
        x=iou_data['class'],
        y=iou_data['Mask R-CNN'],
        mode='lines+markers',
        name='Mask R-CNN IoU',
        line=dict(color=color_scheme[0], width=2),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=iou_data['class'],
        y=iou_data['DeepLabV3+'],
        mode='lines+markers',
        name='DeepLabV3+ IoU',
        line=dict(color=color_scheme[1], width=2),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=iou_data['class'],
        y=iou_data['U-Net'],
        mode='lines+markers',
        name='U-Net IoU',
        line=dict(color=color_scheme[2], width=2),
        marker=dict(size=8)
    ))

    # Update layout for dark theme
    fig.update_layout(
        title='IoU Performance Comparison',
        font=dict(size=14, color=text_color),
        xaxis_title='Class',
        yaxis_title='IoU Score',
        legend_title='Network',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title_font_color=text_color
    )

    # Update grid and axis lines
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color, tickangle=-45)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)

    # Save the figure
    fig.write_html(f"{output_dir}/performance_line_chart.html")
    fig.write_image(f"{output_dir}/performance_line_chart.png", width=1200, height=700, scale=2)

    # Create a second line chart for F1 scores with natural colors
    fig_f1 = go.Figure()

    # Add traces for each network's F1
    fig_f1.add_trace(go.Scatter(
        x=f1_data['class'],
        y=f1_data['Mask R-CNN'],
        mode='lines+markers',
        name='Mask R-CNN F1',
        line=dict(color=color_scheme[0], width=2),
        marker=dict(size=8)
    ))

    fig_f1.add_trace(go.Scatter(
        x=f1_data['class'],
        y=f1_data['DeepLabV3+'],
        mode='lines+markers',
        name='DeepLabV3+ F1',
        line=dict(color=color_scheme[1], width=2),
        marker=dict(size=8)
    ))

    fig_f1.add_trace(go.Scatter(
        x=f1_data['class'],
        y=f1_data['U-Net'],
        mode='lines+markers',
        name='U-Net F1',
        line=dict(color=color_scheme[2], width=2),
        marker=dict(size=8)
    ))

    # Update layout for dark theme
    fig_f1.update_layout(
        title='F1 Score Performance Comparison',
        font=dict(size=14, color=text_color),
        xaxis_title='Class',
        yaxis_title='F1 Score',
        legend_title='Network',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        title_font_color=text_color
    )

    # Update grid and axis lines
    fig_f1.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color, tickangle=-45)
    fig_f1.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)

    # Save the figure
    fig_f1.write_html(f"{output_dir}/f1_line_chart.html")
    fig_f1.write_image(f"{output_dir}/f1_line_chart.png", width=1200, height=700, scale=2)

    return fig, fig_f1


def create_combined_dashboard(output_dir="visualization_outputs"):
    thesis_evaluation_visualizations(output_dir)

    # Create an HTML file that includes all visualizations with dark theme cause we like it dark
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Facade Segmentation Network Analysis Dashboard</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #1f2937; 
                color: #e5e7eb;
            }
            h1 { text-align: center; color: #e5e7eb; }
            .dashboard { display: flex; flex-wrap: wrap; justify-content: center; }
            .chart-container { 
                margin: 10px; 
                padding: 10px; 
                box-shadow: 0 0 10px rgba(0,0,0,0.3); 
                background-color: #111827;
                border-radius: 8px;
            }
            .full-width { width: 100%; }
            .half-width { width: calc(50% - 40px); min-width: 400px; }
            iframe { border: none; background-color: #111827; }
            .network-color { display: inline-block; width: 15px; height: 15px; margin-right: 5px; }
            .mask-rcnn { background-color: #e63946; }
            .deeplabv3 { background-color: #43aa8b; }
            .unet { background-color: #577590; }
        </style>
    </head>
    <body>
        <h1>Facade Semantic Segmentation Network Analysis</h1>

        <div style="text-align: center; margin-bottom: 20px;">
            <div style="display: inline-block; margin: 0 15px;"><span class="network-color mask-rcnn"></span> Mask R-CNN</div>
            <div style="display: inline-block; margin: 0 15px;"><span class="network-color deeplabv3"></span> DeepLabV3+</div>
            <div style="display: inline-block; margin: 0 15px;"><span class="network-color unet"></span> U-Net</div>
        </div>

        <div class="dashboard">
            <div class="chart-container full-width">
                <h2>Overall Network Performance</h2>
                <iframe src="overall_metrics_chart.html" width="100%" height="500px"></iframe>
            </div>

            <div class="chart-container half-width">
                <h2>IoU Comparison (Radar Chart)</h2>
                <iframe src="radar_chart.html" width="100%" height="600px"></iframe>
            </div>

            <div class="chart-container half-width">
                <h2>Average Performance</h2>
                <iframe src="average_performance_chart.html" width="100%" height="600px"></iframe>
            </div>

            <div class="chart-container full-width">
                <h2>IoU Values by Class</h2>
                <iframe src="class_iou_chart.html" width="100%" height="600px"></iframe>
            </div>

            <div class="chart-container full-width">
                <h2>F1 Scores by Class</h2>
                <iframe src="class_f1_chart.html" width="100%" height="600px"></iframe>
            </div>

            <div class="chart-container full-width">
                <h2>IoU Performance Comparison</h2>
                <iframe src="performance_line_chart.html" width="100%" height="600px"></iframe>
            </div>

            <div class="chart-container full-width">
                <h2>F1 Score Performance Comparison</h2>
                <iframe src="f1_line_chart.html" width="100%" height="600px"></iframe>
            </div>
        </div>
    </body>
    </html>
    """

    # Write the HTML file
    with open(f"{output_dir}/dashboard.html", "w") as f:
        f.write(html_content)

    print(f"Combined dashboard has been saved to {output_dir}/dashboard.html")


if __name__ == "__main__":
    # Generate all visualizations and create the dashboard
    create_combined_dashboard()