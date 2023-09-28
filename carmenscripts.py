from ast import Return
from math import nan
import os
import sys
from ipywidgets.widgets.trait_types import date_from_json

import json
import numpy as np
import pandas as pd
from IPython.core.display import display
from PIL import ImageDraw, ImageFont, Image
from ipywidgets import widgets
from scipy.spatial.distance import cdist
import skimage.io
from sklearn.cluster import KMeans
from tqdm.notebook import trange
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import metrics

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


cluster_max_dist = 0.15
bead_class_col = "Bead_Class_Cluster"
single_image_name = "Blue"

# Disable pandas assignment warnings for end users
pd.options.mode.chained_assignment = None

class Labeldict(dict):
    def __missing__(self, key):
        return key


def load_and_merge_data(meta_csv, object_csv, donut_csv, droplet_csv):
    from tqdm.notebook import trange
    i = 0
    with trange(6) as t:
        t.bar_format = "{desc} {bar} {percentage:3.0f}%"
        t.set_description(f'Working on {meta_csv}')

        # Read in image dataframe, construct meta dataframe
        image_df = pd.read_csv(meta_csv, engine='python')

        meta_df = image_df[['ImageNumber', f'FileName_{single_image_name}', f'PathName_{single_image_name}', 'Metadata_Chip', 'Metadata_Lane', 'Metadata_Timepoint', 'Metadata_Date', 'Metadata_Experiment']]
        t.set_description(f'Working on {object_csv}')
        t.update(1)

        # Read in beads.
        main_df = pd.read_csv(object_csv, engine='python')
        main_df = main_df.merge(meta_df, on="ImageNumber")
        # Turn filenames into shorter sample names
        main_df['Sample'] = main_df[f'FileName_{single_image_name}'].str.strip('.tif')
        # Add a dataframe column to group samples from both fields into one
        main_df['SampleShort'] = main_df['Sample'].apply(lambda x: x[:-2])

        t.set_description(f'Working on {donut_csv}')
        t.update(1)

        # Read in donuts.
        donut_df = pd.read_csv(donut_csv, engine='python')
        donut_df = donut_df.merge(meta_df, on="ImageNumber")

        t.set_description(f'Working on {droplet_csv}')
        t.update(1)

        # Read in droplets.
        droplet_df = pd.read_csv(droplet_csv, engine='python')
        droplet_df = droplet_df.merge(meta_df, on="ImageNumber")

        t.set_description(f'Generating metadata fields')
        t.update(1)

        # Generate and add metadata fields from file and folder names
        meta_df_2 = main_df.loc[:, [f'FileName_{single_image_name}', f'PathName_{single_image_name}']]
        #meta_df_2['Chip'] = meta_df_2[f'PathName_{single_image_name}'].apply(lambda x: x.replace('/', '\\').split('\\')[-1].split(' ')[0])
        main_df = main_df.rename(columns={'Metadata_Date': 'Date', "Metadata_Chip": "Chip", "Metadata_Timepoint": "Timepoint", "Metadata_Lane": "Lane"})
        main_df = main_df.astype({"Date": str, "Lane": str, "Chip":str, "Timepoint":str})
        main_df["Timepoint"] = main_df["Timepoint"].str.zfill(2) #new
        main_df = pd.concat([main_df, meta_df_2.drop([f'PathName_{single_image_name}', f'FileName_{single_image_name}'], axis=1)], axis=1)
        
        # Create a label column which properly describes all the groups
        cols=["Chip", "Date", "Timepoint", "Lane"]
        main_df["Chip_Date_Time_Lane"] = main_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        
        # Isolate donut metadata which we want to add to the main table
        donut_meta = donut_df[[
            'ImageNumber', f'FileName_{single_image_name}', 'Parent_FilteredBeads',
            'Intensity_MeanIntensityEdge_CropBlue',
            'Intensity_MeanIntensityEdge_CropGreenCorrected',
            'Intensity_MeanIntensityEdge_CropRedCorrected',
            'Intensity_MeanIntensityEdge_CropYellowCorrected',
            'Intensity_MeanIntensity_CropBlue',
            'Intensity_MeanIntensity_CropGreenCorrected',
            'Intensity_MeanIntensity_CropRedCorrected',
            'Intensity_MeanIntensity_CropYellowCorrected',
            'Intensity_MedianIntensity_CropBlue',
            'Intensity_MedianIntensity_CropGreenCorrected',
            'Intensity_MedianIntensity_CropRedCorrected',
            'Intensity_MedianIntensity_CropYellowCorrected',
            'Intensity_IntegratedIntensity_CropBlue',
            'Intensity_IntegratedIntensity_CropGreenCorrected',
            'Intensity_IntegratedIntensity_CropRedCorrected',
            'Intensity_IntegratedIntensity_CropYellowCorrected',
            'AreaShape_Area'
        ]]

        # Insert a prefix to the donut meta table
        donut_meta = donut_meta.add_prefix("Donut_")

        # Isolate donut metadata which we want to add to the main table
        droplet_meta = droplet_df[[
            'ImageNumber', f'FileName_{single_image_name}', 'ObjectNumber',
            'Intensity_MeanIntensity_CropBlue',
            'Intensity_MedianIntensity_CropBlue',
            'AreaShape_Area'
        ]]

        # Insert a prefix to the donut meta table
        droplet_meta = droplet_meta.add_prefix("Droplet_")

        t.set_description(f'Merging data frames')
        t.update(1)

        # Merge in donut information
        main_df = main_df.merge(
            donut_meta,
            how='left',
            left_on=["ImageNumber", "Parent_FilteredBeads"], #RS changed from ObjectNumber to Parent_FilteredBeads
            right_on=["Donut_ImageNumber","Donut_Parent_FilteredBeads"]
        )

        # Merge in droplet information
        main_df = main_df.merge(
            droplet_meta,
            how='left',
            left_on=["ImageNumber", "Parent_AcceptedDroplets"],
            right_on=["Droplet_ImageNumber","Droplet_ObjectNumber"]
        )
        t.update(1)
        t.set_description(f'Completed Loading')

    return main_df, image_df


def save_condensed_dataframe(data, outfile, extra_cols=None):
    condensed_columns = ["ImageNumber",
                         "Chip",
                         "Date",
                         "Timepoint",
                         "Lane",
                         "Chip_Date_Time_Lane",
                         "ObjectNumber",
                         bead_class_col,
                         "Intensity_MedianIntensity_CropBlue",
                         "Donut_Intensity_MedianIntensity_CropBlue",
                         "Droplet_AreaShape_Area",
                         ]
    if extra_cols is not None:
        condensed_columns += extra_cols
    main_df_condensed = data[condensed_columns].sort_values(['ImageNumber', bead_class_col])

    # Export simplified class-annotated dataframe to CSV
    main_df_condensed.to_csv(outfile, index=False)
    print(f"Wrote {outfile}")


def get_remappings(classes, classfile, samplesfile, null_class_name=None, palette="default"):
    # Apply more friendly labels to the bead classes and samples
    # Location of a csv containing 2 columns. Column 1 ("class") is the bead class code from the block above,
    # column 2 ("classname") is the name you want to assign. Ensure there is nothing else in the file.
    # Missing classes will retain their class code label

    class_remap = Labeldict()
    if os.path.exists(classfile):
        # Import labels and generate a new classes list.
        class_labels = pd.read_csv(classfile).dropna()
        for index, row in class_labels.iterrows():
            class_remap[row["class"]] = row["classname"]

        print(f"Imported {len(class_labels)} class labels")
        classes = [class_remap[i] for i in classes]

        # Fetch class order from the sheet
        ordered_classes = list(class_remap.values())
        # Remove unused class labels
        for x in set(ordered_classes) - set(classes):
            ordered_classes.remove(x)
        # Add back classes missing from the original list
        if len(classes) != len(ordered_classes):
            ordered_classes += [c for c in classes if c not in ordered_classes]
        if set(classes) != set(ordered_classes):
            print("WARNING: Something went wrong when importing the class label csv. Please report this")
        if null_class_name in class_remap:
            null_class_name = class_remap[null_class_name]

    else:
        ordered_classes = classes
        print(f"WARNING: Classes file {classfile} could not be found, no names were changed.")

    print(f"Adjusted class labels are {classes}")

    label_remap = Labeldict()
    if os.path.exists(samplesfile):
        sample_labels = pd.read_csv(samplesfile).dropna()
        for index, row in sample_labels.iterrows():
            label_remap[row["sample"]] = row["label"]
        print(f"Loaded {len(label_remap)} sample labels")
    else:
        print("No sample labels loaded")
    if palette=="default":
        palette=px.colors.qualitative.Dark24
    if palette=="alt1":
        palette=px.colors.qualitative.Plotly + px.colors.qualitative.Dark2
    class_color_dict = {}
    for index, classname in enumerate(classes):
        class_color_dict[classname] = palette[index]
    class_color_dict[null_class_name] = '#AAAAAA'

    return ordered_classes, class_remap, label_remap, class_color_dict


def load_rules_file(rulesfile):
    # Import rules list
    ruleslist = []
    line = 0
    with open(rulesfile) as rules:
        for line in rules:
            line = line[4:]  # Remove the IF
            rule, mods = line.split(',', 1)  # Isolate rule statement
            rule = rule.replace("AcceptedBeads_", "")  # Remove prefix (if using spreadsheets rather than database)
            modifiers = mods.split(')')[0]  # Remove trailing bracket from values
            modifiers = eval(modifiers)  # Convert values into pair of lists
            rulepack = (rule,) + modifiers
            ruleslist += (rulepack,)  # Package a tuple of each rule + values
    return ruleslist


def apply_rules(rules, data, classes):
    """
    deprecated function for applying rules from CPA to dataframe to class beads as VALID/INVALID and assign an initial guess based on measurements
    """
    if any(classname in data or 'Score_' + classname in data for classname in classes):
        # Prevent nasty accidents
        raise ValueError("Dataframe already appears to contain class score columns, did you run this cell already?")

    data = pd.concat([data, pd.DataFrame(columns=classes)])
    data[classes] = data.loc[:, classes].fillna(0)
    count = 0
    with trange(len(rules)) as t:
        t.bar_format = "{desc} {bar} {n_fmt}/{total_fmt}"
        t.set_description(f'Processing classifier rules')
        for ruleset in rules:
            count += 1
            ruleparts = ruleset[0].split(' ')
            selector = data.loc[:, ruleparts[0]] > float(ruleparts[-1])
            true_add = [float(x) for x in ruleset[1]]
            false_add = [float(x) for x in ruleset[2]]
            data.loc[selector, classes] += true_add
            data.loc[~selector, classes] += false_add
            t.update(1)
        t.set_description(f'Processed {count} rules')

    # Choose the 'best' class for each object and annotate columns.
    data['Bead_Class'] = data.loc[:, classes].idxmax(axis=1)
    data.rename(columns={classname: "Score_" + classname for classname in classes}, inplace=True)
    return data


def classify_kmeans(data, classes, class_remap, default_centroids,
                    class_col="Bead_Class_Cluster",
                    null_class_name="NULL", 
                    plotsdir=None):
    global bead_class_col
    bead_class_col = class_col
    # Generate KMeans classifier
    data_cols = ["Math_NormGreen", "Math_NormYellow", "Math_NormRed"]
    #cluster_df = data[data["Bead_Class"] != null_class_name]
    cluster_df=data
    kmeans_input = cluster_df.loc[:, data_cols]
    kmeans = KMeans(n_clusters=len(classes), init='k-means++', n_init=100).fit(kmeans_input) #changed from 20 to 100
    cluster_df.loc[:, bead_class_col] = kmeans.labels_

    # Generate base kmeans cluster labels for use with validation
    values_full = data[["Math_NormGreen", "Math_NormYellow", "Math_NormRed"]]
    data.loc[:, bead_class_col] = kmeans.predict(values_full)
    # Associate bead labels with bead centroids
    default_coords = list(default_centroids.keys())
    # Cols are class, rows are centroids
    cen_data = pd.DataFrame(cdist(kmeans.cluster_centers_, default_coords, metric="cityblock"))
    new_centroids = {}
    relabel_dict = {}
    centroids_table = pd.DataFrame(columns=["Class", "Class_Name", "Math_NormGreen", "Math_NormYellow", "Math_NormRed"])
    header = centroids_table.columns
    for col in cen_data.columns:
        class_id = cen_data.index[cen_data[col].argmin()]
        class_name = default_centroids[default_coords[col]]
        new_centroids[tuple(kmeans.cluster_centers_[class_id])] = class_name
        relabel_dict[class_id] = class_name
        cen_data = cen_data.drop(class_id, axis=0)
        centroid_series = pd.Series([f"<b>{class_name}</b>", class_name] + list(kmeans.cluster_centers_[class_id]),
                      index=header)
        # append row of data to the end of the table
        centroids_table.loc[len(centroids_table)] = centroid_series

    #Apply class labels
    data.loc[:, bead_class_col] = data[bead_class_col].map(relabel_dict)

    # Reject null beads
    data.loc[data["Bead_Class"] == null_class_name, bead_class_col] = null_class_name

    print(f"Clustering completed, found {kmeans.n_clusters} clusters")
    
    return data, kmeans, centroids_table


def add_silhouette_score(data, null_class_name="NULL",data_cols=["Math_NormGreen", "Math_NormYellow", "Math_NormRed"], class_col="Bead_Class_Cluster"):
    """
    Calculates silhouette score for each sample (bead) and adds it to data (usually main_df)
    :param data: data table containing measurements and which to write to (usually main_df)
    :param null_class_name: str, name of the null bead class that will not be considered for silhouette scores
    :param data_cols: array of data columns used by k-means clustering
    :param class_col: name of the class column containing designations for which rows are which bead classes
    """
    data["silhouette_score"]= np.nan
    not_null_beads = data[class_col]!=null_class_name
    kmeans_input = data.loc[not_null_beads, data_cols]
    kmeans_input = kmeans_input.dropna()
    y_predict = data.loc[not_null_beads,class_col]
    silhouette_vals = metrics.silhouette_samples(kmeans_input,y_predict) #slow!
    data.loc[not_null_beads,"silhouette_score"] = silhouette_vals
    score = silhouette_score(kmeans_input, y_predict, metric='euclidean')
    print(f"K-means silhouette score: {score} \nReminder: Silhouette score range is -1 to 1 where 1 is better separated clusters.")
    return data

def plot_silhouette_score(data, bead_class_col, class_color_dict, plotsdir, scale=1.3, plot_visualizer=True, plot_boxplot=True, show=True, save=True):
    """
    Plots silhouette scores for beads

    :return: Nothing
    """
    savedir = (f"{plotsdir}/diagnostic/")
    # Get mean used in both plots
    silhouette_vals = data["silhouette_score"]
    avg_score = np.mean(silhouette_vals)

    #Silhouette box plot version
    if plot_boxplot:
        sns.set_style("darkgrid")
        fig = px.box(data, x="silhouette_score", y=bead_class_col, color=bead_class_col, color_discrete_map=class_color_dict, width=600, height=400)
        fig.add_vline(x=avg_score, line_width=1, line_dash="dash", line_color="green")
        fig.update_traces(marker=dict(size=0.5))        
        if save:
            if not os.path.exists(plotsdir):
                os.mkdir(plotsdir)
            if not os.path.exists(savedir):
                os.mkdir(f"{savedir}")
            fig.write_image(f"{savedir}/silhouette_boxplot.png", scale=2.5)
        if show:
            fig.show()
    if plot_visualizer:
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6*scale,4*scale), dpi=100)
        bead_class_col = data[bead_class_col]
        y_lower = y_upper = 0
        for bead_class in np.unique(bead_class_col):
            cluster_silhouette_vals = silhouette_vals[bead_class_col ==bead_class]
            cluster_silhouette_vals = cluster_silhouette_vals.sort_values()
            y_upper += len(cluster_silhouette_vals)
            ax.barh(range(y_lower,y_upper),cluster_silhouette_vals,height =1, color=class_color_dict[bead_class], edgecolor = "none", alpha=0.7);
            ax.text(-0.03,(y_lower+y_upper)/2,bead_class, ha="right", va="center")
            y_lower += len(cluster_silhouette_vals)
        
        # Get the average silhouette score 
        ax.axvline(avg_score,linestyle ='--', linewidth =2,color = 'green')
        ax.set_yticks([])
        ax.set_xlim([0, 1])
        ax.set_xlabel('Silhouette coefficient values')
        ax.set_ylabel('Bead Class')
        ax.yaxis.set_label_coords(-0.175, 0.5)
        ax.set_title('Silhouette plot for the various clusters');
        if save:
            if not os.path.exists(f"{savedir}"):
                os.mkdir(f"{savedir}")
            fig.savefig(f"{savedir}/silhouette_vizualiser.png", dpi=300,bbox_inches="tight")
        if not show:
            plt.close()

def plot_donuts_over_time(tracked_beads, bead_class_col, class_color_dict, timepoint_list, plotsdir, show=True, save=True, col_wrap=8, scale=1, width=1000, height=800):
    """
    Plots donut blue fluorescence over time for each class and faceted by well

    :return: Nothing
    """
    savedir = plotsdir
    median_donuts = tracked_beads.groupby(["Lane", bead_class_col,"Timepoint"]).agg(Normalized_Donut_Intensity = ('Normalized_Donut_Intensity', 'median')).reset_index()
    fig = px.line(tracked_beads, 
                x="Timepoint", 
                y="Normalized_Donut_Intensity", 
                color=bead_class_col, 
                line_group="TrackObjects_Label", #change from Bead_Class_Cluster to trackobjects label
                facet_col="Lane", 
                facet_col_wrap=col_wrap, 
                category_orders={"Timepoint": timepoint_list},
                width=width, 
                height=height,
                color_discrete_map=class_color_dict)
    fig.update_traces(line=dict(width=1), opacity=.1, hovertemplate=None, hoverinfo='skip')
    fig.update_traces(showlegend=False)
    fig.add_traces(list(px.line(median_donuts, 
                                x="Timepoint", 
                                y="Normalized_Donut_Intensity", 
                                color=bead_class_col, 
                                facet_col="Lane", 
                                facet_col_wrap=col_wrap, 
                                width=width, 
                                height=height, 
                                color_discrete_map=class_color_dict).select_traces()))
    fig.add_hline(y=1, fillcolor="firebrick", opacity=0.9, line_dash="dot", line_width=2)
    fig.update_layout(font_family="Arial")
    fig.update_xaxes(tickfont_size=8)
    fig.update_yaxes(titlefont_size=12)
    fig.update_layout(legend=dict(itemsizing='constant'))
    if save:
        if not os.path.exists(f"{savedir}"):
            os.mkdir(f"{savedir}")
        fig.write_image(f"{savedir}/donuts_over_time.png", scale=scale)
    if show:
        fig.show()

def cluster_metrics(labels_true, labels):
    """
    Calculates metrics for clustering based on assigned labels vs. ground truth labels (if you have them)
    Must have sample_key.csv filled out and must have single well controls for this to work
    :param labels_true: vector of ground truth labels for each bead
    :param labels: assigned labels from clustering.
    :return: Nothing, prints metrics and warnings if values are bad, indicating poor clustering
    """
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels)
    #WARNINGS: / pass / fail
)


def adjust_clusters(data, adjusted_centroids, class_color_dict, null_class_name="NULL", limit=0):
    """
    Creates and displays the Cluster Adjustment Tool widget
    :param data: Bead data frame
    :param adjusted_centroids: Data table of bead class centroids.
    :param class_color_dict: Dictionary mapping class names to colours for display.
    :param null_class_name: Name of the 'null' class to be tossed out.
    :param limit: Maximum number of beads to plot.
    :return:
    """
    global cluster_max_dist
    init_table = adjusted_centroids.copy()

    plot_beads = data[data["Bead_Class"] != null_class_name].sample(limit)

    friendly_names = list(adjusted_centroids["Class_Name"])
    friendly_names.append(null_class_name)

    init_centroid = adjusted_centroids.iloc[0]

    coord_list = list(zip(plot_beads.Math_NormGreen, plot_beads.Math_NormYellow, plot_beads.Math_NormRed))
    colors_list = list(plot_beads[bead_class_col].copy().replace(class_color_dict))

    graphdata = [
        go.Scatterternary(a=plot_beads["Math_NormRed"],
                          b=plot_beads["Math_NormYellow"],
                          c=plot_beads["Math_NormGreen"],
                          mode="markers",
                          marker=dict(size=4, color=colors_list, line=dict(width=0), opacity=0.7),
                          showlegend=False,
                          hoverinfo='none'
                          ),
        go.Scatterternary(a=adjusted_centroids["Math_NormRed"],
                          b=adjusted_centroids["Math_NormYellow"],
                          c=adjusted_centroids["Math_NormGreen"],
                          mode="markers",
                          marker=dict(size=35, color="lightblue", line=dict(width=0), opacity=0.7),
                          showlegend=False,
                          hoverinfo='skip'
                          ),
        go.Scatterternary(a=adjusted_centroids["Math_NormRed"],
                          b=adjusted_centroids["Math_NormYellow"],
                          c=adjusted_centroids["Math_NormGreen"],
                          text=adjusted_centroids["Class"],
                          textfont_color="black",
                          mode="text",
                          showlegend=False,
                          hoverinfo='skip'
                          ),
    ]

    fig = go.FigureWidget(graphdata)
    labeltext = widgets.HTML(value="<b>Cluster adjustment tool</b>")

    class_selections = widgets.Dropdown(
        options=friendly_names,
        description='Class:',
        disabled=False,
    )

    green = widgets.BoundedFloatText(
        value=init_centroid["Math_NormGreen"],
        min=0,
        max=1.0,
        step=0.01,
        description='Green:',
        readout_format='.3f',
        continuous_update=False
    )

    yellow = widgets.BoundedFloatText(
        value=init_centroid["Math_NormYellow"],
        min=0,
        max=1.0,
        step=0.01,
        description='Yellow:',
        continuous_update=False
    )

    red = widgets.BoundedFloatText(
        value=init_centroid["Math_NormRed"],
        min=0,
        max=1.0,
        step=0.01,
        description='Red:',
        continuous_update=False
    )

    max_dist = widgets.BoundedFloatText(
        value=cluster_max_dist,
        min=0,
        max=4.0,
        step=0.01,
        description='Radius:',
        continuous_update=False
    )

    apply = widgets.Button(description="Apply", icon='check')

    load = widgets.Button(description="Load previous transform", icon='fa-upload')

    reset = widgets.Button(description="Reset", icon='reset')

    fig.update_layout({
        'ternary': {
            'aaxis': {'title': 'Red'},
            'baxis': {'title': 'Yellow'},
            'caxis': {'title': 'Green'}
        },
    }, autosize=False, width=500, height=500, margin=dict(
        l=20,
        r=20,
        b=20,
        t=20,
        pad=4
    ), modebar=None,
    )

    def on_validate(b):
        if green.value + yellow.value > 1:
            yellow.value = 1 - green.value
        red.value = 1 - green.value - yellow.value

    def on_apply(b):
        global cluster_max_dist
        on_validate(None)
        tgtindex = friendly_names.index(class_selections.value)
        data = adjusted_centroids.iloc[tgtindex].copy()
        data["Math_NormGreen"] = green.value
        data["Math_NormYellow"] = yellow.value
        data["Math_NormRed"] = red.value
        cluster_max_dist = max_dist.value
        adjusted_centroids.iloc[tgtindex] = data
        class_preview_names = generate_class_previews()
        with fig.batch_update():
            fig.data[0]['marker']['color'] = list(map(class_color_dict.get, class_preview_names))
            for trace in fig.data[1:]:
                trace['a'] = adjusted_centroids["Math_NormRed"]
                trace['b'] = adjusted_centroids["Math_NormYellow"]
                trace['c'] = adjusted_centroids["Math_NormGreen"]
        #save ternary_transform csv
        if b is not None: #only save out if button is pressed
            export_data = adjusted_centroids.copy()
            export_data['radius']=cluster_max_dist
            export_data.to_csv('ternary_transform.csv', index=False)

    def on_load(b):
        #Looks for ternary_transform csv
        global cluster_max_dist
        import_data = pd.read_csv('ternary_transform.csv')
        adjusted_centroids = import_data.iloc[:,:-1]
        cluster_max_dist = import_data['radius'].iloc[0]
        class_preview_names = generate_class_previews()
        for eachClass in adjusted_centroids["Class_Name"]:
            tgtindex = friendly_names.index(eachClass)
            data = adjusted_centroids.iloc[tgtindex].copy()
            class_selections.value = eachClass
            green.value = data["Math_NormGreen"]
            yellow.value = data["Math_NormYellow"]
            red.value = data["Math_NormRed"]
            max_dist.value = cluster_max_dist
            on_apply(None)
            #adjusted_centroids.iloc[tgtindex] = data
            #with fig.batch_update():
            #    fig.data[0]['marker']['color'] = list(map(class_color_dict.get, class_preview_names))
            #    for trace in fig.data[1:]:
            #        trace['a'] = adjusted_centroids["Math_NormRed"]
            #        trace['b'] = adjusted_centroids["Math_NormYellow"]
            #        trace['c'] = adjusted_centroids["Math_NormGreen"]

    def on_reset(b):
        global cluster_max_dist
        cluster_max_dist = 0.15
        max_dist.value = cluster_max_dist
        adjusted_centroids = init_table.copy()
        class_preview_names = generate_class_previews()
        with fig.batch_update():
            fig.data[0]['marker']['color'] = colors_list
            for trace in fig.data[1:]:
                trace['a'] = adjusted_centroids["Math_NormRed"]
                trace['b'] = adjusted_centroids["Math_NormYellow"]
                trace['c'] = adjusted_centroids["Math_NormGreen"]

    def on_click(trace, points, state):
        if not points.point_inds or not points.trace_name == "trace 0":
            return
        pointid = points.point_inds[0]
        green.value = trace['c'][pointid]
        yellow.value = trace['b'][pointid]
        red.value = trace['a'][pointid]
        on_apply(None)

    def generate_class_previews():
        adj_centroids_list = list(zip(adjusted_centroids["Math_NormGreen"], adjusted_centroids["Math_NormYellow"],
                                      adjusted_centroids["Math_NormRed"]))
        dist_matrix = cdist(coord_list, adj_centroids_list, metric="cityblock")
        dist_matrix = np.append(dist_matrix, np.full((dist_matrix.shape[0], 1), cluster_max_dist), 1)
        return [friendly_names[p] for p in dist_matrix.argmin(axis=1)]

    apply.on_click(on_apply)
    reset.on_click(on_reset)
    load.on_click(on_load)
    fig.data[0].on_click(on_click)

    container = widgets.VBox([labeltext, class_selections, green, yellow, red, max_dist, apply, load, reset])
    container.layout.align_items = 'flex-end'

    box = widgets.HBox([fig, container])
    display(box)
    dh = display(display_id=True)


def apply_cluster_adjustments(main_df, adjusted_centroids, null_class_name="NULL"):
    """
    Locks in adjustments made by the cluster adjustment tool
    :param main_df: Main dataframe
    :param adjusted_centroids: Table of centroids currently set up with the cluster adjustment tool.
    :param null_class_name: Name of the null class.
    :return: Main dataframe with new classes assigned to beads.
    """
    print("Using modified centroids. Assignment was performed based on distance.")
    global cluster_max_dist
    friendly_names = list(adjusted_centroids["Class_Name"])
    friendly_names.append(null_class_name)

    adj_centroids_list = list(zip(adjusted_centroids["Math_NormGreen"], adjusted_centroids["Math_NormYellow"],
                                  adjusted_centroids["Math_NormRed"]))
    coord_list = list(zip(main_df.Math_NormGreen, main_df.Math_NormYellow, main_df.Math_NormRed))

    dist_matrix = cdist(coord_list, adj_centroids_list, metric="cityblock")
    dist_matrix = np.append(dist_matrix, np.full((dist_matrix.shape[0], 1), cluster_max_dist), 1)

    alt_classes = [friendly_names[p] for p in dist_matrix.argmin(axis=1)]
    main_df[bead_class_col] = main_df["Bead_Class"]
    main_df[bead_class_col] = main_df[bead_class_col].where(main_df["Bead_Class"] == null_class_name, alt_classes)
    return main_df


def generate_overlays(main_df, image_df, tgt_dir, kind="rgb", fmt="tif", fontname=None):
    """
    Cycles through images and generates previews of class identities.
    :param main_df: Main data frame containing bead info.
    :param image_df: Per-image data frame detailing image file paths.
    :param tgt_dir: Output directory to write resulting images to.
    :param kind: Overlay type, see get_image.
    :param fmt: Output file format. 'tif' is faster but takes more drive space than 'png'.
    :param fontname: name of a font.ttf object to use as the font.
    :return:
    """
    if not os.path.exists(tgt_dir):
        os.mkdir(tgt_dir)
    if fontname:
        pass
    elif sys.platform == "win32":
        fontname = "arial.ttf"
    else:
        fontname = "Arial.ttf"
    total = len(image_df)
    with trange(total) as t:
        t.bar_format = "{desc} {bar} {n_fmt}/{total_fmt}{postfix}"
        font = ImageFont.truetype(fontname, 14, encoding="unic")
        with trange(0) as t2:
            t2.bar_format = "{desc} {bar} {n_fmt}/{total_fmt}"

            for index, row in image_df.iterrows():
                name = row["FileName_BeadsRed"]
                t.set_description(f'{name}')
                t.set_postfix_str("Loading images")
                image_data, suffix = get_image(row, kind)
                image_canvas = ImageDraw.Draw(image_data)
                beads_to_label = main_df[main_df["ImageNumber"] == row["ImageNumber"]]
                t.set_postfix_str("Drawing beads")
                if len(beads_to_label) > 0:
                    t2.reset(len(beads_to_label) - 1)
                    imagename = beads_to_label.iloc[0]['Chip_Date_Time_Lane']
                    for i, beadrow in beads_to_label.iterrows():
                        t2.set_description(f'Drawing bead labels')
                        beadx = int(beadrow['AreaShape_BoundingBoxMaximum_X'])
                        beady = int(beadrow['AreaShape_BoundingBoxMinimum_Y'])
                        beadclass = beadrow[bead_class_col]
                        image_canvas.text((beadx,beady), beadclass, fill='white', font=font)
                        t2.update(1)
                t.set_postfix_str("Saving preview")
                image_data.save(f"{tgt_dir}/{imagename}{suffix}.{fmt}")
                t.update(1)
        t.set_description("Completed process")
        t.set_postfix_str(f'Saved all {total} preview images to /{tgt_dir}!')


def get_image(row, kind="rgb"):
    """
    Loads image data from a file.
    :param row: Pandas row containing information about the desired image.
    :param kind: Specifies which image set to fetch. 'rgb' will overlay red, green and yellow channels as red, green and
    blue respectively. 'donuts' will fetch the donut result overlay. 'beads' will fetch the beads overlay. 'droplets'
    will fetch the droplet overlay image. Note that the particular image must have been saved and logged during the
    CellProfiler pipeline.
    :return: PIL Image
    """
    if kind == "rgb":
        red = skimage.io.imread(os.path.join(row["PathName_BeadsRed"], row["FileName_BeadsRed"]))
        yellow = skimage.io.imread(os.path.join(row["PathName_BeadsYellow"], row["FileName_BeadsYellow"]))
        green = skimage.io.imread(os.path.join(row["PathName_BeadsGreen"], row["FileName_BeadsGreen"]))
        data = np.dstack((red, green, yellow))
        suffix = "_preview"
    elif kind == "donuts":
        data = skimage.io.imread(os.path.join(row["PathName_DonutsOnBlue"], row["FileName_DonutsOnBlue"]))
        suffix = "_donuts"
    elif kind == "beads":
        data = skimage.io.imread(os.path.join(row["PathName_BeadsOnFluor"], row["FileName_BeadsOnFluor"]))
        suffix = "_beads"
    elif kind == "droplets":
        data = skimage.io.imread(os.path.join(row["PathName_DropletsOnBlue"], row["FileName_DropletsOnBlue"]))
        suffix = "_droplets"
    else:
        raise NotImplementedError(f"Image {kind} is not a valid overlay base image.")
    return Image.fromarray(data.astype("uint8")), suffix


def make_ternary_plot(data, centroids_table=None, class_color_dict=None, width=None, height=None, legend=True, opacity=0.4, size=3, null_class_name=None):
    """
    Creates a ternary plot.
    :param data: Input data to plot
    :param centroids_table: A table of centroids to display on the diagram. Optional.
    :param class_color_dict: Dictionary mapping classes to colours for plotting.
    :param width: Output width
    :param height: Output height
    :param legend: Whether to display legend.
    :param opacity: Opacity of points.
    :param size: Size of points.
    :param null_class_name: Name of the 'invalid' or 'null' class. If provided, this class will be plotted first so that
    it's points are behind those of other classes..
    """

    if not width:
        width = 800
    if not height:
        height = 800

    # if null_class_name:
    #     selector = data[bead_class_col] == null_class_name
    #     data = data[selector].append(data[~selector]).reset_index(drop=True)

    # Construct a ternary plot mapping KMeans centroids results, coloured by class

    fig = px.scatter_ternary(data,
                             a="Math_NormRed",
                             b="Math_NormYellow",
                             c="Math_NormGreen",
                             color=bead_class_col,
                             color_discrete_map=class_color_dict,
                             size=[size] * data.shape[0],
                             size_max=size,
                             opacity=opacity,
                             width=width,
                             height=height,
                             )

    fig.update_traces(marker=dict(line=dict(width=0)),
                      selector=dict(type='scatterternary'))

    fig.update_layout(showlegend=legend, title="Cluster positions", legend_title_text='<b>Classes</b>')
    if centroids_table is not None:
        fig.add_trace(go.Scatterternary(
            a=centroids_table["Math_NormRed"],
            b=centroids_table["Math_NormYellow"],
            c=centroids_table["Math_NormGreen"],
            mode="markers",
            marker=dict(size=35, color="lightgrey", line=dict(width=0), opacity=0.7),
            showlegend=False,
        ))

        fig.add_trace(go.Scatterternary(
            a=centroids_table["Math_NormRed"],
            b=centroids_table["Math_NormYellow"],
            c=centroids_table["Math_NormGreen"],
            text=centroids_table["Class"],
            textfont_color="black",
            mode="text",
            showlegend=False,
        ))

    # Calculate and add counts per class
    if legend:
        for i, item in enumerate(fig.data):
            namevar = item['name']
            count = len(item['a'])
            fig.data[i].name = f"{namevar}  <sub>({count})</sub>"

    return fig


def generate_ternary(data, class_color_dict, outdir, centroids_table=None, save=True, show=False, **kwargs):
    """
    Makes directories and generates ternary plots
    :param data: Input data frame for plotting.
    :param class_color_dict: Dictionary mapping classes to hex colours.
    :param outdir: Path where the output file should be saved.
    :param centroids_table: Optional table of centroids to display on the chart.
    :param save: Whether to save the output to a file. Disable this for testing.
    :param show: Whether to display the chart in the notebook. Enable this for testing. Displaying many plots is
    intensive so it's best to avoid showing all charts during a run.
    :param kwargs: Other arguments to pass through to plotly.
    :return:
    """
    if save and not os.path.exists(os.path.dirname(outdir)):
        os.mkdir(os.path.dirname(outdir))
    plot = make_ternary_plot(data, centroids_table=centroids_table, class_color_dict=class_color_dict, width=None, height=None, legend=True, **kwargs)
    if save:
        plot.write_image(outdir)
    if show:
        plot.show()


def subset_ternary(data, subset_param, class_color_dict, plotsdir, **kwargs):
    """
    Generates ternary plots for each sample.
    :param data: Main data frame
    :param subset_param: Metadata column to be used to split individual plots.
    :param class_color_dict: Dictionary mapping bead names to specific colours for display.
    :param plotsdir: Directory to save plots into.
    :param kwargs: Other arguments to be passed to the ternary generator.
    :return:
    """
    samples = pd.unique(data[subset_param])
    total = len(samples)
    with trange(total) as t:
        t.bar_format = "{desc} {bar} {n_fmt}/{total_fmt}"
        for samplename in samples:
            t.set_description(f'Working on {samplename}')

            subset_df = data[data[subset_param] == samplename]
            subset_df = subset_df.sort_values(bead_class_col, ascending=False)
            generate_ternary(subset_df, class_color_dict, f"{plotsdir}/ternary_per_image/{samplename}.png",
                             save=True, show=False, opacity=0.7, size=6, **kwargs)
            t.update(1)
        t.set_description(f"Completed export of all {total} plots")


def plot_intensity_all(data, plotsdir, classes, color, class_color_dict, stat="Median", save=True, show=False):
    """
    Constructs a plot of intensity per bead, split by class.
    :param data: Table of results to plot
    :param plotsdir: Directory to save resulting plot
    :param classes: List of classes in desired order to be displayed
    :param color: Wavelength to plot, as found in the image name. Usually capitalized.
    :param class_color_dict: Dictionary mapping classes to colors
    :param stat: Intensity statistic to plot, as found in the measurements table.
                 Useful options are typically "Mean", "Median", "Min", "Max" or "Integrated"
    :param save: Whether to save the plot to a file.
    :param show: Whether to display the plot in the notebook.
    :return:
    """
    data = data.set_index(bead_class_col).loc[classes].reset_index()

    # We generate an artifical legend
    markers = []
    for classname in classes:
        markers.append(Line2D([], [], color=None, linewidth=0, marker='.', markersize=15,
                              markerfacecolor=class_color_dict[classname]))

    mainfigure = sns.catplot(x=bead_class_col, y=f"Donut_Intensity_{stat}Intensity_Crop{color}",
                             data=data,
                             height=4, aspect=2,
                             hue=bead_class_col,
                             palette=class_color_dict,
                             linewidth=0,
                             )
    ident, cn = np.unique(data[bead_class_col], return_counts=True)
    countdict = dict(zip(ident, cn))
    counts = [f"{label} ({countdict.get(label, 0)})" for label in classes]
    plt.legend(markers, counts, bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.subplots_adjust(top=0.9)
    mainfigure.fig.suptitle(f'{stat} {color} Donut Intensity per Bead')
    mainfigure.set_xticklabels(rotation=-45, horizontalalignment='left', )
    mainfigure.set(xlabel='')
    if save:
        mainfigure.savefig(f"{plotsdir}/Class_{stat}Intensity_{color}.png")
    if not show:
        plt.close()


def plot_intensity_persample(data, plotsdir, classes, splitter, stat="Median", save=True, class_color_dict=None,
                             threshold_dict=None):
    """
    # Generates intensity plots, split by sample
    :param data: Main data frame
    :param plotsdir: Directory to save resulting plots into
    :param classes: List of classes
    :param splitter: Metadata column name used to seperate individual samples.
    :param stat: Intensity statistic to display. Case-sensitive. Typically "Mean" or "Median".
    :param save: If False, function stops after the first sample and displays the result as a preview instead of saving.
    :param class_color_dict: Dictionary mapping classes to colours.
    :param threshold_dict: Dictionary mapping sample ids to thresholds.
    :return:
    """
    if not os.path.exists(f"{plotsdir}/intensity_per_image"):
        os.mkdir(f"{plotsdir}/intensity_per_image")
    samples = pd.unique(data[splitter])
    total = len(samples)
    measure = f"Donut_Intensity_{stat}Intensity_CropBlue"
    ymax = data[measure].max() * 1.1

    # We generate an artifical legend
    markers = []
    for classname in classes:
        markers.append(Line2D([], [], color=None, linewidth=0, marker='.', markersize=15,
                              markerfacecolor=class_color_dict[classname]))

    with trange(total) as t:
        t.bar_format = "{desc} {bar} {n_fmt}/{total_fmt}"
        for sampleid in samples:
            t.set_description(f'Working on {sampleid}')
            to_plot = data[data[splitter] == sampleid]
            diff = set(classes) - set(to_plot[bead_class_col].unique())
            used_classes = classes.copy()
            for i in diff:
                used_classes.remove(i)
            to_plot = to_plot.set_index(bead_class_col).loc[used_classes].reset_index()
            mainfigure = sns.catplot(x=bead_class_col, y=measure,
                                     linewidth=0,
                                     data=to_plot,
                                     order=classes,
                                     palette=class_color_dict,
                                     height=4,
                                     aspect=2,
                                     legend=False,
                                     )
            if threshold_dict and sampleid in threshold_dict and threshold_dict[sampleid] < 1:
                plt.axhline(threshold_dict[sampleid], color='black', linestyle='dashed', label="FFFFFFFFFF")
                plt.annotate(f"{threshold_dict['STAT_DEVMULT']}x SD",
                             (1, threshold_dict[sampleid] / ymax),
                             annotation_clip=False, xycoords='axes fraction')
            plt.subplots_adjust(top=0.9)
            ident, cn = np.unique(to_plot[bead_class_col], return_counts=True)
            countdict = dict(zip(ident, cn))
            counts = [f"{label} ({countdict.get(label, 0)})" for label in classes]
            plt.legend(markers, counts, bbox_to_anchor=(1.05, 1), loc="upper left")
            mainfigure.fig.suptitle(f'{stat} Blue Donut Intensity per Bead - {sampleid}')
            mainfigure.set(ylim=(0, ymax))
            mainfigure.set_xticklabels(rotation=-45, horizontalalignment='left', )
            mainfigure.set(ylabel=f'{stat} Donut Blue Intensity')
            mainfigure.set(xlabel='')
            if save:
                mainfigure.savefig(f"{plotsdir}/intensity_per_image/{sampleid}.png")
                plt.close()
                t.update(1)
            else:
                t.reset()
                break


def generate_intensity_csvs_persample(data, plotsdir, classes, splitter, stat, save=True):
    if not os.path.exists(f"{plotsdir}/intensity_csvs"):
        os.mkdir(f"{plotsdir}/intensity_csvs")
    samples = pd.unique(data[splitter])
    total = len(samples)
    measure = f"Donut_Intensity_{stat}Intensity_CropBlue"
    with trange(total) as t:
        t.bar_format = "{desc} {bar} {n_fmt}/{total_fmt}"
        for sampleid in samples:
            t.set_description(f'Working on {sampleid}')
            to_export = data[data[splitter] == sampleid]
            data_dict = {'Sample': sampleid}
            for classname in classes:
                class_data = to_export[to_export[bead_class_col] == classname]
                data_dict[classname] = pd.Series(class_data[measure].tolist())
            export_df = pd.DataFrame(data_dict)
            if save:
                export_df.to_csv(f"{plotsdir}/intensity_csvs/{sampleid}.csv", index=False)
                t.update(1)
            else:
                t.reset()
                return export_df

def annotate_classified_beads(outputfile_condensed, samplekey):
    """
    merges the output file of bead measurements with the sample 
    key (user-generated file that labels each sample (well or lane)
    """
    beadtable = pd.read_csv(outputfile_condensed)
    key = pd.read_csv(samplekey)
    beadtable = beadtable.merge(key, left_on=["Chip_Date_Time_Lane"], right_on="sample", how='left')
    beadtable.to_csv("classified_beads_key_annotated.csv")
    return beadtable

def plot_errormap(beadtable, plotsdir, save=True, show=True):
    """
    Plots an error map (if you have single label controls, this is useful)
    """
    tab_norm = pd.crosstab(beadtable.label+beadtable.Chip.astype(str), beadtable.Bead_Class_Cluster, normalize="index")
    fig, ax = plt.subplots(figsize=(24,12))
    sns.set(style='white')
    sns.heatmap(tab_norm, square=True, annot=True, fmt=".2g", annot_kws={"size":8})
    ax.legend(ax.get_ylabel(), bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    if save:
        plt.savefig(f"{plotsdir}/errormap.png", dpi=300, bbox_inches='tight')
    if not show:
        plt.close()

def readyDataForStripPlot(data,thresholds, classes):
    """
    This function transforms the main data table to be ready to plot in the normalized stripplot function
    :param data: the main_df data frame
    :param thresholds: dictionary of thresholds to normalize the data per-lane
    :param classes: List of classes in desired order to be displayed. The function will exclude "NULL" beads
    """
    condensed_columns = ["ImageNumber",
                         "Chip",
                         "Date",
                         "Timepoint",
                         "Chip_Date_Time_Lane",
                         "Lane",
                         "ObjectNumber",
                         bead_class_col,
                         "Intensity_MedianIntensity_CropBlue",
                         "Donut_Intensity_MedianIntensity_CropBlue",
                         "Droplet_AreaShape_Area",
                         ]
    df = data[condensed_columns].sort_values(['ImageNumber', bead_class_col])
    df.drop(df.loc[df[bead_class_col]=='NULL'].index, inplace=True)
    df = df.groupby(['Chip','Date', 'Lane', "Timepoint", "Chip_Date_Time_Lane",'Bead_Class_Cluster'], as_index=False).median('Donut_Intensity_MedianIntensity_CropBlue')
    #organize according to the 'classes'
    mapping = {Bead_Class_Cluster: i for i, Bead_Class_Cluster in enumerate(classes)}
    key = df['Bead_Class_Cluster'].map(mapping)
    df = df.iloc[key.argsort()]
    df['threshold'] = [float(thresholds[sample_id]) for sample_id in df['Chip_Date_Time_Lane']]
    df['Normalized_Donut_Intensity'] = [raw/thres for raw, thres in zip(df['Donut_Intensity_MedianIntensity_CropBlue'], df['threshold'])]
    return df

def readyDataTrackedBeadsPlot(data, thresholds, classes):
    """
    This function transforms the main data table to be ready to plot in the plotIndBeadsOverTime function
    :param data: the main_df data frame
    :param thresholds: dictionary of thresholds to normalize the data per-lane
    :param classes: List of classes in desired order to be displayed. The function will exclude "NULL" beads
    """
    if 'TrackObjects_Label' not in data.columns:
        exit("'TrackObjects_Label' column missing in data. This is a function that requires output from a CP pipeline where TrackObjects is used to track individual beads")
    condensed_columns = ["ImageNumber",
                         "Chip",
                         "Date",
                         "Timepoint",
                         "Chip_Date_Time_Lane",
                         "Lane",
                         "ObjectNumber",
                         bead_class_col,
                         "Intensity_MedianIntensity_CropBlue",
                         "Donut_Intensity_MedianIntensity_CropBlue",
                         "Droplet_AreaShape_Area",
                         "TrackObjects_Label"
                         ]
    df = data[condensed_columns].sort_values(['ImageNumber', bead_class_col])
    #drop null beads that were not classified by k-means
    df.drop(df.loc[df[bead_class_col]=='NULL'].index, inplace=True)
    #drop nan beads that were not tracked over enough frames
    num_beads_removed = df['TrackObjects_Label'].isna().sum()
    total_beads = df.shape[0] # num rows
    percent_beads = '{:g}'.format(float('{:.3g}'.format(100*num_beads_removed/total_beads)))
    print(f"{num_beads_removed} images of beads removed by because they were not tracked across enough frames, representing {percent_beads}% of bead images")
    df = df[df['TrackObjects_Label'].notnull()]
    #df = df.groupby(['Chip','Date', 'Lane', "Timepoint", "Chip_Date_Time_Lane",'Bead_Class_Cluster', 'TrackObjects_Label'], as_index=False).median('Donut_Intensity_MedianIntensity_CropBlue')
    #organize according to the 'classes'
    mapping = {Bead_Class_Cluster: i for i, Bead_Class_Cluster in enumerate(classes)}
    key = df['Bead_Class_Cluster'].map(mapping)
    df = df.iloc[key.argsort()]
    df['threshold'] = [float(thresholds[sample_id]) for sample_id in df['Chip_Date_Time_Lane']]
    df['Normalized_Donut_Intensity'] = [raw/thres for raw, thres in zip(df['Donut_Intensity_MedianIntensity_CropBlue'], df['threshold'])]
    df.sort_values(by=['Chip','Lane',bead_class_col, 'TrackObjects_Label','Timepoint'], ascending=True, inplace=True)
    return df

def plot_norm_stripplot_wrapper(data, plotsdir, chip_list, timepoint_list, swapTimeandChip=False, showStat=False, stat="median", save=True, summarize=False, show=True, palette="muted"):
    """
    Wrapper function for constructing strip plots of normalized intensity per lane. 
    This function figures out which variable should be hue and which should be column in catplot and subsets palettes appropriately
    : params are same as plotting function but have additional switch param for overriding the column vs. hue decision 
    """
    if swapTimeandChip:
        hue_list=chip_list
        hue = "Chip"
        column_list=timepoint_list
        column = "Timepoint"
    else:
        hue_list=timepoint_list
        hue = "Timepoint"
        column_list=chip_list
        column = "Chip"
    plot_norm_stripplot(data, plotsdir, hue_list, hue, column_list, column, showStat=showStat, save=save, stat=stat, summarize=summarize, show=show, palette=palette)


def plot_norm_stripplot(data, plotsdir, hue_list, hue, column_list, column, showStat=False, pad_label=None, font_scale=1, stat="median", save=True, height=4, col_wrap=1, summarize=False, show=True, palette="muted"):
    """
    Constructs a strip plot of intensity per lane, normalized by the threshold level for that lane.
    :param df: Table of results to plot
    :param plotsdir: Directory to save resulting plot
    :param hue_list: list of items be plotted in different colors in the graph, used for hue_order in plotting
    :param hue: which data column should be coded by color (Timepoint or Chip)
    :param column_list: list of items to be plotted as separate graphs (columns)
    :param column: which data column should be used to separate data into different graphs (default is Chip)
    :param showStat: whether to show the mean or median of the data
    :param stat: Intensity statistic to plot, as found in the measurements table. Options are 'mean' or 'median'
    :param save: Whether to save the plot to a file.
    :param summarize: whether to plot summarized data as boxplots (default False, meaning all points are shown in stripplots)
    :param show: Whether to display the plot in the notebook.
    :param palette: the color scheme for the graph
    :return:
    Note: could separate the data frame operations into a different function that preps data for plotting
    """
    saveDir=plotsdir+"/Normalized_plots/"
    df = data.copy()
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    meanprops={'visible':False}
    medianprops={'visible':False}
    if stat=="median":
        medianprops={'color': 'k', 'ls': '-', 'lw': 2, 'visible':True}
    if stat=="mean":
        meanprops={'color': 'k', 'ls': '-', 'lw': 2, 'visible':True}
    if pad_label is not None:
        df[column]=df[column].astype(str).str.zfill(pad_label)
        column_list=np.sort(df[column].unique())
    sns.set(context='notebook', style='whitegrid', font_scale=font_scale)
    if summarize:
        if showStat and stat=="mean":
            g = sns.catplot(data=df, kind="box", x='Bead_Class_Cluster', y='Normalized_Donut_Intensity',hue=hue, hue_order=hue_list,col=column, col_order=column_list,
                   palette=palette, showmeans=True, meanprops={"marker": "o",
                       "markersize": "4", "alpha": 0.5}, height=height, col_wrap=col_wrap)
        else: 
            g = sns.catplot(data=df, kind="box", x='Bead_Class_Cluster', y='Normalized_Donut_Intensity',hue=hue, hue_order=hue_list,col=column, col_order=column_list,
                   palette=palette, height=height, col_wrap=col_wrap)

    else:
        if showStat:
            g = sns.catplot(kind="box", data=df, x='Bead_Class_Cluster', y='Normalized_Donut_Intensity',hue=hue, hue_order=hue_list,
                    palette=palette, col=column, col_order=column_list,showmeans=True,meanline=True,
                    meanprops=meanprops, medianprops=medianprops, whiskerprops={'visible': False}, showfliers=False, 
                    showbox=False, showcaps=False, zorder=10, height=height, col_wrap=col_wrap)
            g.map_dataframe(sns.stripplot, x='Bead_Class_Cluster', y='Normalized_Donut_Intensity',hue=hue, 
                    hue_order=hue_list, dodge=True, palette=palette, s=4, marker='o', edgecolor='#373737', alpha=0.5, linewidth=1, jitter=True)
        else:
            g = sns.catplot(data=df, x='Bead_Class_Cluster', y='Normalized_Donut_Intensity',hue=hue, hue_order=hue_list,
                dodge=True, palette=palette, col=column, col_order=column_list,marker='o', edgecolor='#373737', alpha=0.5, linewidth=1, height=height, col_wrap=col_wrap)
    for ax in g.axes.flat:
        ax.set_ylabel('Normalized Median Reporter Intensity')
        ax.set_xlabel('Bead Class')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', rotation_mode="anchor")
        ax.axhline(y=1, color='k', linestyle=(0,(1,5)), linewidth=1)
        plt.setp(ax.get_xticklabels(), visible=True, rotation=45)
        ax.tick_params(labelbottom=True, pad=0)

    plt.subplots_adjust(hspace=1)

    if len(column_list)==1:
        sep=f"col-{column_list[0]}"
    else:
        sep="col-"+column
    if len(hue_list)==1:
        color_name=f"hue-{hue_list[0]}"
    else:
        color_name="hue-"+hue
    if save and summarize:
        g.savefig(f"{saveDir}/BoxPlot_{stat}Intensity_{sep}_{color_name}.png")
    elif save:
        g.savefig(f"{saveDir}/StripPlot_{stat}Intensity_{sep}_{color_name}.png")
    if show:
        plt.show()
    if not show: 
        plt.close()

def plot_stacked_bar(beadtable, plotsdir, color_dict, save=True, show=True, normalize=True):
    """
    Function that plots stacked bar graphs of bead types within each condition
    :param data: beadtable data that has info on bead type, chip (aka plate), lane (aka well), Condition name, beadtype etc. 'classified_beads_key_annotated.csv'
    :param plotsdir: directory to save plots in
    :param save: whether to save the figure
    :param show: whether to show the figure in the notebook
    :param normalize: whether to normalize the stacked bar graph data or not
    :returns:
    """
    fig, ax = plt.subplots(figsize=(8,4))
    sns.set(style='white')

    if normalize:
        data = pd.crosstab(beadtable.label+beadtable.Chip.astype(str), beadtable.Bead_Class_Cluster, normalize="index")
    else:
        data = pd.crosstab(beadtable.label+' '+beadtable.Chip.astype(str), beadtable.Bead_Class_Cluster)
    data.plot(kind='bar', stacked=True, color=color_dict, ax=ax)
    plt.tick_params(axis='x', which='major', labelsize=8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    fig.tight_layout()
    if save and normalize:
        plt.savefig(f"{plotsdir}/Stacked_bar_norm.png", dpi=300)
    elif save and not normalize: 
        plt.savefig(f"{plotsdir}/Stacked_bar_raw.png", dpi=300)
    if not show:
        plt.close()

def plot_bar(beadtable, plotsdir, palette, save=True, show=True, normalize=False, col_wrap=1, height=4, font_scale=1):
    """
    Function that makes basic bar plots that are useful in comparing raw or relative 
    bead counts across different classes and expoeriment types 
    """
    tab_bar = beadtable.groupby(["Chip", "label", "Bead_Class_Cluster"], as_index=False).size()
    #tab_bar[['well_type','true_bead']] = tab_bar['label'].str.split(';', expand=True)

    if normalize:
        tab_bar['counts'] = 100*tab_bar['size'] / tab_bar.groupby("label")["size"].transform('sum')
    else:
        tab_bar['counts'] = tab_bar['size']
    
    sns.set(context='notebook', style='whitegrid', font_scale=font_scale)
    g = sns.catplot(data=tab_bar, kind="bar", col="label", y="counts", x="Bead_Class_Cluster", height=height, palette=palette, col_wrap=col_wrap, errwidth=0)

    num_classes = len(pd.unique(tab_bar['Bead_Class_Cluster']))
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    for ax in g.axes.flat:
        if normalize:
            ax.set_ylabel('Bead Count (%)')
            ax.axhline(y=100/num_classes, color='gray', linestyle=(0,(3,5)), linewidth=0.5)
            ax.bar_label(ax.containers[0],fmt="%.1f",label_type='edge', padding=1, fontsize=7)

        else:
            ax.set_ylabel('Raw Bead Count')
            ax.bar_label(ax.containers[0],fmt="%d",label_type='edge', padding=1, fontsize=7)
        ax.grid(False)
        plt.xlabel("Bead Class", labelpad=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right', rotation_mode="anchor")
        plt.setp(ax.get_xticklabels(), visible=True, rotation=45)
        ax.tick_params(labelbottom=True)
    g.fig.tight_layout()
    plt.show()
    if save and normalize:
        g.savefig(f"{plotsdir}/bar_norm.png")

    elif save and not normalize:
        g.savefig(f"{plotsdir}/bar_raw.png")

    if not show:
        plt.close()


def calculate_thresholds(data, control_classes, mult_outlier, mult_dev, fold_minimum=1):
    """
    Calculates per lane thresholds for positive beads based on control classes
    :param data: Input data frame
    :param control_classes: A list of class names representing control bead classes
    :param mult_outlier: Multiplier for the IQR used to determine outliers.
    :param mult_dev: Multiplier for the standard deviation to determine positivity.
    :param fold_minimum: Multiplier for the control class donut fluorescence to set a minimum bound for the threshold
    :return: Dictionary mapping sample names to threshold values. Also stores the multipliers used.
    """
    out = {"STAT_OUTLIERMULT": mult_outlier, "STAT_DEVMULT": mult_dev}
    samples = pd.unique(data["Chip_Date_Time_Lane"])
    c = 0
    for sample in samples:
        subset = data[data["Chip_Date_Time_Lane"] == sample]
        # Fetch control beads from the lane
        controls = subset[subset[bead_class_col].isin(control_classes)]
        # Get median blue intensity
        stats = controls["Donut_Intensity_MedianIntensity_CropBlue"].dropna()
        if not len(stats):
            threshold = 1
        else:
            # Take IQR, multiply by a custom factor and exclude outliers
            q25, q75 = np.percentile(stats, [25, 75])
            iqr = (q75 - q25) * mult_outlier
            filtered = stats[stats.between(q25 - iqr, q75 + iqr)]
            # Calculate threshold based on median + std * configurable multiplier
            threshold = filtered.median() + (filtered.std() * mult_dev)
            #ensure thresholds are a minimum fold change over the control class fluoresence - RS
            minimum_threshold = filtered.median()*fold_minimum
            if threshold<minimum_threshold:
                threshold = minimum_threshold
            c += 1
        out[sample] = threshold
    print(f"Calculated thresholds for {c} of {len(samples)} samples")
    return out


def generate_stats(data, classes, splitter, threshold=None, stat="Median", summaryfile=None, extra_cols=None):
    """
    # Generates a data frame containing summary statistics.
    :param data: Primary dataframe
    :param classes: List of bead classes.
    :param splitter: Column to use to group samples together.
    :param threshold: Dictionary mapping samples to thresholds to consider an object as positive.
    :param stat: Statistic for intensity averaging. Case-sensitive. Typically "Mean" or "Median".
    :param summaryfile: Optional, file name to save resulting table to.
    :param extra_cols: Allows you to supply additional metadata columns that will be included in the resulting table.
    :return:
    """
    # Generate a summary stats data frame
    colnames = ["ImageNumber",
                "FileName_Blue",
                "Chip_Date_Time_Lane",
                "Sample",
                "SampleShort",
                "Date",
                "Chip",
                "Timepoint",
                "Lane",
                "Threshold",
                "Total_Beads",
                ]
    if extra_cols:
        colnames += extra_cols
    colnames += ["Count_" + beadclass for beadclass in classes]
    colnames += ["Positive_" + beadclass for beadclass in classes]
    summary_data = data.reindex(columns=colnames)
    summary_data.drop_duplicates(inplace=True, ignore_index=True)

    # Generate summary statistics and fill table

    for sample in summary_data[splitter]:
        subset = data[data[splitter] == sample]
        a, b = np.unique(subset[bead_class_col], return_counts=True)
        bead_dictionary = dict(zip(a, b))
        th = threshold.get(sample, 1)

        summary_data.loc[summary_data[splitter] == sample, "Threshold"] = th
        summary_data.loc[summary_data[splitter] == sample, "Total_Beads"] = len(subset)

        for beadid in classes:
            if beadid not in bead_dictionary:
                bead_dictionary[beadid] = 0
            pos = np.count_nonzero(
                subset[subset[bead_class_col] == beadid][f"Donut_Intensity_{stat}Intensity_CropBlue"] > th)
            summary_data.loc[summary_data[splitter] == sample, f"Count_{beadid}"] = bead_dictionary[beadid]
            summary_data.loc[summary_data[splitter] == sample, f"Positive_{beadid}"] = pos

    if summaryfile is not None:
        summary_data.to_csv(summaryfile)
        print(f"Wrote {summaryfile}")
    else:
        print("Generated summary dataframe")
    return summary_data


def split_summary_data(data, classes, splitter, repr_method="median", outfile=None):
    """
    Generates summary data on a per-class basis
    :param data: Primary data frame
    :param classes: List of classes to split by
    :param splitter: Column used to identify unique samples
    :param repr_method: Method used for averaging mean and median intensity measurements. Supports 'mean' and 'median'.
    :param outfile: Path for table export, leave as "None" for no export.
    :return:
    """
    newcolumns = [splitter, bead_class_col, 'Count_Beads', "Mean_Blue_Intensity", "Median_Blue_Intensity"]

    output_df = pd.DataFrame(columns=newcolumns)

    for sample in data[splitter].unique():
        subset = data[data[splitter] == sample]

        for beadid in classes:
            bead_subset = subset[subset[bead_class_col] == beadid]
            buffer = [sample, beadid]
            intensities = []

            count = len(bead_subset.index)
            if count > 0:
                if repr_method == "median":
                    meanint = bead_subset["Donut_Intensity_MeanIntensity_CropBlue"].median()
                    medianint = bead_subset["Donut_Intensity_MedianIntensity_CropBlue"].median()
                elif repr_method == "mean":
                    meanint = bead_subset["Donut_Intensity_MeanIntensity_CropBlue"].mean()
                    medianint = bead_subset["Donut_Intensity_MedianIntensity_CropBlue"].mean()
                else:
                    raise ValueError("Unrecognised averaging method")
            else:
                meanint = 0
                medianint = 0
            intensities.append(meanint)
            if count != 0:
                buffer.append(count)
                buffer.append(meanint)
                buffer.append(medianint)
                output_df.loc[len(output_df)] = buffer

    # Save the resulting summary table
    if outfile:
        output_df.to_csv(outfile)
        print(f"Wrote {outfile}")
    else:
        print("Generated class summary dataframe")
    return output_df


def generate_heatmap_table(summary_data, classes, splitter, min_beads=10, kind="Median", outfile=None):
    """
    # Generate data for heatmaps
    :param summary_data: Summary table
    :param classes: List of classes
    :param splitter: Column used to identify unique samples
    :param min_beads: Minimum number of beads needed to evaluate a sample.
    :param kind: Averaging method
    :param outfile: Optional path to save table to.
    :return:
    """
    assay_list = [splitter] + classes

    frame = summary_data.reindex(columns=assay_list).drop_duplicates(ignore_index=True)
    frame.sort_values(by=splitter, inplace=True)
    frame_count = frame.copy()

    for target in summary_data[splitter].unique():
        mini_df = summary_data[summary_data[splitter] == target]
        for _, data in mini_df.iterrows():
            classtgt = data[bead_class_col]
            count = data['Count_Beads']
            frame_count.loc[frame_count[splitter] == target, classtgt] = count
            if count >= min_beads:
                reading = data[f"{kind}_Blue_Intensity"]
                frame.loc[frame[splitter] == target, classtgt] = reading

    frame_count = frame_count.fillna(0).astype(int, errors='ignore')
    if outfile:
        frame.to_csv(outfile)
        print(f"Wrote {outfile}")
    else:
        print("Generated heatmap dataframes")
    return frame, frame_count


def create_heatmap(data, count_data, cmap=None, stat="Median", label_remap=None, min_beads=5, display_counts=True,
                   title="", outdir=None, minival=None, maxival=None, thresholds=None):
    """
    # Creates a heatmap
    :param data: Data to draw heatmap with
    :param count_data: Frame of the same shape as data, containing counts of each bead.
    :param cmap: Matplotlib colour map for display of values
    :param stat: Averaging method in use
    :param label_remap: Dictionary mapping sample names to readable labels.
    :param min_beads: Minimum bead count needed to draw colour.
    :param display_counts: Whether to annotate counts as text or not.
    :param title: Name for the plot
    :param outdir: Export directory for the plot ("None" won't save).
    :param minival: Minimum value for the scale
    :param maxival: Maximum value for the scale
    :return:
    """
    hmp_data = data.set_index('Chip_Date_Time_Lane').transpose()
    heats = hmp_data.to_numpy()
    hmp_count_data = count_data.set_index('Chip_Date_Time_Lane').transpose()
    counts = hmp_count_data.to_numpy()
    classes_used = hmp_data.index.to_list()
    samples = hmp_data.columns.to_list()

    fig, ax = plt.subplots(figsize=(max(len(samples) / 2.5, 3), len(classes_used)))
    im = ax.imshow(heats, cmap=cmap, vmin=minival, vmax=maxival)

    # Generate ticks
    ax.set_xticks(np.arange(len(samples)))
    ax.set_yticks(np.arange(len(classes_used)))
    # Add labels
    if label_remap:
        ax.set_xticklabels([label_remap[k] for k in samples])
    else:
        ax.set_xticklabels(samples)
    ax.set_yticklabels(classes_used)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="left",
             rotation_mode="anchor")

    if maxival is None:
        maxival = max(data.max().to_list()[1:])

    # Add count annotations
    if display_counts and not thresholds:
        maxval = maxival / 2
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                if heats[i, j] < maxval:
                    txtcolor = "black"
                elif counts[i, j] < min_beads:
                    txtcolor = "lightgrey"
                else:
                    txtcolor = "white"
                ax.text(j, i, counts[i, j], ha="center", va="center", color=txtcolor)
    elif thresholds is not None:
        th_data = hmp_data.copy()
        for column_name in th_data.columns.tolist():
            th_data[column_name] = th_data[column_name] > thresholds[column_name]
        th_data = th_data.to_numpy()
        for i in range(th_data.shape[0]):
            for j in range(th_data.shape[1]):
                if th_data[i, j]:
                    ax.text(j, i, "*", ha="center", va="center", color="black")

    ax.set_title(f"Heatmap - {title}")

    # Display and size tweaks
    plt.grid(False) #new
    im_ratio = counts.shape[0] / counts.shape[1]
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046 * im_ratio, pad=0.04)
    cbar.ax.set_ylabel(f"{stat} Reporter Intensity", rotation=-90, va="bottom")
    fig.tight_layout()

    if outdir:
        fig.savefig(f"{outdir}/heatmap_{title}.png", bbox_inches='tight')
        plt.close()


def generate_split_heatmaps(data, count_data, chips, timepoints=None, cmap=None, stat="Median", label_remap=None,
                            min_beads=5, display_counts=True, plotsdir=None, thresholds=None):
    """
    # Generates individual heatmaps split by condition
    """
    minival = min(data.min().to_list()[1:])
    maxival = max(data.max().to_list()[1:])
    if timepoints is None:
        timepoints = ['all']
    if plotsdir:
        plotsdir = os.path.join(plotsdir, "heatmaps")
        if not os.path.exists(plotsdir):
            os.makedirs(plotsdir, exist_ok=True)
    total = len(chips) * len(timepoints)
    with trange(total) as t:
        t.bar_format = "{desc} {bar} {n_fmt}/{total_fmt}"
        t.set_description(f'Generating job list')
        for chip in chips:
            hmp_data = data[data['Chip_Date_Time_Lane'].str.startswith(chip)]
            hmp_count_data = count_data[count_data['Chip_Date_Time_Lane'].str.startswith(chip)]
            for timepoint in timepoints:
                t.set_description(f'Working on {chip}, T{timepoint}')
                if timepoint == 'all':
                    title = chip
                    subset_hmp_data = hmp_data
                    subset_count_data = hmp_count_data
                else:
                    title = f"{chip} - T{timepoint}"
                    indexer = hmp_data['Chip_Date_Time_Lane'].str.slice(start=-5, stop=-3) == timepoint
                    subset_hmp_data = hmp_data[indexer]
                    if subset_hmp_data.shape[0] == 0:
                        # No samples at this timepoint
                        print(f"No data for {chip}, T{timepoint}, skipping plot generation")
                        t.update(1)
                        continue
                    subset_count_data = hmp_count_data[indexer]
                create_heatmap(subset_hmp_data, subset_count_data, cmap=cmap, stat=stat, label_remap=label_remap,
                               min_beads=min_beads, display_counts=display_counts, title=title, outdir=plotsdir,
                               minival=minival, maxival=maxival, thresholds=thresholds)
                t.update(1)


def fetch_misc(main_df):
    # Create a few objects needed by heatmaps
    # Define a custom colour sequence map
    cmap = LinearSegmentedColormap.from_list("redwhite", ["white", "#ffc8bf", "#ff6347", "darkred"])
    # Identify unique chips and timepoints
    chips = pd.unique(main_df['Chip'])
    timepoints = pd.unique(main_df['Timepoint'])
    return chips, timepoints, cmap
