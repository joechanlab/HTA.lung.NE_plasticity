import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load exported data
nodes = pd.read_csv("phylogeny_nodes.csv", index_col=0)
cells = pd.read_csv("cell_assignments.csv", index_col=0)
edges = pd.read_csv("phylogeny_edges.csv", index_col=0)

# Create the figure and draw edges (your existing code)
fig, ax = plt.subplots(figsize=(12, 12))

# Draw Edges
for _, edge in edges.iterrows():
    ax.plot([edge.x_from, edge.x_to], [edge.y_from, edge.y_to], 
            color='grey', alpha=0.3, zorder=1)

# --- NEW: Add Leaf Node Dots ---
# Filter for leaf nodes (where leaf == True)
leaf_nodes = nodes[nodes['leaf'] == True].copy()

# Assuming the 'name' in nodes matches the 'cell' column in your assignments
# Merge to get the cell_type for each leaf node
COLOR = "cell_type"
if COLOR == 'clone':
    unique_clones = sorted(cells['clone'].unique())
    palette = sns.color_palette("husl", len(unique_clones)).as_hex()
    color_map = dict(zip(unique_clones, palette))
elif COLOR == "cell_type":
    color_map = {
        'SCLC-A': '#d62728',    # Blue
        'SCLC-N': '#17becf',    # Orange
        "KRT5/6A+ NonNE SCLC": "#800080",  # purple
        "AGR2+ NonNE SCLC": "#9370db",  # mediumpurple
        'NSCLC': '#ffd700'       # Red
    }   

leaf_nodes = leaf_nodes.merge(cells[['cell', COLOR]], 
                              left_on='name', right_on='cell', how='left')

# Plot the dots
for ctype, color in color_map.items():
    subset = leaf_nodes[leaf_nodes[COLOR] == ctype]
    ax.scatter(subset['x'], subset['y'], 
               c=color, label=ctype, s=10, zorder=4, alpha=0.8)

root_node = nodes[nodes['name'] == 'Node1']

if not root_node.empty:
    ax.scatter(root_node['x'], root_node['y'], 
               color='black', 
               marker='*', 
               s=200,          # Larger size to stand out
               label='Root', 
               zorder=5)       # Highest zorder to stay on top
    
    # Optional: Add a text label
    ax.text(root_node['x'].values[0], root_node['y'].values[0] + 0.5, 
            'Root', fontsize=12, fontweight='bold', ha='center')
    
# Formatting
ax.set_aspect('equal')
ax.legend(title=COLOR.replace("_", ""), loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("RU151 PDX")
plt.show()

# compare two versions of the RU151
RU151_new_clone = pd.read_csv("/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat/RU151/numbat_output_FACETS_HISENS/RU151_PDX/clone_post_2.tsv", sep='\t')
RU151_old_clone = pd.read_csv("/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat/RU151_PDX/numbat_output_FACETS_HISENS/RU151_PDX/clone_post_2.tsv", sep='\t')
clone_new = RU151_new_clone[['cell', 'clone_opt']]
clone_old = RU151_old_clone[['cell', 'clone_opt']]
clone_new.columns = ['cell', 'clone_new']
clone_old.columns = ['cell', 'clone_old']
clone_old.loc[:,'cell'] = clone_old.cell.apply(lambda x: 'RU151_PDX_' + x)
pd.crosstab(clone_new.clone_new, clone_old.clone_old)



# 1. Store your existing crosstab calculation in a variable
ct = pd.crosstab(clone_new.clone_new, clone_old.clone_old)

# 2. Plot as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')

plt.title('Comparison of New vs Old Clone Assignments')
plt.xlabel('Old Clones')
plt.ylabel('New Clones')
plt.show()

