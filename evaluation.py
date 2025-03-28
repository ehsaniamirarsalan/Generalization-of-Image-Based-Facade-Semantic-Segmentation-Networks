import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# The original confusion matrix by Mask R-CNN
# original_cm = np.array([
#     [3665, 0, 56, 1135, 0, 35, 3, 86, 154, 0, 16, 506, 0],
#     [5710613, 787419, 1069631, 35042, 0, 8229, 11468, 12733, 461, 9765, 172100, 11906, 25584],
#     [11616253, 168439, 3928356, 299607, 0, 173087, 148630, 105743, 37505, 25221, 576758, 263686, 161093],
#     [695998, 3512, 143484, 5127473, 0, 22596, 26781, 14864, 14272, 7804, 24483, 2607, 15261],
#     [482006, 790, 85972, 86853, 0, 496, 195, 6397, 0, 2215, 229, 608, 61247],
#     [517252, 724, 47424, 19829, 0, 1233093, 14935, 3497, 340, 4389, 54032, 6228, 253],
#     [238001, 1265, 68437, 34618, 0, 13979, 738355, 4765, 214, 3010, 64434, 784, 0],
#     [605637, 1106, 46014, 47743, 0, 5898, 38643, 975316, 449, 334, 65820, 2027, 4170],
#     [102239, 0, 14606, 126099, 0, 2074, 2981, 2797, 174623, 0, 263, 122, 1882],
#     [460918, 8, 34315, 17400, 0, 54186, 25378, 675, 0, 134445, 96843, 12425, 30070],
#     [1638513, 17433, 334944, 11580, 0, 26179, 51052, 35539, 406, 11715, 3084622, 12143, 34505],
#     [476952, 494, 91179, 10831, 0, 16673, 955, 3959, 1126, 20279, 40122, 419944, 210],
#     [568924, 8270, 73791, 12921, 0, 6154, 1350, 2984, 925, 1238, 19788, 22, 865143]]
# )

##################################
# The original confusion matrix by DeepLabV3+
# original_cm = np.array([
#     [       0,        0,        0,        0,        0,        0,        0,        0,        0,        0,        0,        0,        0],
#     [   37090, 28570310,  7167305,   115863,       19,    11295,     3440,     6953,    2163,    12201,  1617247,    28165,     6461],
#     [   58847, 15328935, 71864710,  1412616,      255,   225508,    31215,    63704,    88169,   109403,  4261794,   277032,    68462],
#     [   22587,  1570102,  6897905, 23387900,     3600,    31233,    14368,    24869,   297459,    22352,   497628,    15146,    10497],
#     [   15337,   633106,  2123381,   334200,    23662,        0,       31,      562,     3470,     1983,    56600,        0,    37199],
#     [       0,   365971,  5192995,   141095,        0,  3353109,    34483,    16984,     4567,    34875,   533780,     9793,      473],
#     [      82,   439555,  3813601,   342395,        0,    28144,   719852,     1831,    12049,    14790,  1372624,     2735,       23],
#     [     724,   694135,  3989992,   632559,        0,    11272,    42927,  2760347,    46893,    69989,  1417561,    10361,     1952],
#     [     398,    58007,  1881948,   774048,      165,      535,      188,     3164,   883779,      410,    79748,      365,     3326],
#     [     266,   461422,  2736807,   139925,       32,    24261,    21093,     6325,      576,   515777,  1061793,    39386,     2159],
#     [    3797,  2971565,  5956634,    66940,        3,    32207,    10290,    27206,      941,    54092, 18809561,    17985,     6426],
#     [    1055,   443745,  3847809,    38520,        0,    18350,     1123,     7828,     2261,    17534,   404398,   969618,     1206],
#     [   17606,   699664,  5091712,   590948,     6316,      255,      186,     4760,     9933,    31294,   384516,     1250,  1110504]
# ])
##################################
# The original confusion matrix by U-Net
original_cm = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 24804198, 10509470, 708729, 20901, 66685, 39171, 39707, 54096, 33829, 1260557, 13183, 27986],
    [0, 17022281, 67319905, 2939175, 234040, 298613, 159534, 93646, 562197, 114218, 4759751, 167634, 119656],
    [0, 789200, 12240991, 16171033, 272895, 599131, 416901, 171233, 1023734, 14309, 1029949, 55642, 10628],
    [0, 469165, 1782011, 471225, 387094, 5577, 4238, 14620, 11972, 1939, 64935, 3027, 13728],
    [0, 274126, 5644861, 524365, 6971, 681733, 10890, 44350, 48884, 445165, 1980464, 22835, 3481],
    [0, 375942, 3553638, 406613, 7080, 64361, 327207, 19363, 38537, 56435, 1894438, 1549, 2518],
    [0, 396480, 4994898, 1514593, 12239, 41663, 45059, 383424, 69443, 95611, 2105070, 2868, 17364],
    [0, 85621, 1800124, 1139910, 81747, 38642, 56333, 27369, 373279, 1519, 76420, 1741, 3376],
    [0, 295146, 3220439, 569612, 12848, 53232, 16153, 36365, 29304, 213256, 539899, 21114, 2454],
    [0, 3636749, 10883087, 1097302, 11476, 82920, 30302, 99027, 138103, 198977, 11690929, 25975, 62800],
    [0, 632369, 4329325, 242532, 2888, 45495, 10940, 4319, 18265, 40687, 324327, 101288, 1012],
    [0, 590311, 4815110, 1399457, 180969, 15430, 14450, 86090, 80376, 9545, 569471, 6446, 181289]]
)
##################################

# Combine the first two rows (class 0 and 1) and first two columns
new_cm = np.zeros((12, 12))

# First, combine the first two rows and columns properly
# For the top-left cell (combined background classes)
new_cm[0, 0] = original_cm[0, 0] + original_cm[0, 1] + original_cm[1, 0] + original_cm[1, 1]

# For the first row (combined background predicted as other classes)
new_cm[0, 1:] = original_cm[0, 2:] + original_cm[1, 2:]

# For the first column (other classes predicted as combined background)
new_cm[1:, 0] = original_cm[2:, 0] + original_cm[2:, 1]

# For the rest of the matrix
new_cm[1:, 1:] = original_cm[2:, 2:]

# Create facade class labels
class_names = [
    "Background",
    "facade", "window", "door", "cornice",
    "sill", "balcony", "blind", "deco",
    "molding", "pillar", "shop"
]
# Function to create and display the confusion matrix
def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues,
                          figsize=(18, 16)):
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with zero
        plot_data = cm_norm
        fmt = '.2f'
    else:
        plot_data = cm
        fmt = 'g'

    plt.figure(figsize=figsize)
    ax = sns.heatmap(plot_data, annot=True, fmt=fmt, annot_kws={"size": 10},
                     cmap=cmap, xticklabels=class_names, yticklabels=class_names)

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()

    return plt


# Calculate all the requested metrics
def calculate_metrics(cm):
    n_classes = cm.shape[0]

    # Pixel Accuracy per class (recall)
    pixel_accuracy_per_class = np.zeros(n_classes)
    for i in range(n_classes):
        if np.sum(cm[i, :]) > 0:
            pixel_accuracy_per_class[i] = cm[i, i] / np.sum(cm[i, :])

    # Overall Pixel Accuracy
    pixel_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    # IoU (Jaccard index) for each class
    iou = np.zeros(n_classes)
    for i in range(n_classes):
        # True positives
        tp = cm[i, i]
        # False positives + false negatives
        fp_fn = np.sum(cm[i, :]) + np.sum(cm[:, i]) - tp
        if tp + fp_fn > 0:
            iou[i] = tp / (tp + fp_fn)

    # Mean IoU
    mean_iou = np.mean(iou)

    # Precision per class
    precision = np.zeros(n_classes)
    for i in range(n_classes):
        if np.sum(cm[:, i]) > 0:
            precision[i] = cm[i, i] / np.sum(cm[:, i])

    # Recall per class (same as pixel accuracy per class)
    recall = pixel_accuracy_per_class

    # F1 score per class
    f1 = np.zeros(n_classes)
    for i in range(n_classes):
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    # Mean F1 score
    mean_f1 = np.mean(f1)

    return {
        'pixel_accuracy': pixel_accuracy,
        'pixel_accuracy_per_class': pixel_accuracy_per_class,
        'iou': iou,
        'mean_iou': mean_iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_f1': mean_f1,
    }


# Plot raw counts
plt.figure(1)
plot_raw = plot_confusion_matrix(new_cm, class_names, normalize=False,
                                 title='U-Net Confusion Matrix (Raw Counts)')
plt.savefig('confusion_matrix_raw.png', dpi=300, bbox_inches='tight')

# Plot normalized (by row)
plt.figure(2)
plot_norm = plot_confusion_matrix(new_cm, class_names, normalize=True,
                                  title='U-Net Confusion Matrix (Normalized)')
plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')

# Log scale visualization for very large numbers - FIXED VERSION
plt.figure(3, figsize=(18, 16))
log_cm = np.log1p(new_cm)  # log(1+x) to handle zeros
heatmap = sns.heatmap(log_cm, annot=False, cmap='viridis',
                      xticklabels=class_names, yticklabels=class_names)
plt.setp(heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
plt.setp(heatmap.get_yticklabels(), fontsize=12)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.title('Confusion Matrix (Log Scale)', fontsize=16)
cbar = heatmap.collections[0].colorbar
cbar.set_label('log(1+value)', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_log.png', dpi=300, bbox_inches='tight')

# Calculate metrics
metrics = calculate_metrics(new_cm)

# Print the results
print("Confusion Matrix Shape:", new_cm.shape)
print(f"\nOverall Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
print(f"Mean IoU: {metrics['mean_iou']:.4f}")
print(f"Mean F1 Score: {metrics['mean_f1']:.4f}")

# Print per-class metrics
print("\nPer-class Metrics:")
headers = ['Class', 'Pixel Acc', 'IoU', 'Precision', 'Recall', 'F1', 'Z-score']
print(
    f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10} {headers[6]:<10}")
print("-" * 80)

for i, name in enumerate(class_names):
    print(f"{name:<20} "
          f"{metrics['pixel_accuracy_per_class'][i]:.4f}     "
          f"{metrics['iou'][i]:.4f}     "
          f"{metrics['precision'][i]:.4f}     "
          f"{metrics['recall'][i]:.4f}     "
          f"{metrics['f1'][i]:.4f}     "
          )

# Visualize the per-class metrics in bar charts
metrics_to_plot = [
    ('Pixel Accuracy per Class', metrics['pixel_accuracy_per_class']),
    ('IoU (Jaccard Index) per Class', metrics['iou']),
    ('F1 Score per Class', metrics['f1'])

]

plt.figure(figsize=(24, 20))
for i, (title, data) in enumerate(metrics_to_plot):
    plt.subplot(2, 2, i + 1)
    bars = plt.bar(range(len(class_names)), data)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right', fontsize=12)
    plt.title(title, fontsize=16)
    plt.ylim(0, max(1.1, np.max(data) * 1.1))  # Adjust y-limit for Z-scores which can be > 1

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig('segmentation_metrics.png', dpi=300, bbox_inches='tight')

# Each cell (i,j) shows the IoU between class i and class j
n_classes = len(class_names)
iou_matrix = np.zeros((n_classes, n_classes))
for i in range(n_classes):
    for j in range(n_classes):
        # Intersection
        intersection = new_cm[i, j]
        # Union
        union = np.sum(new_cm[i, :]) + np.sum(new_cm[:, j]) - new_cm[i, j]
        if union > 0:
            iou_matrix[i, j] = intersection / union

plt.figure(figsize=(18, 16))
ax = sns.heatmap(iou_matrix, annot=True, fmt='.2f', annot_kws={"size": 10},
                 cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.ylabel('True Class', fontsize=14)
plt.xlabel('Predicted Class', fontsize=14)
plt.title('IoU (Jaccard Index) Matrix Between Classes', fontsize=16)
plt.tight_layout()
plt.savefig('iou_matrix.png', dpi=300, bbox_inches='tight')

plt.show()