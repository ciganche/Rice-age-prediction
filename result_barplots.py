import matplotlib
import matplotlib.pyplot as plt
import numpy as np


level = "Genus"
location = "outputs/result_plots/"

labels = ["RIDGE+RFE", "LASSO+RFE", "ENET+RFE",
          "RF+PCA", "RF+RF IMP", "RF+SPRMN", "RF+F-VAL",
          "SVM+PCA", "SVM+SVM IMP", "SVM+SPRMN", "SVM+F-VAL"]

# GENUS
test_mse = [285.34, 141.95, 142.49,
            174.63, 153.88, 169.29, 200.49,
            112.56, 160.19, 120.35, 108.41]

cv_mse = [193.67, 257.9, 174.71,
          154.92, 216.96, 190.94, 197.72,
          96.74, 175.63, 123.30, 121.78]

# FAMILIA
# test_mse = [190.60, 129.20, 85.82,
#             152.56, 122.39, 129.54, 152.20,
#             98.89, 104.90, 87.94, 89.46]
#
# cv_mse = [158.44, 191.80, 146.87,
#           137.27, 186.35, 157.25, 166.32,
#           85.06, 145.69, 111.83, 109.64]




# ORDO
# test_mse = [436.77, 160.96, 113.94,
#             198.06, 144.50, 160.12, 172.43,
#             105.17, 133.27, 101.49, 101.49]
#
# cv_mse = [207.81, 228.84, 169.08,
#           166.14, 179.74, 156.04, 162.38,
#           100.96, 172.11, 117.64, 115.66]



# CLASSIS
# test_mse = [322.04, 237.39, 181.57,
#            196.05, 192.77, 234.42, 223.66,
#            159.84, 231.74, 163.32, 156.11]
#
# cv_mse = [453.65, 280.27, 250.45,
#           254.90, 199.91, 183.98, 185.37,
#           164.23, 268.89, 167.46, 168.64]




# PHYLUM
# test_mse = [1845, 445.66, 450.37, 411.34,
#             491.32, 391.88, 339.25, 353.11,
#             263.25, 475.79, 261.05, 261.10]
#
# cv_mse = [0, 442.95, 361.11, 374.51,
#           414.94, 365.59, 219.90, 215.55,
#           221.35, 383.93, 217.31, 223.11]

x = np.arange(len(labels))  # the label locations
width_test = 0.35  # the width of the bars
width_cv = 0.15

fig, ax = plt.subplots(figsize=(19, 5))
rects1 = ax.bar(x - width_test/2, test_mse, width_test, label="Test MSE", align="center", )
rects2 = ax.bar(x + width_cv/2, cv_mse, width_cv, label="CV MSE", color="y", align="center")

ax.set_ylabel("Mean squared error")
ax.set_title(level + " level results")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel_left(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=((rect.get_x() + rect.get_width() / 2), height),
                    xytext=(-5, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom').set_fontsize(8)


def autolabel_right(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=((rect.get_x() + rect.get_width() / 2), height),
                    xytext=(5, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom').set_fontsize(8)


autolabel_left(rects1)
autolabel_right(rects2)


plt.savefig(location + level + ".png")
plt.show()
