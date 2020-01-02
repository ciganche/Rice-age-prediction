import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def edit_name(name):
    name_elements = name.split(";")
    new_name = ""
    for i in range(0,len(name_elements)):
        element = name_elements[i]
        if "uncultured" in element or "metagenome" in element:
            element = "D_"+str(i)+"__uncultured"
        new_name= new_name + element + ";"
    return  new_name


number_to_print = 15

location="outputs/elastic_net/"
name = "feature_importance.txt"
microbes = pd.read_csv(location+name,delimiter=',',encoding='utf-8', header=None)
microbes = microbes.drop(microbes.columns[2], axis=1)
selected_microbes = microbes.loc[0:number_to_print,:]

new_names = map(edit_name, selected_microbes.loc[:,0])
prints = pd.Series(selected_microbes.loc[:,1].to_numpy(), index=new_names)
prints = prints.iloc[::-1]
marker = prints > 0

matplotlib.rcParams['figure.figsize'] = (20.0, 11.0)
prints.plot(kind='barh', color=marker.map({True: 'g', False: 'r'}))
plt.grid()
plt.title("Coefficients in the Elastic Net")
plt.savefig(location + "feature_importance.png")
plt.show()
