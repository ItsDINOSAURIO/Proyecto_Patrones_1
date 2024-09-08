import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Práctica 1. Si realmente se pueden separar machos de hembras, cual característica = culmen depth, solamente dos clases 
train_data=pd.read_csv(r'D:\\Upiita\6to\Reconocimiento de Patrones\penguins_training.csv')

sex={'MALE':'green', 'FEMALE':'magenta'}

#clase_1
val=train_data[train_data['Sex']=='MALE'] 
car=[val['Culmen Length (mm)'],val['Culmen Depth (mm)'],val['Flipper Length (mm)']]
clase_obj1=np.mean(car,1) 

#clase_2
val=train_data[train_data['Sex']=='FEMALE'] 
car=[val['Culmen Length (mm)'],val['Culmen Depth (mm)'],val['Flipper Length (mm)']]
clase_obj2=np.mean(car,1)

salida=[]

test_data=pd.read_csv(r'D:\\Upiita\6to\Reconocimiento de Patrones\penguins_testing.csv')
for sex,color in sex.items():
    val=test_data[test_data['Sex']==sex]
    for i, fila in val.iterrows():
        Clase_inicial=fila['Sex']
        dato = [fila['Culmen Length (mm)'],fila['Culmen Depth (mm)'],fila['Flipper Length (mm)']]
        dist1= np.sqrt((dato[0]-clase_obj1[0])**2+(dato[1]-clase_obj1[1])**2+(dato[2]-clase_obj1[2])**2)
        dist2= np.sqrt((dato[0]-clase_obj2[0])**2+(dato[1]-clase_obj2[1])**2+(dato[2]-clase_obj2[2])**2)
        Distancia=np.array([dist1,dist2])
        minimo=np.argmin(Distancia)
        valor_minimo=Distancia[minimo]
        if minimo ==0 and valor_minimo<50:
            Clase_otorgada='MALE'
        elif minimo ==1 and valor_minimo<50:
            Clase_otorgada='FEMALE'
        else: Clase_otorgada='Desconocida'

        salida.append([i+1,Clase_inicial,Clase_otorgada,valor_minimo])

excel = pd.DataFrame(salida, columns=['Muestra', 'Clase inicial', 'Clase otorgada', 'Distancia euclidiana mínima'])
print(excel)
#excel.to_excel('resultados_penguins_sex.xlsx')

'''
fig = plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')
fig1 = plt.figure(figsize=(10,8))
ax1=fig1.add_subplot(111,projection='3d')
species={'Adelie Penguin (Pygoscelis adeliae)':'blue',
         'Chinstrap penguin (Pygoscelis antarctica)':'green',
         'Gentoo penguin (Pygoscelis papua)':'red'}
for sex,color in sex.items():
    subset=train_data[train_data['Sex']==sex]
    ax.scatter(subset['Culmen Length (mm)'],subset['Culmen Depth (mm)'], subset['Flipper Length (mm)'],color=color,label=sex,alpha=0.6)

for specie,color in species.items():
    subset=train_data[train_data['Species']==specie]
    ax1.scatter(subset['Culmen Length (mm)'],subset['Culmen Depth (mm)'], subset['Flipper Length (mm)'],color=color,label=specie,alpha=0.6)

ax.set_xlabel('Culmen Length(mm)')
ax.set_ylabel('Culmen Depth(mm)')
ax.set_zlabel('Flipper Length (mm)')
plt.title('Largo vs Alto del Pico vs Largo de la aleta')
plt.legend()
plt.grid(True)

ax1.set_xlabel('Culmen Length(mm)')
ax1.set_ylabel('Culmen Depth(mm)')
ax1.set_zlabel('Flipper Length (mm)')
plt.title('Largo vs Alto del Pico vs Largo de la aleta')
plt.legend()
plt.grid(True)

plt.show()
'''
##Hacer excel con: Muestra | Clase inicial(del excel test) | Clase otorgada (Por el programa) | distancia euclidiana min
