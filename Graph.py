import statistics as st
import matplotlib.pyplot as plt


# l=[3,5,2,1,5,6,7,4,3,2]

# mode=st.mode(l)
# median=st.median(l)
# mean=st.mean(l)
# sd=st.stdev(l)
# var=st.variance(l)
# print(mean)
# print(mode)
# print(median)
# print(sd)
# print(var)

# :::::::::::::::LINE PLOT:::::::::::
''' 
x_days=[1,2,3,4,5]
y_prise=[9,9.5,10,11,15]
x_prise=[10,11,16,12,17]

plt.plot(x_days, y_prise, label='Y Stock')
plt.plot(x_days, x_prise, label='X Stock')

plt.title("stock prise ")
plt.xlabel('week days ')
plt.ylabel('prise in usd ')
plt.legend()
plt.show()
'''

# :::::::::::::::BAR PLOT:::::::::
'''
cities=['dto','Rnc','Hzb','Viz','Vab','Hyd']
temp=[30,40,50,20,15,50]

plt.xlabel('cities')
plt.ylabel('temp')
plt.title('Tempreture varience')

plt.bar(cities,temp)
plt.show()
'''

#::::::::::::::::HISTOGRAM ::::::::::::
'''
import matplotlib.pyplot as plt

location = r'C:\Users\mayan\Downloads\agedata.csv'
f = open(location, 'r')
agefile = f.readlines()
f.close()

age_list=[]

for record in agefile:
    age = int(record)
    age_list.append(age)

x_axis = [0,10,20,30,40,50,60,70,80,90,100]

plt.xlabel('Group')
plt.ylabel('Age')
plt.title('Age Histogram')
plt.hist(age_list, bins=x_axis, edgecolor='black', rwidth=0.9)
plt.show()
'''
# ::::::::::::::::BOX PLOT::::::::;
'''
import matplotlib.pyplot as plt

location=r'C:\Users\mayan\Downloads\salesdata.csv'
f=open(location,'r')
sales=f.readlines()
f.close()

sales_list=[]

for record in sales:
    sale=int(record)
    sales_list.append(sale)

plt.title('Box Plot')
plt.boxplot(sales_list)
plt.show()
'''
#:::::::::::::::::PIE CHART:::::::::::::::::
'''
import matplotlib.pyplot as plt

location= r'C:\Users\mayan\Downloads\agedata2.csv'
f=open(location,'r')
age_data=f.readlines()
f.close()

city_list=[]

for record in age_data:
    age,city=record.split(sep=',')
    city_list.append(city)

from collections import Counter
city_count=Counter(city_list)

city_name=list(city_count.keys())
city_values=list(city_count.values())
print(city_values)
print(city_name)

plt.pie(city_values,labels=city_name,autopct='%2f%%')
plt.show()
'''

#:::::::::::::::::SCATTER PLOT::::::::::::::::
'''
import matplotlib.pyplot as plt

location= r'C:\Users\mayan\Downloads\salesdata2.csv'
f=open(location,'r')
sales=f.readlines()
f.close()

s_list=[]
c_list=[]


for record in sales:
    sale,cost=record.split(sep=',')
    s_list.append(int(sale))
    c_list.append(int(cost))

plt.title('Scatter plot')
plt.xlabel('Salel')
plt.ylabel('Cost')

plt.scatter(s_list,c_list)
plt.show()
'''

#:::::::::::::::::: DIFFERENT FIGURE PLOT::::::
'''
import matplotlib.pyplot as plt

location= r'C:\Users\mayan\Downloads\salesdata2.csv'
f=open(location,'r')
sales=f.readlines()
f.close()

s_list=[]
c_list=[]


for record in sales:
    sale,cost=record.split(sep=',')
    sale_list=[]
    s_list.append(int(sale))
    c_list.append(int(cost))

sale_list.append(s_list)
sale_list.append(c_list)

#plot the scatter plot
plt.figure('My scatter plot')
plt.title('Scatter plot')
plt.xlabel('Salel')
plt.ylabel('Cost')
plt.scatter(s_list,c_list)

#plot the boxplot
plt.figure('My box plot')
plt.title('Boxplot plot')
plt.xlabel('Salel')
plt.ylabel('Cost')
plt.boxplot(sale_list)
plt.show()
'''

#::::::::::::::::::: ALL figure in single grid::::::
'''
import matplotlib.pyplot as plt

location= r'C:\Users\mayan\Downloads\salesdata2.csv'
f=open(location,'r')
sales=f.readlines()
f.close()

location = r'C:\Users\mayan\Downloads\agedata.csv'
t = open(location, 'r')
agefile = t.readlines()
t.close()

s_list=[]
c_list=[]

for record in sales:
    sale,cost=record.split(sep=',')
    sale_list=[]
    s_list.append(int(sale))
    c_list.append(int(cost))

sale_list.append(s_list)
sale_list.append(c_list)

age_list=[]

for record in agefile:
    age = int(record)
    age_list.append(age)

x_axis = [0,10,20,30,40,50,60,70,80,90,100]


#plot the scatter plot
plt.subplot(2, 3, 1)
plt.title('Scatter plot')
plt.xlabel('Salel')
plt.ylabel('Cost')
plt.scatter(s_list,c_list,marker='o',s=30,c='green')

#plot the boxplot
plt.subplot(2, 3, 2)
plt.title('Boxplot plot')
plt.xlabel('Salel')
plt.ylabel('Cost')
plt.boxplot(sale_list,
            patch_artist=True,
            boxprops=dict(facecolor='r', color='b',linewidth=2),
            whiskerprops=dict(color='g',linewidth=2),
            medianprops=dict(color='g',linewidth=3),
            flierprops=dict(markerfacecolor='r',marker='o',markersize=7)
            )


#plot the histogram
plt.subplot(2, 3, 3)
plt.xlabel('Group')
plt.ylabel('Age')
plt.title('Age Histogram')
plt.hist(age_list, bins=x_axis, edgecolor='black', rwidth=0.9,color='c')

# Bar plot
days=['sun','mon','tue','wed','thu','fri','sat']
working_hour=[20,40,16,35,25,29,20]

plt.subplot(2, 3, 4)
plt.xlabel('DAYS')
plt.ylabel('Hour')
plt.title('WEEKLY Working hour')
plt.bar(days,working_hour,color=['r','g','b','y','pink','orange','g'])


#plot Line plot

plt.subplot(2, 3, 5)
plt.title("stock prise ")
plt.xlabel('week days ')
plt.ylabel('prise in usd ')

x_days=[1,2,3,4,5]
y_prise=[9,9.5,10,11,15]
x_prise=[10,11,16,12,17]

plt.plot(x_days, y_prise, label='Y Stock',color='green',marker='o',linestyle='--')
plt.plot(x_days, x_prise, label='X Stock',color='red',marker='o',linestyle='--')
plt.legend()

#PIE CHART
categories = ['Category A', 'Category B', 'Category C', 'Category D']

sizes = [15, 30, 45, 10]  # Example data
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)  # explode the 1st slice (Category A)

plt.subplot(2, 3, 6)
plt.title('Pie Chart')
plt.pie(sizes, explode=explode, labels=categories, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
# Overlapping_layout
plt.tight_layout()

#Show the graph
plt.show()



'''





