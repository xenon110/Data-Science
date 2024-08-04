# list1=[["mohan","sohan","rupesh"],["eating","dancing","sleeping"]]
# lis=[]
# for i in list1:
#     lis.append(i[1:3])
# print(lis)


# Use raw strings or double backslashes for file paths
cittytemp_path = r'C:\Users\mayan\Downloads\citytemp.csv'
cittytemp = open(cittytemp_path, 'r')
cittywrite = open(cittytemp_path, 'a')
rec1 = cittytemp.readline()
rec2 = cittytemp.readlines()
print("First line: \n", rec1)    # for to print the only first line
# print("First line: \n", rec2)    # for to print all line

#data processing

city,tempreture,unit = rec1.split(',')
tempreture= (float(tempreture)-32)*5/9
cittywrite.write(rec1)
cittywrite.close()

cittytemp.seek(0)
for record in cittytemp:
    record=record.rstrip('\n')
    print(record)