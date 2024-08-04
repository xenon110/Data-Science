
cittytemp_path = r'C:\Users\mayan\Downloads\citytemp.csv'
citytemp = open(cittytemp_path, 'r')

rec1=citytemp.readline()
city,tempreture,unit = rec1.split(',')

prev_city=city
citytemp.seek(0)

tempsum=0.0
count=0
avgtemp=0.0

for record in citytemp:
    record = record.rstrip('\n')
    city, tempreture, unit = record.split(',')


    if unit=="C":
        tempreture = (float(tempreture) *9/5 + 32)
    if city!=prev_city:
         avgtemp = tempsum/count
         print(prev_city + " " + str(round(avgtemp,2)))
         prev_city=city
         tempsum=0.0
         count=0
         avgtemp=0.0
    tempsum=tempsum+float(tempreture)
    count=count+1

else:
  avgtemp=tempsum/count
  print(prev_city + " " + str(round(avgtemp,2)))



