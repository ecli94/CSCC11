import numpy as np
import pandas as pd
from scipy.stats import skewnorm
import matplotlib.pyplot as mpl

def round_to_half(n):
	if n <= 0:
		return 0.5
	return int(n*2)/2
	
def is_together(members, df):
	for i in range (0, len(members)-1):
		m1 = df.loc[(member[i])+1]
		m2 = df.loc[(member[i+1])+1]
		if m1['FamilySize'] == 1 | m2['FamilySize'] == 1:
			return False
		if m1['FamilySize'] != m2['FamilySize'] & \
			m1['LastName'] != m2['LastName']:
			return False
	return True	

df = pd.read_csv('csv/train.csv', header=0)
df['Gender'] = df['Sex'].map({ 'female': 0, 'male': 1}).astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['BoardOrder'] = df['Embarked'].fillna(0).map ({ 'S': 1, 'C': 2, 'Q': 3, 0: 3 })

mean_ages = np.zeros((2,3))
std_ages = np.zeros((2,3))
skew_ages = np.zeros((2,3))

for i in range (0,2):
	for j in range (0,3):
		mean_ages[i][j] = df[(df['Gender'] == i) & \
		(df['Pclass'] == j+1)]['Age'].dropna().mean()
		std_ages[i][j] = df[(df['Gender'] == i) & \
		(df['Pclass'] == j+1)]['Age'].dropna().std()
		skew_ages[i][j] = df[(df['Gender'] == i) & \
		(df['Pclass'] == j+1)]['Age'].dropna().skew()
		
df['AgeFill'] = df['Age']
for i in range (0,2):
	for j in range (0,3):
		for index, value in df[(df['Age'].isnull()) & (df['Gender'] == i) & \
			(df['Pclass'] == j+1)]['PassengerId'].iteritems():
			# uses a skewed normal distribution to produce random values for age
			val = round_to_half(skewnorm.rvs(skew_ages[i][j], \
				loc=mean_ages[i][j], scale=std_ages[i][j]))
			df.loc[(index),'AgeFill'] = val

			
entities = []
for i in list(df['PassengerId'].groupby(df['Ticket'])):
	entities.append(str(i[1]).split("\n"))

for i in range (0, len(entities)):
	entities[i].remove(entities[i][-1])
	for j in range (0, len(entities[i])):
		entities[i][j] = entities[i][j].split("    ")
		entities[i][j].remove(entities[i][j][0])
	temp = entities[i]
	entities[i] = []
	for k in temp:
		if len(k) != 1:
			del k[0]
		entities[i].append(int(k[0].strip()))
		
df['LastName'] = df['Name'].str.split(",").str[0]
families = entities
for i in range (0, len(families)):
	if len(families[i]) != 1:
		if (!is_together(families[i], df)):
			del families[i]
	else:
		del families[i]	

takenRoomsArray = []
for index, cabin in df['Cabin'].dropna().str.split(" ").iteritems():
	takenRoomsArray.append(cabin)

for i in range (0, len(takenRoomsArray)):
	if len(takenRoomsArray[i]) > 1:
		for j in range (0, len(takenRoomsArray[i])):
			if len(takenRoomsArray[i][j]) == 1:
				del takenRoomsArray[i][j]
	else if len(takenRoomsArray[i]) == 1:
		if takenRoomsArray[i][0] == "T":
			continue
		else:
			del takenRoomsArray[i]
	
takenRoomsList = []
for i in range (0, len(takenRoomsArray)):
	for j in range (0, len(takenRoomsArray[i]):
		takenRoomsList.append(takenRoomsArray[i][j])

deck = ["A", "B", "C", "D", "E", "F", "G"]
roomsDeck = [37, 102, 148, 100, 172, 184, 98]
allRooms = [] # Total of 840 rooms
allRooms.append('T') # Special instance of room located on unmentioned deck
					   # Has highest priority w.r.t. decks as it is the highest deck
for i in range (0,7):
	for j in range (0, roomsDeck[i]):
		allRooms.append(deck[i] + str(j + 1))
		
roomsAvailable = list(set(allrooms).symmetric_difference(set(takenRoomsList))
availCount = 0
df['CabinFill'] = df['Cabin']
for i in range (0,3):
	for j in range (0,3):
		passengerOrder = df[(df['BoardOrder'] == i+1) & (df['Pclass'] == j+1) & \
			(df['Cabin'].isnull())]['Fare'].sort_values(ascending=False)
		for index, value in passengerOrder.iteritems():
			# Note that some passengers are assigned the same room due to them being part of the same family
			for k in range 
			if (index + 1) in families
			df.loc[(index),'CabinFill'] = roomsAvailable[availCount]
			
df['Deck'] = df['CabinFill'].str.split(" ").str[0].str[0]
df['CabinNum'] = df['CabinFill'].str.split(" ").str[0].str.slice(1,4)

df.drop(['Name', 'Sex', 'Cabin', 'CabinFill', 'Embarked', 'Ticket', 'Fare', 'Age', 'LastName', 'SibSp', 'Parch'], axis=1)

train_data = df.values
