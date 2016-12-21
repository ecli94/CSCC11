# CSCC11 Bonus Project
# Author: Echo Li
# Student#: 1000169505
# University of Toronto: Scarborough

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
		m1 = df.loc[(members[i])+1]
		m2 = df.loc[(members[i+1])+1]
		if m1['FamilySize'] == 1 | m2['FamilySize'] == 1:
			return False
		if m1['FamilySize'] != m2['FamilySize'] and \
			m1['LastName'] != m2['LastName']:
			return False
	return True
	
def sort_alphanumeric_cabin(L):
	temp_L1 = []
	temp_L2 = []
	for i in L:
		deck = i[:1]
		room = int(i[1:])
		temp_L1.append([deck, room])
	temp_L1.sort()
	for j in temp_L1:
		temp_L2.append(j[0] + str(j[1]))
	return temp_L2
	

df = pd.read_csv('csv/train.csv', header=0)
#df = pd.read_csv('C:/Users/3Aceli/Desktop/CSCC11Bonus/csv/train.csv', header=0)
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
families = []
for i in entities:
	if len(i) != 1:
		if not (is_together(families[i], df)):
			families.append(i)

takenRoomsArray = []
for index, cabin in df['Cabin'].dropna().str.split(" ").iteritems():
	takenRoomsArray.append(cabin)

for i in takenRoomsArray:
	if len(i) > 1:
		for j in i:
			if len(j) == 1:
				i.remove(j)
	elif len(i) == 1:
		if i[0] == "T":
			continue
		elif len(i[0]) != 1:
			continue
		else:
			takenRoomsArray.remove(i)
	
takenRoomsList = []
for i in range (0, len(takenRoomsArray)):
	for j in range (0, len(takenRoomsArray[i])):
		takenRoomsList.append(takenRoomsArray[i][j])

deck = ["A", "B", "C", "D", "E", "F", "G"]
roomsDeck = [37, 102, 148, 100, 172, 184, 98]
allRooms = [] # Total of 840 rooms
allRooms.append('T') # Special instance of room located on unmentioned deck
					   # Has highest priority w.r.t. decks as it is the highest deck
for i in range (0,7):
	for j in range (0, roomsDeck[i]):
		allRooms.append(deck[i] + str(j + 1))
		
roomsAvailable = sort_alphanumeric_cabin(list(set(allRooms).symmetric_difference(set(takenRoomsList))))
df['CabinFill'] = df['Cabin']
for i in range (0,3):
	for j in range (0,3):
		passengerOrder = df[(df['BoardOrder'] == i+1) & (df['Pclass'] == j+1) & \
			(df['Cabin'].isnull())]['Fare'].sort_values(ascending=False)
		for index, value in passengerOrder.iteritems():
			if not roomsAvailable:
				break
			# Note that some passengers are assigned the same room due to them being part of the same family
			for k in families:
				if (index + 1) in k:
					familyRoom = roomsAvailable.pop(0)
					break
			df.loc[(index),'CabinFill'] = roomsAvailable.pop(0)
			
df['Deck'] = df['CabinFill'].str.split(" ").str[0].str[0]
df['CabinNum'] = df['CabinFill'].str.split(" ").str[0].str.slice(1,4)

df.drop(['Name', 'Sex', 'Cabin', 'CabinFill', 'Embarked', 'Ticket', 'Fare', 'Age', 'LastName', 'SibSp', 'Parch'], axis=1)

train_data = df.values
