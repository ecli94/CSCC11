# CSCC11 Bonus Project
# Author: Echo Li
# Student#: 1000169505
# University of Toronto: Scarborough

import numpy as np
import pandas as pd
from scipy.stats import skewnorm
import matplotlib.pyplot as mpl
from sklearn.ensemble import RandomForestClassifier

def round_to_half(n):
	if n <= 0:
		return 0.5
	return int(n*2)/2
	
def find_entities(ds):
	entities = []
	for i in list(ds):
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
	return entities
	
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
	
def sort_board_order(toSort, sortKey):
	sortedOrder = []
	for i in sortKey:
		for j in toSort:
			if i in j:
				sortedOrder.append(j)
	return unique(sortedOrder)
	
def unique(input):
	output = []
	for x in input:
		if x not in output:
			output.append(x)
	return output
	
def handleDF(df):
	df['Gender'] = df['Sex'].map({ 'female': 0, 'male': 1}).astype(int)
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	df['BoardOrder'] = df['Embarked'].fillna('S').map ({ 'S': 1, 'C': 2, 'Q': 3 })
	df['AgeFill'] = df['Age']
	df['LastName'] = df['Name'].str.split(",").str[0]
	df['CabinFill'] = df['Cabin']
	
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
	
	for i in range (0,2):
		for j in range (0,3):
			for index, value in df[(df['Age'].isnull()) & (df['Gender'] == i) & \
				(df['Pclass'] == j+1)]['PassengerId'].iteritems():
				# uses a skewed normal distribution to produce random values for age
				val = round_to_half(skewnorm.rvs(skew_ages[i][j], \
					loc=mean_ages[i][j], scale=std_ages[i][j]))
				df.loc[(index),'AgeFill'] = val
				
	allEntities = find_entities(df['PassengerId'].groupby(df['Ticket']))
	roomlessEntities = find_entities(df[(df['Cabin'].isnull())]['PassengerId'].groupby(df['Ticket']))
	families = []
	for i in allEntities:
		if len(i) != 1:
			if not (is_together(i, df)):
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
			allRooms.append(deck[i] + str(j+1))
	
	roomsNeededIds = []
	for i in range (0,3):
		for j in range (0,3):
			passengerOrder = df[(df['BoardOrder'] == j+1) & (df['Pclass'] == i+1) & \
				(df['Cabin'].isnull())]['Fare'].sort_values(ascending=False)
			for index, value in passengerOrder.iteritems():
				roomsNeededIds.append(index+1)
		
	sortedRoomsNeededIds = sort_board_order(roomlessEntities, roomsNeededIds)
	roomsAvailable = sort_alphanumeric_cabin(list(set(allRooms).symmetric_difference(set(takenRoomsList))))
	for i in sortedRoomsNeededIds:
		if not roomsAvailable:
			print("Too many passengers aboard")
			break
		# Some passengers are assigned the same room due to them being part of the same family
		familyRoom = roomsAvailable.pop(0)
		for j in i:
			df.loc[(j-1), 'CabinFill'] = familyRoom
				
	df['Deck'] = df['CabinFill'].str.split(" ").str[0].str[0].map({"A": 1, \
		"B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7})
	#df['CabinNum'] = df['CabinFill'].str.split(" ").str[0].str.slice(1,4)
	df.drop(['Name', 'Sex', 'Cabin', 'CabinFill', 'Embarked', 'BoardOrder' \
		'Ticket', 'Fare', 'Age', 'LastName', 'SibSp', 'Parch'], axis=1)

def main(train, test):
	traindf = pd.read_csv(input, header=0)
	testdf = pd.read_csv(input, header=0)
	#traindf = pd.read_csv('C:/Users/3Aceli/Desktop/CSCC11Bonus/csv/train.csv', header=0)
	#testdf = pd.read_csv('C:/Users/3Aceli/Desktop/CSCC11Bonus/csv/test.csv', header=0)
	postTrain = handleDF(traindf)
	postTest = handleDF(testdf)
	
	x_train = postTrain.drop("Survived", axis=1).copy()
	y_train = postTrain["Survived"]
	x_test = postTest.drop("PassengerId", axis=1).copy()
	
	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(x_train, y_train)
	forest.score(x_train, y_train)
	y_pred = forest.predict(x_test)
	
	submission = pd.DataFrame({
		"PassengerId": postTest["PassengerId"],
		"Survived": y_pred
	})
	submission.to_csv('titanicTest.csv', index=False)
	
