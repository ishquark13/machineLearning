ish_dict = {'Name': 'Ishmael', 'Level': 'Senior',
			'Major': 'Electrical Engineering & Math', 'Dept': 'ECE', 'username': 'ishquark13',
			'learn': 'Neural Networks for augmenting topologies' }

def assignment0():

	''' Assignment 0 for Introduction to Machine Learning, Sp 2018
		Provide information on your background in programming, python, git, and machine learning. 
		Provide insight into what you are hoping to learn in this class this semester. 
	 	Submit your assignment as described in class by pushing to github. 
	''' 

	print('My name is ' + ish_dict['Name'])
	print('I am a ' + ish_dict['Level'] + ' majoring in ' + ish_dict['Major'] +  ' in the Department of ' + ish_dict['Dept'] + '.')
	print('I have some programming experience in general.')
	print('I have some python programming experience.')
	print('I have some experience with using version control software')
	print('I have some experience with git.')
	print('I have some experience in machine learning.')
	print('My github username is ' + ish_dict['username'] + '.')
	print('I am excited to learn the following in this class: ' +  ish_dict['learn'])

if __name__ == "__main__":
	assignment0()