from sys import argv
f = open(argv[1])

lines = f.readlines()
# output = "id,country"
for line in lines:
	curr_line = line.rstrip().split(',')
	if curr_line[1] == 'NDF':
		token = ',US'
	else:
		token = ',NDF'
	print line.rstrip()
	print curr_line[0] + token


