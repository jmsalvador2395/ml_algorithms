import numpy.random as rnd
import sys

testset=open(sys.argv[1] + ".t", "w")
trngset=open(sys.argv[1] + ".tr", "w")

with open(sys.argv[1], "r") as src:
	for line in src:
		if rnd.randint(0, 100) > 10:
			trngset.write(line)
		else:
			testset.write(line)

testset.close()
trngset.close()
