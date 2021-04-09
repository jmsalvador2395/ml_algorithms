import sys
import random
import math

def split(fname, percentage):
	
	#count samples
	samples=0
	src=open(fname, 'r')
	for i in src:
		samples+=1
	src.close()
	#end count of samples

	choices=random.sample(range(0,samples), math.floor(samples*(percentage/100.)))
	testset=open(fname + ".t", "w")
	trngset=open(fname + ".tr", "w")

	count=0
	with open(fname, "r") as src:
		for line in src:
			if count in choices:
				trngset.write(line)
			else:
				testset.write(line)
			count+=1

	testset.close()
	trngset.close()
