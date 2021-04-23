import random
import math
import sys

def subsample(input_f, percentage):
	samples=0
	inp=open(input_f, 'r')
	for i in inp:
		samples+=1
	inp.close()

	choices=random.sample(range(0,samples), math.floor(samples*(percentage/100.)))

	inp=open(input_f, 'r')
	out=open(input_f + ".subs", 'w')

	for i in range(samples):
		line=inp.readline()
		if i in choices:
			out.write(line)
	inp.close()
	out.close()

if __name__ == "__main__":
    subsample(sys.argv[1], int(sys.argv[2]))
