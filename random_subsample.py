import random
import math
import sys

def subsample(input_f, samples, percentage):
	choices=random.sample(range(0,samples), math.floor(samples*(percentage/100.)))

	inp=open(input_f, 'r')
	out=open(input_f + ".subs", 'w')

	for i in range(samples):
		line=inp.readline()
		if i in choices:
			out.write(line)
	inp.close()
	out.close()


subsample(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
