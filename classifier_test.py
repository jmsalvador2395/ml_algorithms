import sys
import datetime
import perceptron.perceptron as pc
import perceptron.multiclass_perceptron as mpc
import knn.knn as knn
import random_subsample as rs
import split as sp

f_iris=sys.argv[1]
iris_features=int(sys.argv[2])
f_a4a=sys.argv[3]
a4a_features=int(sys.argv[4])



#-------------------------------------------#
#				iris section				#
#-------------------------------------------#
sp.split(f_iris, 20)

print("-------------------iris multiclass perceptron-------------------")
for steps in range(10,25,5):
	print("**mcp for iris dataset with step size=" + str(steps) + "**")
	start_time=datetime.datetime.now()
	w=mpc.learning(f_iris + ".tr", 10, iris_features, steps)
	total_time=datetime.datetime.now()-start_time
	print("it took " + str(total_time.total_seconds()) + " seconds to learn the weights")
	print("\nlearned weights:")
	print(w)
	mpc.activation(f_iris + ".t", w, iris_features)
	print()

print("-------------------iris knn manhattan-------------------")
for i in range(5,20,5):
	knn.knn(i, f_iris + ".tr", f_iris + ".t", iris_features, 'manhattan')
	print()
print("-------------------iris knn euclidean-------------------")
for i in range(5,20,5):
	knn.knn(i, f_iris + ".tr", f_iris + ".t", iris_features, 'euclidean')
	print()





#-------------------------------------------#
#				a4a section					#
#-------------------------------------------#

rs.subsample(f_a4a, 30)

print("-------------------a4a perceptron-------------------")
for steps in range(10,25,5):
	print("**mcp for a4a dataset with step size=" + str(steps) + "**")
	start_time=datetime.datetime.now()
	w=mpc.learning(f_a4a + ".subs", 10, a4a_features, steps)
	total_time=datetime.datetime.now()-start_time
	print("it took " + str(total_time.total_seconds()) + " seconds to learn the weights")
	print("\nlearned weights:")
	print(w)
	mpc.activation(f_a4a + ".t", w, a4a_features)
	print()


print("-------------------a4a knn manhattan-------------------")
for i in range(10,310,100):
	knn.knn(i, f_a4a + ".tr", f_a4a + ".t", a4a_features, 'manhattan')
	print()
print("-------------------a4a knn euclidean-------------------")
for i in range(10,310,100):
	knn.knn(i, f_a4a + ".tr", f_a4a + ".t", a4a_features, 'euclidean')
	print()
