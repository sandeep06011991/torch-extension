all:
	cd cpp; python setup.py build_ext --inplace
	cd python; python naive.py
