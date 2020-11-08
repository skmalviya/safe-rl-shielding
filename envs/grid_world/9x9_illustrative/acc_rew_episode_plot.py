from myplot import *

grid_world_files = ['9x9_illustrative_0','9x9_illustrative_1','9x9_illustrative_3']

for gwf in grid_world_files:
	with open(gwf+'_episodelen.data','r') as f:
		episodelen = [int(next(f)) for x in range(100)]
	data = {}
	data[gwf] = []
	with open(gwf+'_avg_reward.data','r') as f:
		i=0
		j=0
		for e in episodelen:
			print (e)
			j = e
			lst = f.readlines()[i:j]
			lst = [float(i) for i in lst]
			print (lst[0], type(lst[0]))
			i = e

			#exit()
			#print ([float(f.next()) for x in range(e)])
			#exit()
			#data[gwf].append(sum())
print (data['9x9_illustrative_3'])
exit()
data = {}
data['GSAT']	= 	[61.03, 73.25, 76.05, 79.69, 83.19]
data['SUMBT']	=	[59.23, 71.37, 74.41, 74.47, 77.14]
data['GCE']		=	[51.67, 64.49, 71.73, 71.98, 76.83]
data['GLAD']	= 	[49.24, 61.76, 71.98, 72.57, 75.25]
data['Simple BERT-DST'] = [49, 49.60, 63.89, 64.44, 68.75]
data['NBT-CNN'] = 	[40.50, 53.60, 62.90, 67.20, 69.00]
data['NBT-DNN'] = 	[35.30, 46.60, 51.90, 56.20, 61.50]


index = ['20','40','60','80','100']
myplot(data,index,'line_plot')

#bar_plot(data,index,'bar_plot')