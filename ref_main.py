from data import Data
from subset import PrivateDomain
from model import sess_runner

#main_dataset = Data(log=False,
#	filename='../data/3732_filtered.txt',
#	batch_size=50,
#	sep=' ')

main_dataset = Data(filename='../data/3732_filtered.txt',
        split=True,
        split_start=800,
	#additional_info='../data/gse80655_annotation.txt',
 	batch_size=50,
	sep=' ',
 	log=False)

supporting_dataset = Data(filename='../data/3732_filtered.txt',
        split=True,
        split_end=800,
        ind=1,
        #additional_info='../data/gse80655_annotation.txt',
        batch_size=50,
        sep=' ',
        log=False)


def runner():
	model = [PrivateDomain(main_dataset, delay=1, tagged=False)]

	model.append(PrivateDomain(supporting_dataset, 
	 	weight=1, 
	 	ind=1, 
	 	tagged=False))

	sess_runner(model)

runner()




