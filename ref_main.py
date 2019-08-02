from data import Data
from subset import PrivateDomain
from model import sess_runner

main_dataset = Data(log=True, filename='test_data/GSE80655_filtered.csv')


model = [PrivateDomain(main_dataset, delay=1, tagged=False, ind=1)]

# model.append(PrivateDomain(supporting_dataset, 
# 	weight=1, 
# 	ind=1, 
# 	tagged=False))

#model.append(nn.Model(check_dataset, column = 'Disease', ind=2, classes=[0, 1]))
print(next(main_dataset))

sess_runner(model)
#nn.sess_runner(model, ["red/model0.ckpt", "red/model1.ckpt", "red/model2.ckpt"])




