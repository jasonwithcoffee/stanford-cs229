import os
from zipfile import ZipFile

current_dir = os.path.dirname(os.path.abspath(__file__))

k_means_path = os.path.join(current_dir, 'src', 'k_means', 'k_means.py')
semi_supervised_path = os.path.join(current_dir, 'src', 'semi_supervised_em', 'gmm.py')

zipObj = ZipFile(os.path.join(current_dir, 'submission.zip'), 'w')
zipObj.write(k_means_path, os.path.basename(k_means_path))
zipObj.write(semi_supervised_path, os.path.basename(semi_supervised_path))
zipObj.close() 
