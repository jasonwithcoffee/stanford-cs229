import os
from zipfile import ZipFile

current_dir = os.path.dirname(os.path.abspath(__file__))

gda_path = os.path.join(current_dir, 'linearclass', 'src', 'gda.py')
logreg_path = os.path.join(current_dir, 'linearclass', 'src', 'logreg.py')
poisson_path = os.path.join(current_dir, 'poisson', 'src', 'poisson.py')

zipObj = ZipFile(os.path.join(current_dir, 'PS2_submission.zip'), 'w')
zipObj.write(gda_path, os.path.basename(gda_path))
zipObj.write(logreg_path, os.path.basename(logreg_path))
zipObj.write(poisson_path, os.path.basename(poisson_path))
