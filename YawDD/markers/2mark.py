import pandas as pd
import os


for path in ['../Male/', '../Female/']:
    for root,subdir,files in os.walk(path):
        for file in files:
            with open(path+file, 'r') as f:
                yawn = []
                degree = f.readline()
                while degree:
                    yawn.append(int(degree))
                    degree = f.readline()
            df = pd.DataFrame(yawn, columns=['yawn'])
            df.to_csv(file.replace('_mark.txt', '.csv'), index=False)
        