import json
import pandas

'''
Load config.json
'''
f = open('config.json')
config = json.load(f)
f.close()

print(config)

data_df = pandas.read_excel(io=config["DATA_PATH"])
print(data_df)