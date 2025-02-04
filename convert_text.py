import pandas as pd
import base64

new_dataset = pd.read_csv("news-docs.2015.en.filtered",sep="\t",usecols=[0,1]) 
new_dataset.columns = ["date","en"]
new_dataset["en"] = new_dataset["en"].apply(base64.b64decode).apply(lambda x: x.decode("utf-8"))
new_dataset.to_csv("news_docs_en_2015.csv",index=False)