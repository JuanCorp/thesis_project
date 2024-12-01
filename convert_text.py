import pandas as pd
import base64

chunksize = 10 ** 4
new_dataset = pd.DataFrame([])
with pd.read_csv("news-docs.2015.en.filtered",sep="\t",usecols=[0,1], chunksize=chunksize) as reader:
    for chunk in reader:
        new_dataset= pd.concat([new_dataset,chunk])


new_dataset.columns = ["date","en"]
new_dataset["en"] = new_dataset["en"].apply(base64.b64decode).apply(lambda x: x.decode("utf-8"))
new_dataset.to_csv("news_docs_en_2015.csv",index=False)