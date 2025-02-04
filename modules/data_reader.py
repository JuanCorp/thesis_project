import pandas as pd


class DataReader(object):

    def __init__(self,filepath="data/",filename="news_docs_en_2015.csv",sample=None):
        self.filepath=filepath
        self.filename=filename
        self.sample=sample
    

    def _read_data(self):
        full_filepath = self.filepath + self.filename
        if self.sample:
            self.data= pd.read_csv(full_filepath).sample(self.sample).reset_index(drop=True)
        else:
            self.data= pd.read_csv(full_filepath)
    
    def _select_text_features(self):
        text = self.data[self.usecols[0]]
        for col in self.usecols[1:]:
            text+=self.data[col]
        self.text = text
    

    def obtain_text_data(self,**kwargs):
        self._read_data()
        return self.data
        
    
