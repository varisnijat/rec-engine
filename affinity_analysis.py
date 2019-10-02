from pyspark.sql import SparkSession
from pyspark.sql.functions import col,collect_set,size
from pyspark.ml.fpm import FPGrowth
import pyspark.sql.functions as sf


class AffinityAnalysis:

    """
    A class object for doing market basket analyisis. Dataset should be
    transactional data
    TODO a method for saving the model and another one for loading the model
    """
    user = "person_id"
    product = "product_categoryid_id"

    # modify values
    minSupport = 0.01
    minConfidence = 0.7

    def __init__(self,sc,data_path):

        """
        :sc: String. Name of the Spark App such as Recommendation Engine
        :dataset_path: File Path. Should be absolute file path
        """

        self.sc = sc
        self.spark = SparkSession.builder.appName(self.sc).getOrCreate()
        self.data = self.spark.read.csv(data_path, inferSchema = True, header = True)
        self.data2 = self.data.groupBy(self.user).agg(collect_set(self.product).alias('items'))
        self.__train_fpgrowth_model()


    def __train_fpgrowth_model(self):

        fpGrowth = FPGrowth(itemsCol="items",minSupport=self.minSupport, minConfidence=self.minConfidence)
        self.model = fpGrowth.fit(self.data2)


    def get_frequent_items(self):
        return self.model.freqItemsets

    
    def write_frequent_items(self,file_path):
        result = self.get_frequent_items().toPandas()
        result.to_csv(file_path,index = False,encoding = 'utf-8')
        return


    def get_association_rules(self):
        return self.model.associationRules


    def write_association_rules(self,file_path):
        result = self.get_association_rules().toPandas()
        result.to_csv(file_path,index = False,encoding = 'utf-8')
        return


    def set_min_support(self,minSupport):
        self.minSupport = minSupport
        self.__train_fpgrowth_model


    def get_min_support(self):
        return self.minSupport


    def set_min_confidence(self,minConfidence):
        self.minConfidence = minConfidence
        self.__train_fpgrowth_model


    def get_min_confidence(self):
        return self.minConfidence
    
    def transform_dataset(self,file_path):

        prediction = self.model.transform(self.data2).where(size(col("prediction"))>0).toPandas()
        prediction.to_csv(file_path,index = False,encoding = 'utf-8')
        return 
        