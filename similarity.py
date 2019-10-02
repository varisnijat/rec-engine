import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf,collect_list,split,regexp_replace,col
from pyspark.ml.feature import HashingTF, IDF, Normalizer


class Similarity:

    """A class object for calculating similarity between items and users
        Content-Based Recommendation
    """

    def __init__(self,sc,data_path, productCol = "product_categoryid_id",
        userCol = "person_id", taxonomyCol = "taxonomy_id"):

        self.sc = sc
        self.spark = SparkSession.builder.appName(self.sc).getOrCreate()
        self.data = self.spark.read.csv(
            data_path, inferSchema = True, header = True)
        self.productCol = productCol
        self.userCol = userCol
        self.taxonomyCol = taxonomyCol


    def __data_manipulation(self,col):

        data = self.data.select(col,self.taxonomyCol).distinct()
        data = data.withColumn(self.taxonomyCol,data[self.taxonomyCol].cast(StringType()))

        concat_list = udf(lambda lst: ", ".join(lst), StringType())
        data = data.groupby(col).agg(collect_list(self.taxonomyCol).alias(self.taxonomyCol))

        data = data.withColumn(
            self.taxonomyCol, concat_list(self.taxonomyCol))
        data = data.withColumn(
            self.taxonomyCol, split(regexp_replace(self.taxonomyCol, " ", ""), ','))

        hashingTF = HashingTF(inputCol=self.taxonomyCol, outputCol="tf")
        tf = hashingTF.transform(data)

        idf = IDF(inputCol="tf", outputCol="feature").fit(tf)
        tfidf = idf.transform(tf)

        normalizer = Normalizer(inputCol="feature", outputCol="norm")
        norma_data = normalizer.transform(tfidf)

        return norma_data


    def get_product_similarity(self):

        """
        Calculate the similarity between items/users
        """
        product_taxonomy = self.data.select(
            self.productCol,self.taxonomyCol).distinct()
        product_taxonomy = self.__data_manipulation(product_taxonomy)

        hashingTF = HashingTF(inputCol=self.taxonomyCol, outputCol="tf")
        tf = hashingTF.transform(product_taxonomy)

        idf = IDF(inputCol="tf", outputCol="feature").fit(tf)
        tfidf = idf.transform(tf)

        normalizer = Normalizer(inputCol="feature", outputCol="norm")
        norma_data = normalizer.transform(tfidf)

        col1 = "i." + self.productCol
        col2= "j." + self.productCol

        dot_udf = udf(lambda x,y: float(x.dot(y)), DoubleType())
        result = norma_data.alias("i").crossJoin(norma_data.alias("j"))\
            .select(
                col(col1).alias("i"), 
                col(col2).alias("j"), 
                dot_udf("i.norm", "j.norm").alias("dot"))\
            .sort("i", "j")

        result = result.filter(result.i < result.j & result.dot > 0.5)

        return result


    def get_similarity(self, file_path,product_col = None, user_col = None):

        """
        Calculate similarity among prodoucts or users.
        """

        assert(product_col != None) != (user_col != None),"Must indicate which column to use"
        dot_udf = udf(lambda x,y: float(x.dot(y)), DoubleType())

        if product_col:
            norma_data = self.__data_manipulation(self.productCol)

            col1 = "i." + self.productCol
            col2= "j." + self.productCol

            result = norma_data.alias("i").crossJoin(norma_data.alias("j"))\
                .select(
                    col(col1).alias("i"),
                    col(col2).alias("j"),
                    dot_udf("i.norm", "j.norm").alias("dot"))\
                .sort("i", "j")

        if user_col:
            norma_data = self.__data_manipulation(self.userCol)

            col1 = "i." + self.userCol
            col2= "j." + self.userCol
            result = norma_data.alias("i").crossJoin(norma_data.alias("j"))\
                .select(
                    col(col1).alias("i"),col(col2).alias("j"),
                    dot_udf("i.norm", "j.norm").alias("dot")).sort("i", "j")

        result = result.filter(result.i < result.j).filter("dot > 0.5").toPandas()
        result.to_csv(file_path,index = False)
        # result.coalesce(1).write.csv(filePath)

        return


    def get_productCol(self):
        return self.productCol


    def get_userCol(self):
        return self.userCol

    def get_taxonomyCol(self):
        return self.taxonomyCol
