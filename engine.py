import csv
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
from pyspark.sql.functions import desc
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import array,concat_ws
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from math import sqrt


class RecommendationEngine:

    """Acumen Recommendation Engine
        Collaborative Filtering Approach
    """

    def __init__(self,sc,dataset_path, mlInstance = None):

        """Initiat the recommendation engine given a Spark context and a dataset
            path
            :sc: String. Name of the Spark App such as Recommendation Engine
            :dataset_path: File Path. Should be absolute file path
            :mlInstance: File path to the ml model
        """

        self.sc = sc
        self.spark = SparkSession.builder.appName(self.sc).getOrCreate()
        self.data = self.spark.read.csv(dataset_path,inferSchema = True, header = True)
        self.training, self.test = self.data.randomSplit([0.7,0.3])
        self.rank = 8
        self.maxIter = 10
        self.regParam = 0.01
        self.__train_ALS_model()

        if mlInstance == None:
            self.__train_ALS_model()
        else:
            self.__import_model(mlInstance)


    def __import_model(self,mlInstance):

        """
        Reads an ML instance from the input path
        :mlInstance: Path to the saved model
        """
        self.model = ALSModel.load(mlInstance)
        predictions = self.model.transform(self.test)
        evaluator = RegressionEvaluator(
            metricName = 'rmse', labelCol = 'product_rating', predictionCol = 'prediction')
        self.rmse = evaluator.evaluate(predictions)


    def __train_ALS_model(self):

        """Train the ALS model with the current dataset
        """

        self.als = ALS(
            rank = self.rank, maxIter = self.maxIter, regParam = self.regParam, 
            userCol = "person_id", itemCol = "product_categoryid_id", 
            ratingCol = "product_rating",coldStartStrategy="drop")
        self.model = self.als.fit(self.training)
        predictions = self.model.transform(self.test)
        evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'product_rating', predictionCol = 'prediction')
        self.rmse = evaluator.evaluate(predictions)


    def add_ratings(self,ratings):

        """
        TODO add additional ratings in the format (person_id,product_categoryid_id,
            product_rating)
        TODO determine the format for ratings
        """

    
    def __predict_ratings(self,user_and_products,product_count):

        """
        Gets predictions for a given (userID, productID) 
        Returns: a DataFrame with format (productID, userID, predictedRating)
        """

        predictions = self.model.transform(user_and_products).na.drop()
        predictions = predictions.sort(desc("prediction")).limit(
            product_count)

        return predictions


    def item_to_item(self):

        def computeCosineSimilarity(ratingPairs):

            numPairs = 0
            sum_xx = sum_yy = sum_xy = 0
            for ratingX, ratingY in ratingPairs:
                sum_xx += ratingX * ratingX
                sum_yy += ratingY * ratingY
                sum_xy += ratingX * ratingY
                numPairs += 1

            numerator = sum_xy
            denominator = sqrt(sum_xx) * sqrt(sum_yy)

            score = 0
            if (denominator):
                score = (numerator / (float(denominator)))
            
            return score

        #FIXME do i need it or not
        data = self.data.select("person_id","product_categoryid_id","product_rating")

        ratings = data.withColumn("pairs",array(
            data.product_categoryid_id,data.product_rating)).drop(
                "product_categoryid_id","product_rating")

        r1 = ratings.select("person_id",col("pairs").alias("pairs1"))
        r2 = ratings.select("person_id",col("pairs").alias("pairs2"))
        joinedRatings = r1.join(r2, r1.person_id == r2.person_id, how = "inner").drop(r1.person_id)

        uniqueJoinedRatings = joinedRatings.filter(
            joinedRatings.pairs1[0]<joinedRatings.pairs2[0])

        productPairs = uniqueJoinedRatings.withColumn(
            "key",
            array(uniqueJoinedRatings["pairs1"].getItem(0),
            uniqueJoinedRatings["pairs2"].getItem(0))).withColumn("ratings",
            array(uniqueJoinedRatings["pairs1"].getItem(1),
            uniqueJoinedRatings["pairs2"].getItem(1))).select("key","ratings")

        df = productPairs.groupBy("key").agg(collect_list("ratings").alias("pairs"))

        computeCosineSimilarity_udf = udf(computeCosineSimilarity,FloatType())
        score = df.withColumn("score",computeCosineSimilarity_udf("pairs")). \
            withColumn("movie_1",col("key").getItem(0)). \
                withColumn("movie_2",col("key").getItem(1)). \
                    filter("score>0.5")
        score = score.select("movie_1","movie_2","score")
        score.write.mode("overwrite") \
            .format("jdbc") \
            .option("url", "jdbc:redshift://assocanalytics.cxynt5jzlg3q.us-east-1.redshift.amazonaws.com:5439/db001") \
            .option("dbtable", "source.recm_item_tranx_similarity") \
            .option("user", "pipeline") \
            .option("password", "Data1sGreat!") \
            .save()

        return 


    def user_to_user(self):

        def computeCosineSimilarity(ratingPairs):

            numPairs = 0
            sum_xx = sum_yy = sum_xy = 0
            for ratingX, ratingY in ratingPairs:
                sum_xx += ratingX * ratingX
                sum_yy += ratingY * ratingY
                sum_xy += ratingX * ratingY
                numPairs += 1

            numerator = sum_xy
            denominator = sqrt(sum_xx) * sqrt(sum_yy)

            score = 0
            if (denominator):
                score = (numerator / (float(denominator)))
            
            return score

        data = self.data.select("person_id","product_categoryid_id","product_rating")
        ratings = data.withColumn("pairs",array(data.person_id,data.product_rating)).drop(
            "person_id","product_rating")

        r1 = ratings.select("product_categoryid_id",col("pairs").alias("pairs1"))
        r2 = ratings.select("product_categoryid_id",col("pairs").alias("pairs2"))
        joinedRatings = r1.join(
            r2, r1.product_categoryid_id == r2.product_categoryid_id, how = "inner").drop(r1.product_categoryid_id)

        uniqueJoinedRatings = joinedRatings.filter(
            joinedRatings.pairs1[0]<joinedRatings.pairs2[0])

        productPairs = uniqueJoinedRatings.withColumn(
            "key",
            array(uniqueJoinedRatings["pairs1"].getItem(0),
            uniqueJoinedRatings["pairs2"].getItem(0))).withColumn("ratings",
            array(uniqueJoinedRatings["pairs1"].getItem(1),
            uniqueJoinedRatings["pairs2"].getItem(1))).select("key","ratings")

        df = productPairs.groupBy("key").agg(collect_list("ratings").alias("pairs"))
        computeCosineSimilarity_udf = udf(computeCosineSimilarity,FloatType())

        score = df.withColumn("score",computeCosineSimilarity_udf("pairs")). \
            withColumn("user_1",col("key").getItem(0)). \
                withColumn("user_2",col("key").getItem(1)). \
                    filter("score>0.5")
        score = score.select("user_1","user_2","score")

        score.write.mode("overwrite") \
            .format("jdbc") \
            .option("url", "jdbc:redshift://assocanalytics.cxynt5jzlg3q.us-east-1.redshift.amazonaws.com:5439/db001") \
            .option("dbtable", "source.recm_user_tranx_similarity") \
            .option("user", "pipeline") \
            .option("password", "Data1sGreat!") \
            .save()
            
        return 


    def get_top_ratings(self,user_id,product_count):

        """
        Recommends up to product_count top unrated products to user id
        :user_id: an integer that repersents the identification number of a user
        :product_count: an iteger that represents the number of products that will
            be recommended to the user_id
        """

        #Get all distinct products and append a "person_id" column with given 
        # user_id as value
        products = self.data.select("product_categoryid_id").distinct()
        products = products.withColumn("person_id",lit(user_id))

        #get all products rated by given user_id
        person_rated_products = self.data.where(
            self.data.person_id == user_id).select("person_id","product_categoryid_id")
        person_rated_products = person_rated_products.select(
            col("person_id").alias("person"),col("product_categoryid_id").alias("product"))

        #left anti join to find all products that have not been rated by user
        left_join = products.join(
            person_rated_products,
            products.product_categoryid_id == person_rated_products.product,
            how = 'left_anti')

        ratings = self.__predict_ratings(left_join,product_count)

        # #FIXME
        # ratings.toPandas().to_csv("predictions.csv", encoding = 'utf-8',index = False)

        return ratings


    def recommend_for_all_users(self,count):

        """
        Returns top numItems items recommended for each user, for all users.
        """

        userRecs = self.model.recommendForAllUsers(count)
        userRecsCollect = userRecs.collect()
        
        with open("Recommendation_for_All.csv",'a',encoding = 'utf-8') as file:

            fileWriter = csv.writer(file, delimiter = ',', quotechar='"', lineterminator='\n')
            fileWriter.writerow(["person_id","product_categoryid_id","predicted_product_rating"])

            for row in userRecsCollect:
                person_id = row.person_id
                for recommendation in row.recommendations:
                    ans = []
                    ans.append(person_id)
                    ans.append(recommendation.product_categoryid_id)
                    ans.append(recommendation.rating)
                    fileWriter.writerow(ans)
        return


    def get_all_top_ratings(self, product_count):

        """
        Recommend up to product_count top unrated products to all users
        """
        products = self.data.select("product_categoryid_id").distinct()
        persons = self.data.select("person_id").distinct() 
        person_product = persons.crossJoin(products).select(
            col("person_id").alias("person"),col("product_categoryid_id").alias("product"))

        #get all unrated products for each user
        person_unrated_products = person_product.join(
            self.data,
            (self.data.product_categoryid_id == person_product.product) & (
                self.data.person_id == person_product.person),
            how = 'left_anti'
        ).select(
            col("person").alias("person_id"), col("product").alias("product_categoryid_id")).cache()

        persons = list(persons.toPandas()["person_id"])

        with open("output//all_predictions.csv",'w')as file:
            fileWriter = csv.writer(file, delimiter = ',', quotechar='"', lineterminator='\n')
            fileWriter.writerow(["person_id","product_categoryid_id","predicted_product_rating"])

            for person in persons:

                person_prediction = self.__predict_ratings(
                    person_unrated_products.filter(
                        person_unrated_products["person_id"] == person),product_count)

                for row in person_prediction:
                    fileWriter.writerow([row.person_id,row.product_categoryid_id,row.prediction])

        return 
        

    def get_rmse(self):
        return self.rmse


    def get_rank(self):
        return self.rank


    def set_rank(self,rank):
        self.rank = rank
        self.__train_ALS_model()
        return self.rmse

    
    def get_maxIter(self):
        return self.maxIter

    
    def set_maxIter(self,maxIter):
        self.maxIter = maxIter
        self.__train_ALS_model()
        return self.rmse

    
    def get_regParam(self):
        return self.regParam

    
    def set_regParam(self,regParam):
        self.regParam = regParam
        self.__train_ALS_model()
        return self.rmse


    def get_popular_products(self,count):

        """Function to return popular products up to count
        """
        popular_products = self.data.groupby("product_category_name")   \
            .count().sort(desc("count")).limit(count)
        popular_products.toPandas().to_csv("popular_products.csv",encoding = 'utf-8',index = False)