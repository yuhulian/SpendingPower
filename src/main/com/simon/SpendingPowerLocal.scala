package com.simon

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by simon on 2017/7/27.
  */
object SpendingPowerLocal {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[2]").setAppName("SpendingPower")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    def castNullToZero(s: String): Double = {
      try {
        s.toDouble
      } catch {
        case _: Throwable => 0.0
      }
    }

    /*
    * step1. read and split the raw data and extract ids and features
    */
    val trainData = sc.textFile("file:\\E:\\【04#项目文档】\\#消费能力指数\\features.txt")
    //trainData: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[18] at textFile

    val parsedData = trainData.map { line =>
      val parts = line.split("\\|")
      val user_id = parts(0)
      val features = Vectors.dense(parts.slice(1, parts.length).map(castNullToZero(_)))
      (user_id, features)
    }
    //parsedData: org.apache.spark.rdd.RDD[(String, org.apache.spark.mllib.linalg.Vector)] = MapPartitionsRDD[19] at map

    val features = parsedData.map { x =>
      val label = 1.0
      val features = x._2
//      LabeledPoint(label, features)
      features
    }
      //.toDF().select("features")
    //features: org.apache.spark.sql.DataFrame = [features: vector]

    /*
    * step2. standard scaler proved not working well here
    */

    //    val scaler = new StandardScaler()
    //      .setInputCol("features")
    //      .setOutputCol("scaledFeatures")
    //      .setWithMean(true)
    //      .setWithStd(true)
    //      .fit(features)
    //scaler: org.apache.spark.ml.feature.StandardScalerModel = stdScal_98bd27f7c0fd

//    val scaler = new MinMaxScaler()
//      .setInputCol("features")
//      .setOutputCol("scaledFeatures")
//      .fit(features)

//    val scaledData = scaler.transform(features).select("scaledFeatures")
    //scaledData: org.apache.spark.sql.DataFrame = [scaledFeatures: vector]

    /*
   * step3. use PCA to select features
   * 1. build the RowMatrix with org.apache.spark.mllib.linalg.Vector(build the Vector first)
   * 2. compute Principal Components
   */
//    val scaledDataRDD = scaledData.map { x: Row =>
//      x.getAs[org.apache.spark.mllib.linalg.Vector](0)
//    }

    /*
    //another appoach to transform
    val scaledDataRDD2 = scaledData.rdd.map{
      row => Vectors.dense(row.getAs[org.apache.spark.mllib.linalg.Vector]("scaledFeatures").toArray)
    }
  */

    val rm = new RowMatrix(features)
    /*
     *  //if not using scaling
     *  val rm = new RowMatrix(parsedData.map(x=>x._2))
     */
    //val rows : org.apache.spark.rdd.RDD[org.apache.spark.mllib.linalg.Vector]

    /*
    * use PCA to do feature selection
     */
    val numPCA = 8
    //number of principle components you want to select
    val pc = rm.computePrincipalComponents(numPCA)
    pc.toArray.foreach(println)

    //generate the covariance matrix
    val projected = rm.multiply(pc).rows //generate the matix with columns you selected
    projected.take(10).foreach(println)
    /*
    * do K-means clustering
     */

    /*
    * select best K
     */
//    val ks: Array[Int] = Array(3, 4, 5, 6, 7, 8, 9, 10)
//    ks.foreach(cluster => {
//      val model: KMeansModel = KMeans.train(projected, cluster, 30, 1)
//      val ssd = model.computeCost(projected)
//      println("sum of squared distances of points to their nearest center when k=" + cluster + " -> " + ssd)
//    })


    //step2. clustering
    val numClusters = 7
    val numIterations = 10
    val runs = 3
    val clusters = KMeans.train(projected, numClusters, numIterations, runs)

    //step3. evaluate the result
    val WSSSE = clusters.computeCost(projected)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    //    //step4. get ids and labels, save them to the file
    //    val out = clusters.predict(training)
    //    parsedData.map(x=>(x._1,(x._2(0),x._2(1),x._2(2),x._2(3),x._2(4)))).zip(out).map(x=>x._1._1+"|"+x._1._2._1+"|"+x._1._2._2+"|"+x._1._2._3+"|"+x._1._2._4+"|"+x._1._2._5+"|"+x._2).foreach(println)
    //    parsedData.map(x=>x._1).zip(out).map(x=>x._1+"|"+x._2).coalesce(1).saveAsTextFile("file:\\E:\\【04#项目文档】\\#afflunce\\output\\")
  }
}
