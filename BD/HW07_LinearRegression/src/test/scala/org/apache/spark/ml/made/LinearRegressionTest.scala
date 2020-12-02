package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  lazy val train_data: DataFrame = LinearRegressionTest._train_data
  lazy val test_data: DataFrame = LinearRegressionTest._test_data

  val delta = 0.0001

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val result = model.transform(data).collect()

//    result.length should be(4)
//    //ToDo: add more code
//    result(0) should be(12.0 +- delta)
//    vectors(1)(2) should be(10.0 +- delta)
//    vectors(2)(2) should be(2.0 +- delta)
//    vectors(3)(2) should be(0.5 +- delta)
  }

  "Estimator" should "fit" in {
    val estimator: LinearRegression = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(train_data)
    val vectors = model.transform(test_data).collect()

    vectors.length should be(4)
    println(vectors.mkString(" "))
  }

  "Estimator" should "predict" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(train_data)
    validateModel(model, test_data)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline
      .load(tmpFolder.getAbsolutePath)
      .fit(train_data).stages(0)
      .asInstanceOf[LinearRegressionModel]

    validateModel(model, test_data)
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val model = pipeline.fit(train_data)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)
    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], test_data)
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _train_vectors = Seq(
    Vectors.dense(1, 1, 3),
    Vectors.dense(1, 2, 5),
    Vectors.dense(2, 3, 8),
    Vectors.dense(2, 4, 10),
    Vectors.dense(3, 5, 13)
  )
//  lazy val y = Vectors.dense(12,10,0,2)
  lazy val _train_data: DataFrame = {
    import sqlc.implicits._
    _train_vectors.map(x => Tuple1(x)).toDF("features")
  }

  lazy val _test_vectors = Seq(
    Vectors.dense(1, 1),
    Vectors.dense(1, 2),
    Vectors.dense(2, 3),
    Vectors.dense(2, 4))
  lazy val _test_data: DataFrame = {
    import sqlc.implicits._
    _test_vectors.map(x => Tuple1(x)).toDF("features")
  }
}