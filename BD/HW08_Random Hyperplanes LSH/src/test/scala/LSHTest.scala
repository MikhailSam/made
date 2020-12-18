package org.apache.spark.ml.made

import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

class RandomHyperplaneLSHTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.01
  lazy val vectors: Seq[Vector] = RandomHyperplaneLSHTest._test_vectores
  lazy val data: DataFrame = RandomHyperplaneLSHTest._test_data
  lazy val hyperplanes: Array[Vector] = RandomHyperplaneLSHTest._test_hp

  "Model" should "calculate hash" in {
    val model: RandomHyperplaneLSHModel = new RandomHyperplaneLSHModel(
      randHyperPlanes = hyperplanes
    ).setInputCol("features")
      .setOutputCol("hashes")
    val v = Vectors.fromBreeze(breeze.linalg.Vector(1, 2, 3, 4))
    val h = model.hashFunction(v)
    h.foreach(println)

    h.length should be(3)
    h(0)(0) should be(1.0)
    h(1)(0) should be(1.0)
    h(2)(0) should be(-1.0)
  }

  "Model" should "calculate hash distance" in {
    val model: RandomHyperplaneLSHModel = new RandomHyperplaneLSHModel(
      randHyperPlanes = hyperplanes
    ).setInputCol("features")
      .setOutputCol("hashes")
    val v1 = Vectors.fromBreeze(breeze.linalg.Vector(1, 2, 3, 4))
    val v2 = Vectors.fromBreeze(breeze.linalg.Vector(2, 4, 6, 8))
    val h1 = model.hashFunction(v1)
    val h2 = model.hashFunction(v2)
    val similarity = model.hashDistance(h1, h2)
    h1.foreach(println)
    h2.foreach(println)

    similarity should be (0.0 +- delta)
  }

  "Model" should "calculate distance" in {
    val model: RandomHyperplaneLSHModel = new RandomHyperplaneLSHModel(
      randHyperPlanes = hyperplanes
    ).setInputCol("features")
      .setOutputCol("hashes")
    val v1 = Vectors.fromBreeze(breeze.linalg.Vector(1, 2, 3, 4))
    val v2 = Vectors.fromBreeze(breeze.linalg.Vector(4, 3, 2, 1))
    val dist = model.keyDistance(v1, v2)
    println(dist)
    dist should be(1.0/3.0 +- delta)
  }

  "Model" should "transform" in {
    val plane: RandomHyperplaneLSH = new RandomHyperplaneLSH(
    ).setNumHashTables(2)
      .setInputCol("features")
      .setOutputCol("hashes")
    val output = plane.fit(data).transform(data)
    output.count() should be(3)
  }
}


object RandomHyperplaneLSHTest extends WithSpark {
  lazy val _test_vectores = Seq(
    Vectors.dense(1, 2, 3, 4),
    Vectors.dense(5, 4, 9, 7),
    Vectors.dense(9, 6, 4, 5)
  )

  lazy val _test_data: DataFrame = {
    import sqlc.implicits._
    _test_vectores.map(x => Tuple1(x)).toDF("features")
  }

  lazy val _test_hp = Array(
    Vectors.dense(1, -1, 1, 1),
    Vectors.dense(-1, 1, -1, 1),
    Vectors.dense(1, 1, -1, -1)
  )

}
