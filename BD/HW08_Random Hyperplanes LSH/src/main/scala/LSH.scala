package org.apache.spark.ml.made

import breeze.numerics.signum
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg
import org.apache.spark.ml.feature.{LSH, LSHModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWriter, SchemaUtils}
import org.apache.spark.ml.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

import scala.util.Random


// ESTIMATOR
class RandomHyperplaneLSH (override val uid: String)
  extends LSH[RandomHyperplaneLSHModel] {

  override def setInputCol(value: String): this.type = super.setInputCol(value)
  override def setOutputCol(value: String): this.type = super.setOutputCol(value)
  override def setNumHashTables(value: Int): this.type = super.setNumHashTables(value)

  def this() = this(Identifiable.randomUID("randomHyperplaneLSH"))

  override protected[this] def createRawLSHModel(inputDim: Int): RandomHyperplaneLSHModel = {
    val rand = new Random(0)
    val randHyperPlanes: Array[Vector] = {
      Array.fill($(numHashTables)) {
        Vectors.dense(Array.fill(inputDim)(2*rand.nextDouble - 1))
      }
    }
    new RandomHyperplaneLSHModel(uid, randHyperPlanes)
  }

  override def copy(extra: ParamMap): LSH[RandomHyperplaneLSHModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}


// MODEL
class RandomHyperplaneLSHModel private[made] (
                                     override val uid: String,
                                     val randHyperPlanes: Array[Vector]
                                   ) extends LSHModel[RandomHyperplaneLSHModel] {

  override def setInputCol(value: String): this.type = super.set(inputCol, value)

  override def setOutputCol(value: String): this.type = super.set(outputCol, value)

  private[made] def this(randHyperPlanes: Array[Vector]) =
    this(Identifiable.randomUID("randomHyperplaneLSH"), randHyperPlanes)

  override protected[ml] def hashFunction(elems: linalg.Vector): Array[linalg.Vector] = {
    val hashValues = randHyperPlanes.map { case plane =>
      signum(
        elems.nonZeroIterator.map { case (i, v) =>
          v * plane(i)
        }.sum
      )
    }
    hashValues.map(Vectors.dense(_))
  }

  override protected[ml] def keyDistance(x: linalg.Vector, y: linalg.Vector): Double = {
    if (Vectors.norm(x, 2) == 0 || Vectors.norm(y, 2) == 0){
      1.0
    } else {
      1.0 - x.dot(y) / (Vectors.norm(x, 2) * Vectors.norm(y, 2))
    }
  }

  override protected[ml] def hashDistance(x: Seq[Vector], y: Seq[Vector]): Double = {
    x.iterator.zip(y.iterator).map(vectorPair =>
      vectorPair._1.toArray.zip(vectorPair._2.toArray).count(pair => pair._1 != pair._2)
    ).min
  }

  override def copy(extra: ParamMap): RandomHyperplaneLSHModel = {
    val copied = new RandomHyperplaneLSHModel(uid, randHyperPlanes).setParent(parent)
    copyValues(copied, extra)
  }

  override def write: MLWriter = new RandomHyperplaneLSHModel.RandomHyperplaneLSHModelWriter(this)
}

object RandomHyperplaneLSHModel extends MLReadable[RandomHyperplaneLSHModel] {
  override def read: MLReader[RandomHyperplaneLSHModel] = new MLReader[RandomHyperplaneLSHModel] {
    override def load(path: String): RandomHyperplaneLSHModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val data = sparkSession.read.parquet(path + "/vectors")
      val Row(randPlanes: Matrix) = MLUtils.convertMatrixColumnsToML(data, "randHyperPlanes")
        .select("randHyperPlanes")
        .head()
      val model = new RandomHyperplaneLSHModel(metadata.uid, randPlanes.rowIter.toArray)
      metadata.getAndSetParams(model)
      model
    }
  }

  override def load(path: String): RandomHyperplaneLSHModel = super.load(path)

  private[RandomHyperplaneLSHModel] class RandomHyperplaneLSHModelWriter(instance: RandomHyperplaneLSHModel)
    extends MLWriter {

    private case class Data(randHyperPlanes: Matrix)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val dataPath = new Path(path, "data").toString

      val i = instance.randHyperPlanes.length
      val j = instance.randHyperPlanes.head.size
      val values = instance.randHyperPlanes.map(_.toArray).reduce(Array.concat(_, _))
      val data = Data(Matrices.dense(i, j, values))
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

}