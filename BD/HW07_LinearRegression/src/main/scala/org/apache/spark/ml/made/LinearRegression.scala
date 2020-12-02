package org.apache.spark.ml.made

import breeze.linalg.{DenseVector => BrVector, sum}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

// ESTIMATOR
class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {
  val lr = 0.001
  val num_rounds = 100

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    dataset.show()
    val input: Dataset[Vector] = dataset.select(dataset($(inputCol))).as[Vector]
    println(s"INPUT:")
    input.show()
    val f_size: Int = MetadataUtils.getNumFeatures(dataset, $(inputCol))
    var W: BrVector[Double] = BrVector.rand[Double](f_size) // include B

    for (i <- 0 to num_rounds) {
      val summary = input.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(
          v => {
            val X = v.asBreeze(0 until f_size).toDenseVector
            val preds = X * W(0 until f_size) + W(-1) // !!!!!!!!!!!!!!!!!!!!!
            val error = preds - v.asBreeze(-1)
            val grad = X * sum(error)
            summarizer.add(mllib.linalg.Vectors.fromBreeze(grad))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)
      W = W - lr * summary.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(W).toDense)
    ).setParent(this)
 }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

// MODEL
class LinearRegressionModel private[made](
                                         override val uid: String,
                                         val W: DenseVector) extends Model[LinearRegressionModel]
  with LinearRegressionParams with MLWritable {

  private[made] def this(W: DenseVector) =
    this(Identifiable.randomUID("standardScalerModel"), W)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(W))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        (x.asBreeze dot W.asBreeze(0 until x.size) + W.asBreeze(x.size))
      })
    println("WEIGHTS")
    println(W)
    print()
    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
//      val vectors = W.asInstanceOf[Vector] -> stds.asInstanceOf[Vector]
      val vectors = Tuple1(W.asInstanceOf[Vector])
      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val W = vectors.select(vectors("_1").as[Vector]).first()
      val model = new LinearRegressionModel(W.toDense)
      metadata.getAndSetParams(model)
      model
    }
  }
}
