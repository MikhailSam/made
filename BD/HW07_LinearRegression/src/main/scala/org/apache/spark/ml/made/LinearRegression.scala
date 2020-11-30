package org.apache.spark.ml.made

import breeze.linalg.{Vector => BrVector}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
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

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {
  val lr = 0.0001
  val num_rounds = 100

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val encoder : Encoder[Vector] = ExpressionEncoder()
//    val vectors: Dataset[Vector] = dataset.select(dataset($(inputCol)).as[Vector])
    val W_size: Int = MetadataUtils.getNumFeatures(dataset, $(inputCol))
    var W: BrVector[Double] = BrVector.rand[Double](W_size)

    for (i <- 0 to num_rounds) {

    }

    val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
      val summarizer = new MultivariateOnlineSummarizer()
      data.foreach(v => summarizer.add(mllib.linalg.Vectors.fromBreeze(v.asBreeze)))
      Iterator(summarizer)
    }).reduce(_ merge _)

    copyValues(new LinearRegressionModel(
      summary.mean.asML,
      Vectors.fromBreeze(breeze.numerics.sqrt(summary.variance.asBreeze)))).setParent(this)
 }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                         override val uid: String,
                                         val means: DenseVector,
                                         val stds: DenseVector) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(means: Vector, stds: Vector) =
    this(Identifiable.randomUID("standardScalerModel"), means.toDense, stds.toDense)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(means, stds))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bMean = means.asBreeze
    val bStds = stds.asBreeze
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        Vectors.fromBreeze((x.asBreeze - bMean) /:/ bStds)
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = means.asInstanceOf[Vector] -> stds.asInstanceOf[Vector]

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

      val (means, std) =  vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector]).first()

      val model = new LinearRegressionModel(means, std)
      metadata.getAndSetParams(model)
      model
    }
  }
}
