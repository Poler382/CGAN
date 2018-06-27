import java.util.Date

import GAN.{learning_D, savetxt1, savetxt2}
import breeze.collection.mutable.OpenAddressHashArray

import math._

object CGAN {
  val rand = new scala.util.Random(0)
  val namelist = List("lossG", "lossD", "real rate", "fake rate",
    "max1", "min1", "max2", "min2")

  val filepath = List(
    "src/main/scala/generalAdversalial.scala",
    "src/main/scala/generalAdversalial_layer.scala",
    "src/main/scala/Network.scala")

  val white = Array.ofDim[Double](784).map(_=>1d)
  val black = Array.ofDim[Double](784).map(_=>0d)


  def main(args: Array[String]) {
    val mode = args(0)
    val ln = args(1).toInt
    val dn = args(2).toInt
    val where = args(3)

    val mn = new mnist()

    val (dtrain, dtest) = where match {
      case "home" => {
        mn.load_list("C:/Users/poler/Documents/python/share/mnist")
      }
      case "lab" => {
        mn.load_list("/home/share/fashion-mnist")
      }
    }

    println("finish load")

    var acc_real = List[Double]()
    var acc_fake = List[Double]()
    var loss_Dlist = List[Double]()
    var loss_Glist = List[Double]()
    var loglist = List[String]()

    /*
        val date = "-%tm%<td-%<tHh" format new Date
        sys.process.Process("mkdir GAN/"+mode+date).run

        val path = mode+date

        for(file <-filepath){
          sys.process.Process("cp "+file+" GAN/"+path+"/").run
        }
    */

    val G = gan_Network.select_G(mode)
    val D = gan_Network.select_D(mode)
    println("learning start")

    for (i <- 0 until ln) {
      val start = System.currentTimeMillis
      val z =  makenoise(dn, 100)
      val Z = z._1
      val R = z._2

      val lossG = learning_G(mode, i, dn, G, D, Z)

      val (lossD, counter1, counter2, max1, min1, max2, min2) = learning_D(mode, i, dn, G, D, dtrain, Z)

      val time = System.currentTimeMillis - start

      if ((i + 1) % 100 == 0 || i == ln - 1) {
        println("now save...")
        gan_Network.saves(G, "g_" + mode)
        gan_Network.saves(D, "d_" + mode)

      }

      val log = learning.print_result2(i, time, namelist,
        List(lossG, lossD, counter1 * 100 / (dn / 2), counter2 * 100 / (dn / 2), max1, min1, max2, min2), 2)

      acc_real ::= counter1
      acc_fake ::= counter2
      loss_Dlist ::= lossD
      loss_Glist ::= lossG
      loglist ::= log

    }


    val label = testlabel(dn, 100)
    val y = gan_Network.forwards(G, label)
    Image.write("CGAN/Dropout_C/test" + mode + "_" + ln.toString + ".png", Image.make_image3(y.toArray, 10, 10, 28, 28))

    var path = "CGAN/"
    learning.savetxt1(acc_real,"acc_real_"+mode,path)
    learning.savetxt1(acc_fake,"acc_fake_"+mode,path)
    learning.savetxt1(loss_Dlist,"lossD_"+mode,path)
    learning.savetxt1(loss_Glist,"lossG_"+mode,path)
    learning.savetxt2(loglist,"log_"+mode,path)

  }

  def argmax(a: Array[Double]) = a.indexOf(a.max)

  def onehot(i: Int) = {
    var one = new Array[Double](10)
    one(i) = 1d
    one
  }
  def onehotMatrix(i: Int) = {
    var one = Array.ofDim[Double](10,784)
    for(j <- 0 until 10){
      if(j == i ) one(j) = Array.ofDim[Double](784).map(_=>1d)
      else one(j) = Array.ofDim[Double](784).map(_=>0d)
    }
    one
  }

  def Add_Label(dn: Int, zn: Int, x_size: Int, x: Double, n: Int) = {
    var Z = Array.ofDim[Double](dn, zn + 10)
    var t1 = Array.ofDim[Double](dn, 10)
    var t2 = Array.ofDim[Double](dn, 10, x_size * x_size)

    for (l <- 0 until dn) {
      if (n == 0) {
        val num = l / 10
        t1(l)(num) = x
      } else
        t1(l)(rand.nextInt(t1(l).size)) = x

      for (i <- 0 until t1(l).size) //１データごと後ろにt1のデータを詰めていく
        Z(l)(i) = t1(l)(i)
      for (i <- 0 until zn)
        Z(l)(i + t1(l).size) = rand.nextGaussian()

    }

    for (l <- 0 until dn; i <- 0 until 10) {
      if (t1(l)(i) == x) {
        for (j <- 0 until x_size * x_size)
          t2(l)(i)(j) = x
      }
    }
    (t2, Z)
  }

  def makenoise(dn: Int, xsize: Int) = {
    var seedZ = Array.ofDim[Double](dn, xsize + 10)
    var seedRand = Array.ofDim[Double](dn,10)

    for (i <- 0 until dn) {
      val z = Array.ofDim[Double](xsize).map(_ => rand.nextGaussian())
      val label = onehot(rand.nextInt(10))
      seedZ(i) = z ++ label
      seedRand(i) = label
    }

    (seedZ,seedRand)
  }

  def testlabel(dn: Int, xsize: Int) = {
    val seedZ = Array.ofDim[Double](dn, xsize + 10)
    for (i <- 0 until dn / 10) {
      for (j <- 0 until 10) {
        println(i)
        val z = Array.ofDim[Double](xsize).map(_ => rand.nextGaussian())
        val label = onehot(i)
        seedZ(i * 10 + j) = z ++ label
      }
    }
    seedZ
  }

  def addlabel(xdata: Array[Array[Double]], xlabel: Array[Int]) = {
    val xsize = xdata(0).size
    var returnx = Array.ofDim[Double](xdata.size, xsize * 11)

    for (i <- 0 until xdata.size) {
      //returnx(i) = xdata(i) ++ onehotMatrix(xlabel(i))
    }

    returnx
  }

  def learning_G(mode: String, ln: Int, dn: Int, G: List[Layer], D: List[Layer], Z: Array[Array[Double]]) = {


    var lossG = 0d
    var ys = List[Array[Double]]()


    val y = gan_Network.forwards(G, Z)

    val y2 = gan_Network.forwards(D, y)

    lossG = y2.map(a => math.log(1d - a(0) + 1e-8)).sum

    val d = gan_Network.backwards(D, y2.map(a => a.map(b => (-1d / b))))
    gan_Network.backwards(G, d)

    gan_Network.updates(G)
    gan_Network.resets(D)


    if (ln % 100 == 0 || ln == 9999) {
      Image.write("CGAN/Dropout_C/train" + mode + "_" + ln.toString + ".png", Image.make_image3(y.toArray, 10, 10, 28, 28))
    }

    lossG
  }

  def learning_D(where: String, ln: Int, dn: Int, G: List[Layer], D: List[Layer], dtrain: Array[(Array[Double], Int)], Z: Array[Array[Double]]) = {

    var lossD = 0d
    var fake_counter = 0
    var real_counter = 0
    var max1 = 0d
    var min1 = 0d
    var max2 = 0d
    var min2 = 0d


    val xn = rand.shuffle(dtrain.toList).take(dn / 2)
    val xdata = xn.map(_._1).toArray
    val xlabel = xn.map(_._2).toArray

    val xf = addlabel(xdata, xlabel)


    val y = gan_Network.forwards(D, xf)
    for (i <- 0 until dn / 2) {
      if (y(i)(0) > 0.5) { //本物を見つける
        real_counter += 1
      }
    }
    lossD += y.map(a => -log(a(0) + 1e-8)).sum

    max1 = y.flatten.max
    min1 = y.flatten.min

    val d1 = gan_Network.backwards(D, y.map(a => a.map(b => -1d / (b))))

    gan_Network.updates(D)

    var z = new Array[Array[Double]](dn / 2)
    for (i <- 0 until dn / 2) {
      val z1 = new Array[Double](dn).map(_ => rand.nextGaussian)
      z(i) = z1
    }

    val yy = gan_Network.forwards(D, gan_Network.forwards(G, Z))

    for (i <- 0 until dn / 2) {
      if (yy(i)(0) < 0.5) {
        //偽者を見破る
        fake_counter += 1
      }
    }

    max2 = yy.flatten.max
    min2 = yy.flatten.min

    lossD += yy.map(a => -log(1d - a(0) + 1e-8)).sum

    gan_Network.backwards(D, yy.map(a => a.map(b => 1d / (1d - b))))

    gan_Network.updates(D)
    gan_Network.resets(G)

    (lossD, real_counter.toDouble, fake_counter.toDouble, max1, min1, max2, min2)

  }


}

