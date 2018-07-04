import java.util.Date
import breeze.linalg._

import math._

object CGAN {
  val rand = new scala.util.Random(0)
  val namelist = List("lossG", "lossD", "real rate", "fake rate",
    "max1", "min1", "max2", "min2")

  val filepath = List(
    "src/main/scala/cgeneral.scala",
    "src/main/scala/generalAdversalial_layer.scala",
    "src/main/scala/Network.scala")

  val white = Array.ofDim[Double](784).map(_ => 1d)
  val black = Array.ofDim[Double](784).map(_ => 0d)


  def main(args: Array[String]) {
    val mode = args(0)
    val ln = args(1).toInt
    val dn = args(2).toInt
    val where = args(3)

    val mn = new mnist()
    val L = new ML()

    val (dtrain, dtest) = where match {
      case "home" => {
        mn.load_list("C:/Users/poler/Documents/python/share/fashion-mnist")
      }
      case "home-mnist" => {
        mn.load_list("C:/Users/poler/Documents/python/share/mnist")
      }
      case "lab" => {
        mn.load_list("/home/share/fashion-mnist")
      }
    }

    println("finish load")

    var acc_real = List[Double]()
    var acc_fake = List[Double]()
    var acc_imgae = List[Double]()
    var loss_Dlist = List[Double]()
    var loss_Glist = List[Double]()
    var loglist = List[String]()


    val date = "-%tm%<td-%<tHh" format new Date
    //sys.process.Process("mkdir CGAN/"+mode+date).run

    val path = "CGAN/" + mode + date
    print(path)
    for (file <- filepath) {
      //sys.process.Process("cp "+file+" "+path+"/").run
    }

   val G = gan_Network.select_G(mode)
    val D = gan_Network.select_D(mode)

    println("learning start")

  //  L.load(G, "g_" + mode)
   // L.load(D, "d_" + mode)
    for (i <- 0 until ln) {
      val start = System.currentTimeMillis
      val z = makenoise(dn, 100)
      val Z = z._1
      val R = z._2

      val lossG = learning_G(mode, i, dn, G, D, Z, R)

      val (lossD, counter1, counter2, max1, min1, max2, min2) = learning_D(mode, i, dn, G, D, dtrain, Z, R)
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

      acc_imgae ::= create_image(mode, i, dn,where, G, D, path)
    }

    learning.savetxt1(acc_real, "acc_real_" + mode, path)
    learning.savetxt1(acc_fake, "acc_fake_" + mode, path)
    learning.savetxt1(loss_Dlist, "lossD_" + mode, path)
    learning.savetxt1(loss_Glist, "lossG_" + mode, path)
    learning.savetxt2(loglist, "log_" + mode, path)
    learning.savetxt1(acc_imgae, "accImage_" + mode, path)
  }



  def create_image(mode: String,ln: Int, dn: Int,where:String,
                   G: List[Layer], D: List[Layer],path:String )={

    var s = Array.ofDim[Double](100,110)
    var log = 0d
    for(i <- 0 until 10){
      for(j <- 0 until 10){
        s(i*10+j) =  Array.ofDim[Double](100).map(_=>rand.nextGaussian()) ++ onehot(i)
      }
    }

    val y = gan_Network.forwards(G,s)

    if (ln % 100 == 0 ) {
      Image.write(path+"/sort" + mode + "_" + ln.toString + ".png", Image.make_image3(y.toArray, 10, 10, 28, 28))

      val pathName = path+"/txt/sort" + mode + "_" + ln.toString + ".txt"
      val writer = new java.io.PrintWriter(pathName)
      var ys = ""
      for(line <- y){
        ys += line.mkString(",")+"\n"
      }
      writer.write(ys)
      writer.close()
      println("success...text and image")
    }
    if (ln % 10 == 0) {
      val trains = du(mode, ln, dn, G, path)
      //log = trainCGAN.cganConect(ln, dn, mode, where ,trains)
    }

    log

  }


  def du(mode: String, ln: Int, dn: Int,
         G: List[Layer], path: String) = {
    var fm = Array.ofDim[Double](100, 110)
    var seed = new Array[Int](100)
    for (i <- 0 until 100) {
      seed(i) = rand.nextInt(10)
      fm(i) = Array.ofDim[Double](100).map(_ => rand.nextGaussian()) ++ onehot(seed(i))
    }
    val y = gan_Network.forwards(G, fm)
    y.zip(seed)
  }

  def argmax(a: Array[Double]) = a.indexOf(a.max)

  def onehot(i: Int) = {
    var one = new Array[Double](10)
    one(i) = 1d
    one
  }

  def onehotMatrix(i: Int) = {
    var one = Array.ofDim[Double](10, 784)
    for (j <- 0 until 10) {
      if (j == i) one(j) = Array.ofDim[Double](784).map(_ => 1d)
      else one(j) = Array.ofDim[Double](784).map(_ => 0d)
    }
    one.flatten
  }

  def makenoise(dn: Int, xsize: Int) = {
    var seedZ = Array.ofDim[Double](dn, xsize)
    var seedRand = Array.ofDim[Int](dn)

    for (i <- 0 until dn) {
      seedZ(i) = Array.ofDim[Double](xsize).map(_ => rand.nextGaussian())
      seedRand(i) = rand.nextInt(10)
    }

    (seedZ, seedRand)
  }

  def addlabel(xdata: Array[Array[Double]], xlabel: Array[Int]) = {
    val xsize = xdata(0).size
    var returnx = Array.ofDim[Double](xdata.size, xsize * 11)

    for (i <- 0 until xdata.size) {
      returnx(i) = xdata(i) ++ onehotMatrix(xlabel(i))
    }

    returnx
  }

  def addonehot(Z: Array[Array[Double]], rm: Array[Int]) = {
    val xsize = Z.size
    var returnx = Array.ofDim[Double](xsize, 784 + 10)

    for (i <- 0 until xsize) {
      returnx(i) = Z(i) ++ onehot(rm(i))
    }

    returnx
  }

  def learning_G(mode: String, ln: Int, dn: Int,
                 G: List[Layer], D: List[Layer], Z: Array[Array[Double]], R: Array[Int]) = {


    var lossG = 0d
    var ys = List[Array[Double]]()

    val mm = addonehot(Z, R)

    val y = gan_Network.forwards(G, mm)

    val y2 = gan_Network.forwards(D, addlabel(y, R))

    lossG = y2.map(a => math.log(1d - a(0) + 1e-8)).sum

    val d = gan_Network.backwards(D, y2.map(a => a.map(b => (-1d / b))))

    gan_Network.backwards(G, d.map(_.take(784)))

    gan_Network.updates(G)
    gan_Network.resets(D)

    lossG
  }


  def learning_D(
                  where: String, ln: Int, dn: Int,
                  G: List[Layer], D: List[Layer],
                  dtrain: Array[(Array[Double], Int)],
                  Z: Array[Array[Double]], R: Array[Int]) = {

    var lossD = 0d
    var fake_counter = 0
    var real_counter = 0
    var max1 = 0d;
    var min1 = 0d;
    var max2 = 0d;
    var min2 = 0d;

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

    val oo = addlabel(Z, R)

    val im = gan_Network.forwards(G, oo)

    val yy = gan_Network.forwards(D, addlabel(im, R))


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


object trainCGAN {
  val rand = new scala.util.Random(0)
  val namelist = List("err1", "err2", "a_count", "c_count")

  def main(args: Array[String]) {
    val mode = args(0)
    val ln = args(1).toInt
    val dn = args(2).toInt
    val where = args(3)
    val mn = new mnist()

    val (dtrain, dtest) = where match {
      case "home" => {
        mn.load_list("C:/Users/poler/Documents/python/share/fashion-mnist")
      }
      case "home-mnist" => {
        mn.load_list("C:/Users/poler/Documents/python/share/mnist")
      }
      case "lab" => {
        mn.load_list("/home/share/fashion-mnist")
      }
    }


    println("finish load")


    val a = new Convolution2(3, 28, 28, 1, 10)
    val b = new ReLU()
    val c = new Pooling(2, 10, 26, 26)
    val d = new Convolution2(4, 13, 13, 10, 10)
    val e = new ReLU()
    val f = new Pooling(2, 10, 10, 10)
    val g = new Affine(250, 10)
    val h = new softMax()
    val Network1 = List(a, b, c, d, e, f, g, h)

    val a1 = new Affine(784, 256)
    val a2 = new ReLU()
    val a3 = new Affine(256, 10)

    val Network2 = List(a1, a2, a3)

    trainNet(ln, dn, mode, dtrain, dtest, Network2)

  }

  def cganConect(ln: Int, dn: Int, mode: String, where:String,
                 dtest: Array[(Array[Double], Int)]) = {


    val a = new Convolution(3, 28, 28, 1, 10)
    val b = new ReLU()
    val c = new Pooling(2, 10, 26, 26)
    val d = new Convolution(4, 13, 13, 10, 10)
    val e = new ReLU()
    val f = new Pooling(2, 10, 10, 10)
    val g = new Affine(250, 10)

    val a1 = new Affine(784, 256)
    val a2 = new ReLU()
    val a3 = new Affine(256, 10)

    val Network = List(a1, a2, a3)



    val Network1 = List(a, b, c, d, e, f, g)


    val L = new ML()
    var err2 = 0d
    var c_count = 0
    L.load(Network, "cganNet_" + "ARA"+where)

    for ((x, n) <- dtest.take(dn)) {
      var y = L.forwards(Network, x)
      err2 += sub(y, onehot(n)).map(a => a * a).sum

      if (argmax(y) == n) {
        c_count += 1
      }
    }
    println("###############################################")
    L.print_result2(ln, 10, List("err2", "c_count"),
      List(err2 / dn * 100, c_count / dn * 100), 2)
    println("###############################################")
    val log = c_count / dn * 100
    log
  }

  def trainNet(
                ln: Int, dn: Int, mode: String,
                dtrain: Array[(Array[Double], Int)],
                dtest: Array[(Array[Double], Int)], layers: List[Layer]) {

    val L = new ML()

    for (i <- 0 until ln) {

      var a_count = 0d
      var err1 = 0d
      var err2 = 0d
      var start_l = System.currentTimeMillis

      for ((x, n) <- rand.shuffle(dtrain.toList).take(dn)) {
        var y = L.forwards(layers, x)

        L.backwards(layers, sub(y, onehot(n)))
        L.updates(layers)

        err1 += sub(onehot(argmax(y)), onehot(n)).map(a => a * a).sum

        if (argmax(y) == n) {
          a_count += 1
        }
      }
      var c_count = 0d

      //test zone
      for ((x, n) <- dtest.take(dn)) {
        var y = L.forwards(layers, x)
        err2 += sub(y, onehot(n)).map(a => a * a).sum

        if (argmax(y) == n) {
          c_count += 1
        }
      }
      L.resets(layers)
      var time = System.currentTimeMillis - start_l
      //  println(err1,dn,a_count,c_count)
      val log = L.print_result2(i, time, namelist,
        List(err1 / dn * 100, err2 / dn * 100, a_count / dn * 100, c_count / dn * 100), 2)

      if ((i + 1) % 10 == 0 || i == ln - 1) {
        L.saves(layers, "cganNet_ARA"+where )
        println("                          now save...")
      }
    }
  }


  def onehot(a: Int) = {
    var t = new Array[Double](10)
    t(a) = 1d
    t
  }

  def argmax(a: Array[Double]) = a.indexOf(a.max)

  def sub(a: Array[Double], b: Array[Double]) = {
    var sub = Array.ofDim[Double](a.size)
    for (i <- 0 until a.size) {
      sub(i) = a(i) - b(i)
    }
    sub
  }


}
