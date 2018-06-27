import java.util.Date

import scala.math._

object GAN{
  val rand=new util.Random(0)
  val namelist =List("lossG","lossD","real rate","fake rate",
    "max1","min1","max2","min2")

  val filepath = List(
    "src/main/scala/generalAdversalial.scala",
    "src/main/scala/generalAdversalial_layer.scala",
    "src/main/scala/Network.scala")
  def main(args:Array[String]){

    val mode = args(0)
    val ln   = args(1).toInt
    val dn   = args(2).toInt
    val where = args(3)

    val mn = new mnist()

    val (dtrain, dtest) = "lab" match {
      case "home" =>{
        mn.load_mnist("C:/Users/poler/Documents/python/share/mnist")
      }
      case "lab" =>{
        mn.load_mnist("/home/share/fashion-mnist")
      }
    }

    println("finish load")


    val G = gan_Network.select_G(mode)
    val D = gan_Network.select_D(mode)
    
    var acc_real   = List[Double]()
    var acc_fake   = List[Double]()
    var loss_Dlist = List[Double]()
    var loss_Glist = List[Double]()
    var loglist    = List[String]()

    val date = "-%tm%<td-%<tHh" format new Date
    sys.process.Process("mkdir GAN/"+mode+date).run

    val path = mode+date

    for(file <-filepath){
      sys.process.Process("cp "+file+" GAN/"+path+"/").run
    }


    println("learning start")
  
    for(i <- 0 until ln){
      val start = System.currentTimeMillis
      val lossG = learning_G(mode,i,dn,G,D,path)

      val (lossD,counter1,counter2,max1,min1,max2,min2)= learning_D(mode,i,dn,G,D,dtrain,path)
      
      val time =  System.currentTimeMillis - start
      
      if((i+1) % 100 == 0 || i == ln-1){
        println("now save...")
        gan_Network.saves(G,"g_"+mode)
        gan_Network.saves(D,"d_"+mode)
      }

      val log =learning.print_result2(i,time,namelist,
        List(lossG,lossD,counter1*100/(dn/2),counter2*100/(dn/2),max1,min1,max2,min2),2)
      
      acc_real ::= counter1
      acc_fake ::= counter2
      loss_Dlist ::= lossD
      loss_Glist ::= lossG
      loglist ::= log

    }

    savetxt1(acc_real,"acc_real_"+mode,path)
    savetxt1(acc_fake,"acc_fake_"+mode,path)
    savetxt1(loss_Dlist,"lossD_"+mode,path)
    savetxt1(loss_Glist,"lossG_"+mode,path)
    savetxt2(loglist,"log_"+mode,path)


  }

  //
  def learning_G(mode:String,ln:Int,dn:Int,G:List[Layer],D:List[Layer],date:String)={


    var lossG = 0d
    var ys  = List[Array[Double]]()
    
    var z = new Array[Array[Double]](100)
    for(i <- 0 until 100){// make noise
      val t = new Array[Double](100).map(_ => rand.nextGaussian)
      z(i) = t
    }

    val y =  gan_Network.forwards(G,z)

    val y2 =  gan_Network.forwards(D,y)
    lossG = y2.map(a=> math.log(1d-a(0)+1e-8)).sum

    val d  = gan_Network.backwards(D,y2.map(a=>a.map(b =>(-1d/b))))
    gan_Network.backwards(G,d)

    gan_Network.updates(G)
    gan_Network.resets(D)
    if(ln % 100 == 0){
      Image.write("GAN/"+date+"/train"+mode+"_"+ln.toString+".png",Image.make_image3(y.toArray,10,10,28,28))
    }

    lossG
  }

  def learning_D(where:String,ln:Int,dn:Int,G:List[Layer],D:List[Layer],dtrain:Array[(Array[Double], Int)],date:String)={

    var lossD =0d
    var fake_counter  = 0
    var real_counter  = 0
    var max1 = 0d
    var min1 = 0d
    var max2 = 0d
    var min2 = 0d

  
    val xn = rand.shuffle(dtrain.toList).take(dn/2)
    val xs = xn.map(_._1).toArray
   
    val xf = xs.map(_.map(a => a*2 - 1d))


    val y =  gan_Network.forwards(D,xf)
    for(i <- 0 until dn/2){
      if(y(i)(0) > 0.5){ //本物を見つける
        real_counter += 1
      }
    }
    lossD += y.map(a => -log( a(0) + 1e-8)).sum

    max1 = y.flatten.max
    min1 = y.flatten.min

    val d1 = gan_Network.backwards(D,y.map(a => a.map(b => -1d/(b))))

    gan_Network.updates(D)

    var z = new Array[Array[Double]](dn/2)
    for(i <- 0 until dn/2){
      val z1 = new Array[Double](dn).map(_ => rand.nextGaussian)
      z(i) = z1
    }

    val yy =  gan_Network.forwards(D, gan_Network.forwards(G,z))

    for(i <- 0 until dn/2){
      if(yy(i)(0) < 0.5){//偽者を見破る
        fake_counter += 1
      }
     // println(yy(i)(0))
    }

    max2 = yy.flatten.max
    min2 = yy.flatten.min

    lossD += yy.map(a => -log(1d - a(0) + 1e-8)).sum

    val d2 = gan_Network.backwards(D,yy.map(a => a.map(b => 1d/(1d-b))))

    gan_Network.updates(D)
    gan_Network.resets(G)

    (lossD,real_counter.toDouble,fake_counter.toDouble,max1,min1,max2,min2)

  }

  def savetxt1(list:List[Double],fn:String,path:String){
    val pathName = "GAN/"+path+"/"+fn+".txt"
    val writer =  new java.io.PrintWriter(pathName)
    val ys1 = list.reverse.mkString(",") + "\n"
    writer.write(ys1)
    writer.close()
    println("success "+fn)

  }

  def savetxt2(list:List[String],fn:String,path:String){
    val pathName = "GAN/"+path+"/"+fn+".txt"
    val writer =  new java.io.PrintWriter(pathName)
    val ys1 = list.reverse.mkString(",") + "\n"
    writer.write(ys1)
    writer.close()
    println("success "+fn)

  }


}
