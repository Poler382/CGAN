import breeze.linalg._
import java.util.Date

object CGAN{
  val rand = new scala.util.Random(0)
   val namelist =List("lossG","lossD","real rate","fake rate",
    "max1","min1","max2","min2")

  val filepath = List(
    "src/main/scala/generalAdversalial.scala",
    "src/main/scala/generalAdversalial_layer.scala",
    "src/main/scala/Network.scala")
 

  def load_list(dir:String) = {
    def fd(line:String) = line.split(",").map(_.toDouble / 256*2-1).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    val train_d = scala.io.Source.fromFile(dir + "/train-d.txt").getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(dir + "/train-t.txt").getLines.map(ft).toArray.head
    val test_d = scala.io.Source.fromFile(dir + "/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir + "/test-t.txt").getLines.map(ft).toArray.head
    (train_d.zip(train_t), test_d.zip(test_t))
  }

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

    

    val zn = 100
    val af1 = new Affine(100+10,256)
    val af2 = new Affine(256,512)
    val af3 = new Affine(512,1024)
    val af4 = new Affine(1024,784)
    val af5 = new Affine(784+7840,1024)

    val af6 = new Affine(1024,512)
    val af7 = new Affine(512,1)

    val R1 = new ReLU()
    val R2 = new ReLU()
    val R3 = new ReLU()
    val R4 = new ReLU()
    val R5 = new ReLU()

    val B1 = new BNa(256)
    val B2 = new BNa(512)
    val B3 = new BNa(1024)

    val tan1 = new Tanh()
    val sig1 = new Sigmoid()


    val G = List(af1,R1,B1,af2,R2,B2,af3,R3,B3,af4,tan1)
    val D = List(af5,R4,af6,R5,af7,sig1)

    var acc_real   = List[Double]()
    var acc_fake   = List[Double]()
    var loss_Dlist = List[Double]()
    var loss_Glist = List[Double]()
    var loglist    = List[String]()

    var error_d = new Array[Double](ln)
    var error_g = new Array[Double](ln)
    var corect1 = new Array[Double](ln)
    var corect2 = new Array[Double](ln)


    val date = "-%tm%<td-%<tHh" format new Date
    sys.process.Process("mkdir GAN/"+mode+date).run

    val path = mode+date

    for(file <-filepath){
      sys.process.Process("cp "+file+" GAN/"+path+"/").run
    }


    println("learning start")
  

    for(in <- 0 until ln){
      val start = System.currentTimeMillis

      val xn = rand.shuffle(dtrain.toList).take(dn)
      val x = xn.map(_._1).toArray
      val n = xn.map(_._2).toArray

      val x_size = 28
      val x_num = 1d

      var tmp1 = Add_Label(dn,zn,x_size,x_num,1)
      var t2 = tmp1._1
      var Z = tmp1._2

      var t3 = Array.ofDim[Double](dn,10,x_size*x_size)
      for(l<-0 until dn ; i<-0 until 10){

        if(i == n(l)){
          for(j<-0 until x_size*x_size)
            t3(l)(i)(j) = x_num
        }
      }

      /////////////////////LD上の計算/////////////
      val y1 = learning.forwards(G,Z)

      val z2 = Array.ofDim[Double](dn,11,x_size*x_size)
      for(l<-0 until dn){
        z2(l)(0) = y1(l)
        for(i<-0 until 10)
          z2(l)(i+1) = t2(l)(i) 
      }

      val y2 = learning.forwards(D,z2.map(_.flatten))

      val LD1 = y2.map(a => Math.log((1d - a(0)) + 0.00000001) ).sum
      val d2 = learning.backwards(D.reverse,y2.map(a=>a.map(b => 1d/(1d-b))))
      learning.updates(D)
      learning.resets(G)

      for(i <-0 until dn){//偽判定
        if(y2(i)(0) < 0.5)
          corect1(in) += 1d
      }

      error_d(in) += LD1

     /////////////////LD下の計算////////////////
      val x2 = Array.ofDim[Double](dn,11,x_size*x_size)

      for(l<-0 until dn){
        x2(l)(0) = x(l)
        for(i<-0 until 10)
          x2(l)(i+1) = t3(l)(i)
      }

      val y0 = learning.forwards(D,x2.map(_.flatten))

      for(i <-0 until dn){//本判定
        if(y0(i)(0) > 0.5)
          corect2(in) += 1d
      }

      val LD2 = y0.map(a => Math.log( a(0) + 0.00000001)).sum
      val d1 = learning.backwards(D.reverse,y0.map(a => a.map(b => -1d/b)))
      learning.updates(D)

      error_d(in) += LD2
      error_d(in) = -1d * error_d(in)/dn

     /////////////LGの計算//////////////////

      val g = learning.forwards(G,Z)
      
      val g1 = Array.ofDim[Double](dn,11,x_size*x_size)
      for(l<-0 until dn){
        g1(l)(0) = g(l)
        for(i<-0 until 10)
          g1(l)(i+1) = t2(l)(i)
      }

      val g2 = learning.forwards(D,g1.map(_.flatten))
      val LG = g2.map(a=> Math.log((1d-a(0)) +0.00000001)).sum

      val d3 = learning.backwards(D.reverse,g2.map(_.map(a=>(-1d/a)))).map(_.take(x_size*x_size))

      val d4 = learning.backwards(G.reverse,d3)

      learning.updates(G)
      learning.resets(D)

      error_g(in) = -1d * LG/dn

      /////////////画像生成///////////////////
      if((in+1)%10 == 0 || in == ln-1){
        tmp1 = Add_Label(dn,zn,x_size,x_num,0)
        t2 = tmp1._1
        Z = tmp1._2

        val g = learning.forwards(G,Z)
        val pw = new java.io.PrintWriter("test-d.txt")

        for(i<-0 until dn){
          if(in == ln-1){
            pw.print(g(i).map(a=>((a+1)/2*255).toInt).mkString(",") + "\n")
          }
          val tmp =change(g(i),x_size)
          makepng(tmp,i,in+1)
        }
        pw.close()
        learning.resets(G)
      }

      //////////////////print//////////////
      println(in+1 + "回目")
      println("偽者判別率 : "  + corect1(in)/dn*100 +"%　本物判別率 : " + corect2(in)/dn*100 + "%")
      println("D : " + error_d(in) + " G : " + error_g(in))

      if((in+1)%100 == 0){
        gan_Network.saves(G,"g_"+mode)
        gan_Network.saves(D,"d_"+mode)
      }

    }

    ////////////////print-ALL////////////////////
    println("----------------偽者判別率------------------")
    for(i<-0 until ln){
      print(corect1(i)/dn*100 + ",")
    }
    println()

    println("----------------本物判別率------------------")
    for(i<-0 until ln){
      print(corect2(i)/dn*100 + ",")
    }
    println()

    println("--------------------D----------------------")
    for(i<-0 until ln){
      print(error_d(i)/dn*100 + ",")
    }
    println()

    println("--------------------G----------------------")
    for(i<-0 until ln){
      print(error_g(i)/dn*100 + ",")
    }
    println()

    //savef("png",FL)
  }

  def argmax(a:Array[Double]) = a.indexOf(a.max)

  def change(a:Array[Double],s:Int) = {

    var x = new Array[Double](a.size)
    var AX = Array.ofDim[Int](s,s,3)
    var x1 = new Array[Int](a.size)
    var x2 = Array.ofDim[Int](s,s)
    var xs = List[Int]()

    for(i<-0 until a.size)
      x(i) = a(i)

    for(i<-0 until x.size){
      x(i) = (x(i)+1)/2
      x(i) = x(i)*255

      if(x(i) > 255) x(i) = 255
      else if(x(i) < 0) x(i) = 0

      xs ::= x(i).toInt
    }

    x1 = xs.reverse.toArray

    for(i<-0 until s ; j<-0 until s){
      x2(i)(j) = x1(i*s+j)
    }

    AX = x2.map(_.map(a=>Array(a,a,a)))

    AX
  }

  def loadf(nt:String,fL:List[Layer]){
    var count = 1
    for(i <- fL){
      if(count<10)
        i.load("CGAN-0" + count + "-" + nt)
      else
        i.load("CGAN-" + count + "-" + nt)
      count += 1
    }
  }

  def savef(nt:String,fL:List[Layer]){
    var count = 1
    for(i <- fL){
      if(count<10)
        i.save("CGAN-0" + count + "-" + nt)
      else
        i.save("CGAN-" + count + "-" + nt)
      count += 1
    }
  }

  def makepng(z:Array[Array[Array[Int]]],count:Int,in:Int){

    if(count<10)
      Image.write( "CGAN/00" + count +"-" + in + "-" + "CGAN_List.png",z)
    else if(count<100)
      Image.write( "CGAN/0" + count + "-" + in + "-" + "CGAN_List.png",z)
    //else if(count<1000)
      //Image.write("0" + count + "AEc2-1_d.png",z)
    else
      Image.write("CGAN"+count + "-" + in + "-" + "CGAN_List.png",z)

  }

  def Add_Label(dn:Int,zn:Int,x_size:Int,x:Double,n:Int) = {
    var Z = Array.ofDim[Double](dn,zn+10)
    var t1 = Array.ofDim[Double](dn,10)
    var t2 = Array.ofDim[Double](dn,10,x_size*x_size)

    for(l<-0 until dn){
      if(n==0){
        val num = l/10
        t1(l)(num) = x 
      }else
        t1(l)(rand.nextInt(t1(l).size)) = x

      for(i<-0 until t1(l).size)//１データごと後ろにt1のデータを詰めていく
        Z(l)(i) = t1(l)(i)
      for(i<-0 until zn)
        Z(l)(i+t1(l).size) = rand.nextGaussian()
     
    }

    for(l<-0 until dn ; i<-0 until 10){
      if(t1(l)(i) == x ){
        for(j<-0 until x_size*x_size)
          t2(l)(i)(j) = x
      }
    }
    (t2,Z)
  }
}

