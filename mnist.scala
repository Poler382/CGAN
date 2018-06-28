class mnist () {
  val rand=new util.Random(0)
  def load_mnist(dir:String) = {
    def fd(line:String) = line.split(",").map(_.toDouble / 256).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    var train_d = scala.io.Source.fromFile(dir + "/train-d.txt").getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(dir + "/train-t.txt").getLines.map(ft).toArray.head    
    var test_d = scala.io.Source.fromFile(dir + "/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir + "/test-t.txt").getLines.map(ft).toArray.head


    (train_d.zip(train_t), test_d.zip(test_t))
  }

  def load_list(dir:String) = {
    def fd(line:String) = line.split(",").map(_.toDouble / 256 * 2 -1).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    var train_d = scala.io.Source.fromFile(dir + "/train-d.txt").getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(dir + "/train-t.txt").getLines.map(ft).toArray.head    
    var test_d = scala.io.Source.fromFile(dir + "/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir + "/test-t.txt").getLines.map(ft).toArray.head


    (train_d.zip(train_t), test_d.zip(test_t))
  }
 
  def makeimg(im : Array[Double]){
    //サイズを0〜255までに抑え込む

    var returnim = Array.ofDim[Double](28,28,3)

    for(i <- 0 until 28;j <- 0 until 28; k <- 0 until 3){
      returnim(i)(j)(k) = im(i+28*j)
    }
    returnim
  }
  def onehot(a:Int)={
    var t = new Array[Double](10)
    t(a) = 1d
    t
  }

  def argmax(a:Array[Double]) = a.indexOf(a.max)

  def show(dtrain:Array[(Array[Double],Int)],dn:Int)={
    for((x,n) <- rand.shuffle(dtrain.toList).take(dn) ) {
      println(x)
      println(n)
    }
  }
}
