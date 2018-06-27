object gan_Network{

  def select_G(mode:String)={
    val g = mode match {

      case "0" =>{
        val a = new Affine(100,256,0.0002,0.5)
        val b = new BNa(256)
        val c = new ReLU()
        val d = new Affine(256,512,0.0002,0.5)
        val e = new BNa(512)
        val g = new ReLU()
        val h = new Affine(512,1024,0.0002,0.5)
        val i = new BNa(1024)
        val j = new ReLU()
        val k = new Affine(1024,784,0.0002,0.5)
        val l = new Tanh()
        List(a,b,c,d,e,g,h,i,j,k,l)
      }
      case "LRver" =>{
        val a = new Affine(100,256)
        val b = new BNa(256)
        val c = new LeakyReLU(0.2)
        val d = new Affine(256,512)
        val e = new BNa(512)
        val g = new LeakyReLU(0.2)
        val h = new Affine(512,1024)
        val i = new BNa(1024)
        val j = new LeakyReLU(0.2)
        val k = new Affine(1024,784)
        val l = new Tanh()
        List(a,b,c,d,e,g,h,i,j,k,l)
      }
      case "Dropout" =>{
        val a1 = new Affine(100,256,0.0002,0.5)
        val a2 = new BNa(256)
        val a3 = new ReLU()
        val a4 = new Affine(256,512,0.0002,0.5)
        val a5 = new BNa(512)
        val a6 = new ReLU()
        val a7 = new Affine(512,1024,0.0002,0.5)
        val b1 = new Dropout(0.5)
        val b2 = new BNa(1024)
        val b3 = new ReLU()
        val b4 = new Affine(1024,784,0.0002,0.5)
        val b5 = new Tanh()
        List(a1,a2,a3,a4,a5,a6,a7,b1,b2,b3,b4,b5)
      }
      case "Dropout_C" =>{
        val a1 = new Affine(110,256,0.0002,0.5)
        val a2 = new BNa(256)
        val a3 = new ReLU()
        val a4 = new Affine(256,512,0.0002,0.5)
        val a5 = new BNa(512)
        val a6 = new ReLU()
        val a7 = new Affine(512,1024,0.0002,0.5)
        val b1 = new Dropout(0.5)
        val b2 = new BNa(1024)
        val b3 = new ReLU()
        val b4 = new Affine(1024,784,0.0002,0.5)
        val b5 = new Tanh()
        List(a1,a2,a3,a4,a5,a6,a7,b1,b2,b3,b4,b5)
      }



      case "NonBacth" =>{
        val a = new Affine(100,256)
        val b = new ReLU()
        val c = new Affine(256,512)
        val d = new ReLU()
        val e = new Affine(512,1024)
        val f = new ReLU()
        val g = new Affine(1024,784)
        val h = new Tanh()
        List(a,b,c,d,e,f,g,h)
      }

      case "test" =>{
        val a = new Affine(1,1)
        val b = new Tanh()
        List(a,b)
      }
    }

    g
  }


  def select_D(mode:String)={
    val d = mode match {
      case "0" => {
        val a = new Affine(784,512,0.0002,0.5)
        val b = new ReLU()
        val c = new Affine(512,256,0.0002,0.5)
        val d = new ReLU()
        val e = new Affine(256,1,0.0002,0.5)
        val f = new Sigmoid()
        List(a,b,c,d,e,f)
      }
      case "Dropout" => {
        val a = new Affine(784,512,0.0002,0.5)
        val b = new ReLU()
        val c = new Affine(512,256,0.0002,0.5)
        val d = new ReLU()
        val e = new Affine(256,1,0.0002,0.5)
        val f = new Sigmoid()
        List(a,b,c,d,e,f)
      }

      case "Dropout_C" => {
        val a = new Affine(784+784*10,1024,0.0002,0.5)
        val b = new ReLU()
        val c = new Affine(1024,512,0.0002,0.5)
        val d = new ReLU()
        val e = new Affine(512,1,0.0002,0.5)
        val f = new Sigmoid()
        List(a,b,c,d,e,f)
      }

      case "LRver" => {
        val a = new Affine(784,1024)
        val b = new LeakyReLU(0.2)
        val c = new Affine(1024,512)
        val d = new LeakyReLU(0.2)
        val e = new Affine(512,1)
        val f = new Sigmoid()
        List(a,b,c,d,e,f)
      }

      case "NonBacth" => {
        val a = new Affine(784,1024)
        val b = new ReLU()
        val c = new Affine(1024,512)
        val d = new ReLU()
        val e = new Affine(512,1)
        val f = new Sigmoid()
        List(a,b,c,d,e,f)
      }
      case "test"=>{
        val a= new Affine(1,1)
        val b = new Sigmoid()
        List(a,b)
      }
    }
    d
  }


  def forwards(layers:List[Layer],x:Array[Double])={
    var temp = x
    for(lay <- layers){
      temp =lay.forward(temp)

    }
    temp
  }

  def backwards(layers:List[Layer],x:Array[Double])={
    var d = x
    for(lay <- layers.reverse){d = lay.backward(d)}
    d
  }

  def forwards(layers:List[Layer],x:Array[Array[Double]]): Array[Array[Double]]={
    var temp = x
    for(lay <- layers){
      temp =lay.forward(temp)
    }
    temp
  }

  def backwards(layers:List[Layer],x:Array[Array[Double]]): Array[Array[Double]]={
    var d = x
    for(lay <- layers.reverse){
      d = lay.backward(d)
    }
    d
  }

  def saves(layers:List[Layer],fn:String){

    for(i <- 0 until layers.size){
      layers(i).save("biasdata/"+fn+"_"+i.toString)
    }

  }

  def load(layers:List[Layer],fn:String){
    for(i <- 0 until layers.size ){
      layers(i).load("biasdata/"+fn+"_"+i.toString)
    }


  }



  def updates(layers:List[Layer])={
    for(lay <- layers){lay.update()}
  }

  def resets(layers:List[Layer]){
    for(lay <- layers){lay.reset()}
  }

}
