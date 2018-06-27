object learning{
  val rand = new scala.util.Random(0)
  //学習回数　かかった時間　誤差１…4　
  //学習時正解データ数　テスト時正解データ数 学習データ数　テストデータ数
  def print_result(
    num:Int,
    time:Double,
    errlist:List[Double],
    countL:Double,
    countT:Double,
    dn:Int,
    tn:Int){
    var printdata = "result:"+num.toString+" - time:"+(time/1000d).toString+"\n"

    for(i <- 0 until errlist.size){
      printdata += "err"+(i+1).toString+":"+errlist(i).toString+"/"
    }

    printdata += "\n"

    if(countL != 0d){
      printdata += " /learning rate: " + (countL/dn * 100).toString
    }

    if(countT != 0d){
      printdata += " /learning rate: " + (countT/tn * 100).toString
      printdata += "\n"
    }

    println(printdata)

  }



   def print_result2(
     num:Int,
     time:Double,
     namelist:List[String],
     outlist:List[Double],
     sp:Int)={
     var printdata = "result:"+num.toString+" - time:"+(time/1000d).toString+"\n"

     for(i <- 0 until outlist.size){
       printdata += namelist(i)+":"+outlist(i).toString+"/"
       if((i+1) % sp == 0 && i!= 0  ){printdata += "\n"}
     }

     println(printdata)

     printdata
   }

def savetxt1(list:List[Double],fn:String,path:String){
    val pathName = path+"/"+fn+".txt"
    val writer =  new java.io.PrintWriter(pathName)
    val ys1 = list.reverse.mkString(",") + "\n"
    writer.write(ys1)
    writer.close()
    println("success "+fn)

  }

  def savetxt2(list:List[String],fn:String,path:String){
    val pathName = path+"/"+fn+".txt"
    val writer =  new java.io.PrintWriter(pathName)
    val ys1 = list.reverse.mkString(",") + "\n"
    writer.write(ys1)
    writer.close()
    println("success "+fn)

  }

  def forwards(layers:List[Layer],x:Array[Double])={
    var temp = x
    for(lay <- layers){
      println(111)
      temp =lay.forward(temp)
      println(222)

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

  def updates(layers:List[Layer])={
    for(lay <- layers){lay.update()}
  }

  def resets(layers:List[Layer]){
    for(lay <- layers){lay.reset()}
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

}


class ML(){
  val rand = new scala.util.Random(0)
  //学習回数　かかった時間　誤差１…4　
  //学習時正解データ数　テスト時正解データ数 学習データ数　テストデータ数
  def print_result(
    num:Int,
    time:Double,
    errlist:List[Double],
    countL:Double,
    countT:Double,
    dn:Int,
    tn:Int){
    var printdata = "result:"+num.toString+" - time:"+(time/1000d).toString+"\n"

    for(i <- 0 until errlist.size){
      printdata += "err"+(i+1).toString+":"+errlist(i).toString+"/"
    }

    printdata += "\n"

    if(countL != 0d){
      printdata += " /learning rate: " + (countL/dn * 100).toString
    }

    if(countT != 0d){
      printdata += " /learning rate: " + (countT/tn * 100).toString
      printdata += "\n"
    }

    println(printdata)

  }



   def print_result2(
     num:Int,
     time:Double,
     namelist:List[String],
     outlist:List[Double],
     sp:Int)={
     var printdata = "result:"+num.toString+" - time:"+(time/1000d).toString+"\n"

     for(i <- 0 until outlist.size){
       printdata += namelist(i)+":"+outlist(i).toString+"/"
       if((i+1) % sp == 0 && i!= 0  ){printdata += "\n"}
     }

     println(printdata)

     printdata
   }

def savetxt1(list:List[Double],fn:String,path:String){
    val pathName = path+"/"+fn+".txt"
    val writer =  new java.io.PrintWriter(pathName)
    val ys1 = list.reverse.mkString(",") + "\n"
    writer.write(ys1)
    writer.close()
    println("success "+fn)

  }

  def savetxt2(list:List[String],fn:String,path:String){
    val pathName = path+"/"+fn+".txt"
    val writer =  new java.io.PrintWriter(pathName)
    val ys1 = list.reverse.mkString(",") + "\n"
    writer.write(ys1)
    writer.close()
    println("success "+fn)

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

  def updates(layers:List[Layer])={
    for(lay <- layers){lay.update()}
  }

  def resets(layers:List[Layer]){
    for(lay <- layers){lay.reset()}
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

}
