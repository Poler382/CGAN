import math._
import breeze.linalg._

/*
****************************************************************

convolution
unsumpling
stridPanding
subsumpling
pooling
Affine
Sigmoid
Tanh
Dropout
ReLU
LeakyReLU
ZeroPadding
Adam
softPlus
BNa


 */

abstract class Layer {
  type T = Double
  var is_test = false
  def forward(x:Array[T]) : Array[T]
  def backward(x:Array[T]) : Array[T]
  def forward(xs:Array[Array[T]]) : Array[Array[T]] = {
    xs.map(forward)
  }
  def backward(ds:Array[Array[T]]) : Array[Array[T]] = {
    ds.reverse.map(backward).reverse
  }
  def update() : Unit
  def reset() : Unit
  def save(fn:String) {}
  def load(fn:String) {}
}

class Convolution(
  val KW:Int,
  val IH:Int,
  val IW:Int,
  val IC:Int,
  val OC:Int,
  val eps:Double = 0.001,
  val rho1:Double = 0.9,
  val rho2:Double = 0.999
  ) extends Layer {
  val OH = IH - KW + 1
  val OW = IW - KW + 1
  var Vs = List[Array[T]]()
  var K = Array.ofDim[T](OC * IC * KW * KW)
  var n = 0

  def iindex(i:Int, j:Int, k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH * OW + j * OW + k
  def kindex(i:Int, j:Int, k:Int, l:Int) = i * IC * KW * KW + j * KW * KW + k * KW + l
  def forward(V:Array[T]) = {
    Vs ::= V
    val Z = Array.ofDim[T](OC * OH * OW)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      for(l <- 0 until IC; m <- 0 until KW; n <- 0 until KW) {
        Z(oindex(i,j,k)) += V(iindex(l,j+m,k+n)) * K(kindex(i,l,m,n))
      }
    }
    Z
  }

  var dK = Array.ofDim[T](K.size)
  def backward(G:Array[T]) = {
    n += 1
    val V = Vs.head
    Vs = Vs.tail

    for(i <- 0 until OC; j <- 0 until IC; k <- 0 until KW; l <- 0 until KW) {
      for(m <- 0 until OH; n <- 0 until OW) {
        dK(kindex(i,j,k,l)) += V(iindex(j,m+k,n+l)) * G(oindex(i,m,n))
      }
    }

    val D = Array.ofDim[T](V.size)
    for(i <- 0 until IC; j <- 0 until IH; k <- 0 until IW) {
      for(l <- 0 until OH; m <- 0 until KW if l + m == j) {
        for(n <- 0 until OW; p <- 0 until KW if n + p == k) {
          for(q <- 0 until OC) {
            D(iindex(i,j,k)) += K(kindex(q,i,m,p)) * G(oindex(q,l,n))
          }
        }
      }
    }

    D
  }

  def update() {
    for(i <- 0 until dK.size) {
      dK(i) /= n
    }
    update_adam()
    reset()
  }

  var lr = 0.001
  def update_sgd() {
    for(i <- 0 until K.size) {
      K(i) -= lr * dK(i)
    }
  }

  var adam = new Adam(K.size,eps,rho1,rho2)
  def update_adam() {
    adam.update(K,dK)
  }

  def reset() {
    Vs = List()
    for(i <- 0 until dK.size) {
      dK(i) = 0d
    }
    n = 0
  }

  override def save(fn:String) {
    val pw = new java.io.PrintWriter(fn)
    for(i <- 0 until K.size) {
      pw.write(K(i).toString)
      if(i != K.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")
    pw.close()
  }

  override def load(fn:String) {
    val f = scala.io.Source.fromFile(fn).getLines.toArray
    K = f(0).split(",").map(_.toDouble).toArray
  }
}

class Upsampling(val IC:Int, val IH:Int, val IW:Int, val BH:Int, val BW:Int) extends Layer {
  val OH = IH * BH
  val OW = IW * BW
  val OC = IC

  def iindex(i:Int, j:Int, k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH * OW + j * OW + k
  
  def forward(X:Array[T]) = {
    val Z = Array.ofDim[T](OC * OH * OW)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      Z(oindex(i,j,k)) = X(iindex(i,j/BH,k/BW))
    }
    Z
  }

  def backward(d:Array[T]) = {
    val D = Array.ofDim[T](IC * IH * IW)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
        D(iindex(i,j/BH,k/BW)) += d(oindex(i,j,k))
    }
    D
  }

  def update() {}
  def reset() {}
}

class StridedPadding(val IC:Int, val IH:Int, val IW:Int, val S:Int) extends Layer {
  val OH = S * IH + S - 1
  val OW = S * IW + S - 1
  val OC = IC

  def iindex(i:Int, j:Int, k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH * OW + j * OW + k
  
  def forward(X:Array[T]) = {
    val Z = Array.ofDim[T](OC * OH * OW)
    for(i <- 0 until IC; j <- 0 until IH; k <- 0 until IW) {
      Z(oindex(i,S-1+j*S,S-1+k*S)) = X(iindex(i,j,k))
    }
    Z
  }

  def backward(d:Array[T]) = {
    val D = Array.ofDim[T](IC * IH * IW)
    for(i <- 0 until IC; j <- 0 until IH; k <- 0 until IW) {
        D(iindex(i,j,k)) = d(oindex(i,S-1+j*S,S-1+k*S))
    }
    D
  }

  def update() {}
  def reset() {}
}

class Subsampling(val IC:Int, val IH:Int, val IW:Int, val BH:Int, val BW:Int) extends Layer {
  val OH = IH / BH
  val OW = IW / BW
  val OC = IC

  def iindex(i:Int, j:Int, k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH * OW + j * OW + k
  
  def forward(X:Array[T]) = {
    val Z = Array.ofDim[T](OC * OH * OW)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      Z(oindex(i,j,k)) = X(iindex(i,j*BH,k*BW))
    }
    Z
  }

  def backward(d:Array[T]) = {
    val D = Array.ofDim[T](IC * IH * IW)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
        D(iindex(i,j*BH,k*BW)) += d(oindex(i,j,k))
    }
    D
  }

  def update() {}
  def reset() {}
}

class Pooling(val BW:Int, val IC:Int, val IH:Int, val IW:Int) extends Layer {
  val OH = IH / BW
  val OW = IW / BW
  val OC = IC
  var masks = List[Array[T]]()
  def push(x:Array[T]) = { masks ::= x; x }
  def pop() = { val mask = masks.head; masks = masks.tail; mask }

  def iindex(i:Int, j:Int, k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH * OW + j * OW + k

  def forward(X:Array[T]) = {
    val mask = push(Array.ofDim[T](IC * IH * IW))
    val Z = Array.ofDim[T](OC * OH * OW)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      var v = Double.NegativeInfinity
      var row_max = -1
      var col_max = -1
      for(m <- 0 until BW; n <- 0 until BW if v < X(iindex(i,j*BW+m,k*BW+n))) {
        row_max = j*BW+m
        col_max = k*BW+n
        v = X(iindex(i,j*BW+m,k*BW+n))
      }
      mask(iindex(i,row_max,col_max)) = 1
      Z(oindex(i,j,k)) = v
    }
    Z
  }

  def backward(d:Array[T]) = {
    val mask = pop()
    val D = Array.ofDim[T](mask.size)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      for(m <- 0 until BW; n <- 0 until BW if mask(iindex(i,j*BW+m,k*BW+n)) == 1) {
        D(iindex(i,j*BW+m,k*BW+n)) = d(oindex(i,j,k))
      }
    }
    D
  }

  def update() {}
  def reset() {
    masks = List[Array[T]]()
  }
}

class Affine(val xn:Int, val yn:Int, val eps:Double = 0.001, val rho1:Double = 0.9, val rho2:Double = 0.999) extends Layer {
  val rand = new scala.util.Random(0)

  var W = Array.ofDim[T](xn * yn).map(_ => rand.nextGaussian * 0.01)
  var b = Array.ofDim[T](yn)
  var dW = Array.ofDim[T](xn * yn)
  var db = Array.ofDim[T](yn)
  var n = 0


  def windex(i:Int, j:Int) = i * xn + j

  var xs = List[Array[T]]()
  def push(x:Array[T]) = { xs ::= x; x }
  def pop() = { val x = xs.head; xs = xs.tail; x }

  def forward(x:Array[T]) = {
    push(x)
    val y = Array.ofDim[T](yn)
    for(i <- 0 until yn) {
      for(j <- 0 until xn) {
        y(i) += W(windex(i,j)) * x(j)
      }
      y(i) += b(i)
    }
    y
  }

  def backward(d:Array[T]) = {
    val x = pop()
    n += 1

    for(i <- 0 until yn; j <- 0 until xn) {
      dW(windex(i,j)) += d(i) * x(j)
    }

    for(i <- 0 until yn) {
      db(i) += d(i)
    }

    val dx = Array.ofDim[T](xn)
    for(j <- 0 until yn; i <- 0 until xn) {
      dx(i) += W(windex(j,i)) * d(j)
    }

    dx
  }

  def update() {
    for(i <- 0 until dW.size) {
      dW(i) /= n
    }
    for(i <- 0 until db.size) {
      db(i) /= n
    }
    update_adam()
    reset()
  }

  var adam_W = new Adam(W.size, eps, rho1, rho2)
  var adam_b = new Adam(b.size, eps, rho1, rho2)
  def update_adam() {
    adam_W.update(W,dW)
    adam_b.update(b,db)
  }

  var lr = 0.001
  def update_sgd() {
    for(i <- 0 until W.size) {
      W(i) -= lr * dW(i)
    }

    for(i <- 0 until b.size) {
      b(i) -= lr * db(i)
    }
  }

  def reset() {
    for(i <- 0 until dW.size) {
      dW(i) = 0d
    }
    for(i <- 0 until db.size) {
      db(i) = 0d
    }
    xs = List[Array[T]]()
    n = 0
  }

  override def save(fn:String) {
    val pw = new java.io.PrintWriter(fn)
    for(i <- 0 until W.size) {
      pw.write(W(i).toString)
      if(i != W.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")
    for(i <- 0 until b.size) {
      pw.write(b(i).toString)
      if(i != b.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")
    pw.close()
  }

  override def load(fn:String) {
    val f = scala.io.Source.fromFile(fn).getLines.toArray
    W = f(0).split(",").map(_.toDouble).toArray
    b = f(1).split(",").map(_.toDouble).toArray
  }
}

class Sigmoid() extends Layer {
  var ys = List[Array[T]]()
  def push(y:Array[T]) = { ys ::= y; y }
  def pop() = { val y = ys.head; ys = ys.tail; y }

  def sigmoid(x:Double) = 1 / (1 + math.exp(-x))

  def forward(x:Array[T]) = {
    push(x.map(sigmoid))
  }

  def backward(d:Array[T]) = {
    val y = pop()
    (0 until d.size).map(i => d(i) * y(i) * (1d - y(i))).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[T]]()
  }
}

class Tanh() extends Layer {
  var ys = List[Array[T]]()
  def push(y:Array[T]) = { ys ::= y; y }
  def pop() = { val y = ys.head; ys = ys.tail; y }

  def forward(x:Array[T]) = {
    push(x.map(math.tanh))
  }

  def backward(d:Array[T]) = {
    val y = pop()
    (0 until d.size).map(i => d(i) * (1d - y(i) * y(i))).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[T]]()
  }
}

class Dropout(var dr:Double) extends Layer {
  var masks = List[Array[T]]()
  def push(mask:Array[T]) = { masks ::= mask; mask }
  def pop() = { val mask = masks.head; masks = masks.tail; mask }
  val rand=new util.Random(0)
  def forward(x:Array[T]) = {
    if(is_test) {
      x.map(_ * (1 - dr))
    } else {
      val mask = push(Array.ofDim[T](x.size))
      for(i <- 0 until x.size) {
        if(rand.nextDouble > dr) {
          mask(i) = 1d
        }
      }
      x.zip(mask).map{ case (a,b) => a * b }.toArray
    }
  }

  def backward(d:Array[T]) = {
    val mask = pop()
    (0 until d.size).map(i => if(mask(i) > 0) d(i) else 0d).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    masks = List[Array[T]]()
  }
}

class ReLU() extends Layer {
  var ys = List[Array[T]]()
  def push(y:Array[T]) = { ys ::= y; y }
  def pop() = { val y = ys.head; ys = ys.tail; y }

  def forward(x:Array[T]) = {
    push(x.map(a => math.max(a,0)))
  }

  def backward(d:Array[T]) = {
    val y = pop()
    (0 until d.size).map(i => if(y(i) > 0) d(i) else 0d).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[T]]()
  }
}

class LeakyReLU(val alpha:Double) extends Layer {
  var ys = List[Array[T]]()
  def push(y:Array[T]) = { ys ::= y; y }
  def pop() = { val y = ys.head; ys = ys.tail; y }

  def forward(x:Array[T]) = {
    push(x.map(a => if(a > 0) a else alpha * a))
  }

  def backward(d:Array[T]) = {
    val y = pop()
    (0 until d.size).map(i => if(y(i) > 0) d(i) else alpha * d(i)).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[T]]()
  }
}

class Ident() extends Layer {
  def forward(x:Array[T]) = x
  def backward(d:Array[T]) = d
  def update() {}
  def reset() {}
}

class ZeroPadding(val IC:Int, val IH:Int, val IW:Int, P:Int) extends Layer {
  val OH = IH + 2 * P
  val OW = IW + 2 * P
  val OC = IC
  def iindex(i:Int, j:Int, k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH * OW + j * OW + k
  def forward(x:Array[T]) = {
    val y = new Array[T](OC * OH * OW)
    for(c <- 0 until IC; i <- 0 until IH; j <- 0 until IW) {
      y(oindex(c,i+P,j+P)) = x(iindex(c,i,j))
    }
    y
  }
  def backward(d:Array[T]) = {
    val d1 = new Array[T](IC * IH * IW)
    for(c <- 0 until IC; i <- 0 until IH; j <- 0 until IW) {
      d1(iindex(c,i,j)) = d(oindex(c,i+P,j+P))
    }
    d1
  }
  def update() {}
  def reset() {}
}

class Adam(val n:Int, val eps:Double = 0.0002, val rho1:Double  = 0.5, val rho2:Double  = 0.999) {
  val delta = 1e-8
  var rho1t = 1d
  var rho2t = 1d
  var s = Array.ofDim[Double](n)
  var r = Array.ofDim[Double](n)

  def update(K:Array[Double], dK:Array[Double]) = {
    var nK = Array.ofDim[Double](K.size)
    rho1t *= rho1
    rho2t *= rho2
    val rho1tr = 1 / (1 - rho1t)
    val rho2tr = 1 / (1 - rho2t)
    for(i <- 0 until K.size) {
      s(i) = rho1 * s(i) + (1 - rho1) * dK(i)
      r(i) = rho2 * r(i) + (1 - rho2) * dK(i) * dK(i)
      val d = (s(i) * rho1tr) / (math.sqrt(r(i) * rho2tr) + delta)
      K(i) = K(i) - eps * d
    }
  }
}




class Softplus() extends Layer {
  var ys = List[Array[T]]()
  def push(y:Array[T]) = { ys ::= y; y }
  def pop() = { val y = ys.head; ys = ys.tail; y }

  def softplus(x:Double) = x + math.log(1d + math.exp(-x))
  def sigmoid(x:Double) = 1 / (1d + math.exp(-x))

  def forward(x:Array[T]) = {
    push(x)
    x.map(softplus)
  }

  def backward(d:Array[T]) = {
    val x = pop()
    (0 until d.size).map(i => d(i) * sigmoid(x(i))).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[T]]()
  }
}

class BNa(val xn:Int, val eps:Double = 0.001, val rho1:Double = 0.9, val rho2:Double = 0.999) extends Layer {
  var gamma = new Array[T](xn).map(_ => 1 : T)
  var beta = new Array[T](xn)
  var dgamma = new Array[T](gamma.size)
  var dbeta = new Array[T](beta.size)
  val adam_gamma = new Adam(gamma.size,eps,rho1,rho2)
  val adam_beta = new Adam(beta.size,eps,rho1,rho2)
  var xmu = Array.ofDim[T](1,xn) // rhs is just a placeholder value
  var sigma = new Array[T](xn)
  val delta = 1e-8
  var mmu = new Array[T](xn)
  var msigma = new Array[T](xn)
  val decay = 0.999

  def forward(x:Array[T]) : Array[T] = {
    val y = new Array[T](xn)
    for(i <- 0 until xn) {
      val xh = (x(i) - mmu(i)) / (msigma(i) + delta)
      y(i) = xh * gamma(i) + beta(i)
    }
    y
  }

  def backward(d:Array[T]) : Array[T] = {
    d
  }

  override def forward(xs:Array[Array[T]]) : Array[Array[T]] = {
    val m = xs.size
    xmu = Array.ofDim[T](m,xn)
    for(j <- 0 until xn) {
      var mu = 0d
      for(i <- 0 until m) {
        mu += xs(i)(j)
      }
      mu /= m
      mmu(j) = decay * mmu(j) + (1-decay) * mu
      for(i <- 0 until m) {
        xmu(i)(j) = xs(i)(j) - mu
        sigma(j) += xmu(i)(j) * xmu(i)(j)
      }
      sigma(j) = math.sqrt(sigma(j) / m + delta)
      msigma(j) = decay * msigma(j) + (1-decay) * sigma(j)
    }

    var ys = Array.ofDim[T](m,xn)
    for(j <- 0 until xn) {
      for(i <- 0 until m) {
        ys(i)(j) = gamma(j) * xmu(i)(j) / sigma(j) + beta(j)
      }
    }
    ys
  }

  override def backward(ds:Array[Array[T]]) : Array[Array[T]] = {
    val m = ds.size
    var dx = Array.ofDim[T](m,xn)
    for(j <- 0 until xn) {
      for(i <- 0 until m) {
        dbeta(j) += ds(i)(j)
        dgamma(j) += ds(i)(j) * xmu(i)(j) / sigma(j)
      }

      var d1 = new Array[T](m)
      var d2 = 0d
      for(i <- 0 until m) {
        d1(i) = gamma(j) * ds(i)(j)
        d2 += xmu(i)(j) * d1(i)
      }

      val d3 = -d2 / (sigma(j) * sigma(j))
      val d4 = d3 / (2 *  sigma(j))

      var d8 = 0d
      var d10 = new Array[T](m)
      for(i <- 0 until m) {
        val d5 = d4 / m
        val d6 = 2 * xmu(i)(j) * d5
        val d7 = d1(i) / sigma(j)
        d10(i) = d6 + d7
        d8 -= d10(i)
      }
      val d9 = d8 / m

      for(i <- 0 until m) {
        dx(i)(j) = d9 + d10(i)
      }
    }
    dx
  }

  def update() {
    adam_beta.update(beta,dbeta)
    adam_gamma.update(gamma,dgamma)
    reset()
  }

  def reset() {
    dgamma = new Array[T](gamma.size)
    dbeta = new Array[T](beta.size)
  }
}



object Image {
  def rgb(im : java.awt.image.BufferedImage, i:Int, j:Int) = {
    val c = im.getRGB(i,j)
    Array(c >> 16 & 0xff, c >> 8 & 0xff, c & 0xff)
  }

  def pixel(r:Int, g:Int, b:Int) = {
    val a = 0xff
    ((a & 0xff) << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff)
  }

  def read(fn:String) = {
    val im = javax.imageio.ImageIO.read(new java.io.File(fn))
    (for(i <- 0 until im.getHeight; j <- 0 until im.getWidth)
    yield rgb(im, j, i)).toArray.grouped(im.getWidth).toArray
  }

  def write(fn:String, b:Array[Array[Array[Int]]]) = {
    val w = b(0).size
    val h = b.size
    val im = new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB);
    for(i <- 0 until im.getHeight; j <- 0 until im.getWidth) {
      im.setRGB(j,i,pixel(b(i)(j)(0), b(i)(j)(1), b(i)(j)(2)));
    }
    javax.imageio.ImageIO.write(im, "png", new java.io.File(fn))
  }

  def make_image2(ys:Array[Array[Double]], NW:Int, NH:Int, H:Int, W:Int) = {
    val im = Array.ofDim[Int](NH * H, NW * W, 3)
    val ymax = ys.flatten.max
    val ymin = ys.flatten.min
    def f(a:Double) = ((a - ymin) / (ymax - ymin) * 255).toInt
    for(i <- 0 until NH; j <- 0 until NW) {
      for(p <- 0 until H; q <- 0 until W; k <- 0 until 3) {//k * H * W +
        im(i * H + p)(j * W + q)(k) = f(ys(i * NW + j)( p * W + q))
      }
    }
    im
  }


  def make_image3(ys:Array[Array[Double]], NW:Int, NH:Int, H:Int, W:Int) = {
    val im = Array.ofDim[Int](NH * H, NW * W, 3)
    /*  val ymax = ys.flatten.max
     val ymin = ys.flatten.min*/
    def f(a:Double) = (((a+1)/2)*255).toInt

    for(i <- 0 until NH; j <- 0 until NW) {
      for(p <- 0 until H; q <- 0 until W; k <- 0 until 3) {
        im(i * H + p)(j * W + q)(k) = f(ys(i * NW + j)( p * W + q))
      }
    }
    im
  }

  //三色用

  def make_image(ys:Array[Array[Double]], NW:Int, NH:Int, H:Int, W:Int) = {
    val im = Array.ofDim[Int](NH * H, NW * W, 3)
    val ymax = ys.flatten.max
    val ymin = ys.flatten.min
    def f(a:Double) = ((a - ymin) / (ymax - ymin) * 255).toInt
    for(i <- 0 until NH; j <- 0 until NW) {
      for(p <- 0 until H; q <- 0 until W; k <- 0 until 3) {
        im(i * H + p)(j * W + q)(k) = f(ys(i * NW + j)(k * H * W + p * W + q))
      }
    }
    im
  }

  def to3DArrayOfColor(image:Array[Double],h:Int,w:Int) = {
    val input = image.map(_*256)
    var output = List[Array[Array[Double]]]()
    for(i <- 0 until h) {
      var row = List[Array[Double]]()
      for(j <- 0 until w) {
        val red = input(i*w+j)
        val green = input(i*w+j+h*w)
        val blue = input(i*w+j+h*w*2)
        row ::= Array(red,green,blue)
      }
      output ::= row.reverse.toArray
    }
    output.reverse.toArray.map(_.map(_.map(_.toInt)))
  }



}
