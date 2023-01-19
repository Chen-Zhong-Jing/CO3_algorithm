# Communication-Efficient Federated DNN Training in Three Steps: Convert, Compress, Correct

There are several compression scheme in this code.
## CO3 Algorithm
#### Convert
There are two types you can use for quantization, fp4 and fp8, which follow the IEEE 754 format.
- fp4: [sign, exponent, mantissa] = [1,2,1]
- fp8: [sign, exponent, mantissa] = [1,5,2]

#### Correct
There is a error correction coefficient in this algorithm, which can control the magnitude of error term.

## Count Sketch
The Sketch code is from [csvec](https://github.com/nikitaivkin/csh)
Reference: [FetchSGD: Communication-Efficient Federated Learning with Sketching](https://proceedings.mlr.press/v119/fu20c.html)

## TinyScript
Reference: [Don’t Waste Your Bits! Squeeze Activations and Gradients for Deep Neural Networks via TinyScript](https://proceedings.mlr.press/v119/fu20c.html)
