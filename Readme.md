# Readme

__1.AttributeError: module 'tensorflow.python.ops.image_ops' has no attribute 'random_crop'。__

这个错误来自于cifar10_input.py文件中的distorted_image = tf.image.random_crop(reshaped_image, [height, width])，将此句修改为：

```
distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
```

__2.AttributeError: module 'tensorflow.python.ops.image_ops' has no attribute 'per_image_whitening'。__

这个错误来自于cifar10_input.py文件中的 float_image = tf.image.per_image_whitening(distorted_image)，将此句修改为：

```
 float_image = tf.image.per_image_standardization(distorted_image)
```

__3. AttributeError: module 'tensorflow' has no attribute 'image_summary'。__

这个错误来自于cifar10_input.py文件中的tf.image_summary('images', images)，将此句修改为：

```
tf.summary.image('images', images)
 tf.summary.scalar('learning_rate', lr)#cifar10.py
```

注：整个项目类似的地方做修改。

__4.AttributeError: module 'tensorflow' has no attribute 'histogram_summary'。__

这个错误来自于cifar10.py文件中的 tf.histogram_summary(tensor_name + '/activations', x)，将此句修改为：

```
tf.summary.histogram(tensor_name + '/activations', x)
tf.summary.histogram(var.op.name, var)
```

注：整个项目类似的地方做修改。

__5.AttributeError: module 'tensorflow' has no attribute 'scalar_summary'。__

这个错误来自于cifar10.py文件中的 tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))，将此句修改为：

```
tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
```

__6.AttributeError: module 'tensorflow' has no attribute 'mul'。__

这个错误之前有说过，改为multiply即可。

__7. ValueError: Tried to convert 'tensor' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 128) - got shape [128], but wanted []。__

这个错误定位在cifar10.py的这句代码上：

```
 indices = tf.reshape(range(FLAGS.batch_size), [FLAGS.batch_size, 1])
```

将此句改为：

```
 indices = tf.reshape(list(range(FLAGS.batch_size)), [FLAGS.batch_size, 1])
```

__8. ValueError: Shapes (2, 128, 1) and () are incompatible。__

这个错误是cifar10.py中的 concated = tf.concat(1, [indices, sparse_labels])触发的，此句修改为：

```
 concated = tf.concat([indices, sparse_labels], 1)
```

__9. ValueError: Only call `softmax_cross_entropy_with_logits` with named arguments (labels=..., logits=..., ...)。__

这个错误来自于softmax_cross_entropy_with_logits这个函数，新的tf版本更新了这个函数，函数原型：

```
tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)
```

将其修改为：

```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=dense_labels, name='cross_entropy_per_example')
```

__10. TypeError: Using a `tf.Tensor` as a Python `bool` is not allowed. Use `if t is not None:` instead of `if t:` to test if a tensor is defined, and use TensorFlow ops such as tf.cond to execute subgraphs conditioned on the value of a tensor.__

提示了，应该将 if grad: 修改为 if grad is not None。

__11. AttributeError: module 'tensorflow' has no attribute 'merge_all_summaries'。__

此处错误来自于cifar10_train.py中的 summary_op = tf.merge_all_summaries()，将其修改为：

```
 summary_op = tf.summary.merge_all()
```

__12. AttributeError: module 'tensorflow.python.training.training' has no attribute 'SummaryWriter'。__

这个错误来自于cifar10_train.py的

```
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)
```

将其修改为：

```
summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)
```

__13. WARNING:tensorflow:Passing a `GraphDef` to the SummaryWriter is deprecated. Pass a `Graph` object instead, such as `sess.graph`.__

这个警告来自于cifar10_train.py的

```
summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)
```

将其修改为：

```
summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                            sess.graph)
```

到这里为止，才能成功的运行这个例子，上述错误都是由于tf高低版本不兼容导致的，代码本身没有问题，只是在高版本的tf做了修改，比如本机的版本是1.7.0，某些低版本应该没有问题。



[代码解析]: https://www.cnblogs.com/cvtoEyes/p/8981994.html

