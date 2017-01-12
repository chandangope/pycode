node {
  name: "input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\377\377\377\377H\000\000\000$\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "Reshape"
  op: "Reshape"
  input: "input"
  input: "Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "truncated_normal/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\001\000\000\000\030\000\000\000"
      }
    }
  }
}
node {
  name: "truncated_normal/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "truncated_normal/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "truncated_normal/TruncatedNormal"
  op: "TruncatedNormal"
  input: "truncated_normal/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "truncated_normal/mul"
  op: "Mul"
  input: "truncated_normal/TruncatedNormal"
  input: "truncated_normal/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "truncated_normal"
  op: "Add"
  input: "truncated_normal/mul"
  input: "truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "W_conv1"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 1
        }
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_conv1/Assign"
  op: "Assign"
  input: "W_conv1"
  input: "truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_conv1/read"
  op: "Identity"
  input: "W_conv1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
}
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 24
          }
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "b_conv1"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_conv1/Assign"
  op: "Assign"
  input: "b_conv1"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_conv1/read"
  op: "Identity"
  input: "b_conv1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
}
node {
  name: "Conv2D"
  op: "Conv2D"
  input: "Reshape"
  input: "W_conv1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "add"
  op: "Add"
  input: "Conv2D"
  input: "b_conv1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Relu"
  op: "Relu"
  input: "add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "MaxPool"
  op: "MaxPool"
  input: "Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "truncated_normal_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\030\000\000\0000\000\000\000"
      }
    }
  }
}
node {
  name: "truncated_normal_1/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "truncated_normal_1/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "truncated_normal_1/TruncatedNormal"
  op: "TruncatedNormal"
  input: "truncated_normal_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "truncated_normal_1/mul"
  op: "Mul"
  input: "truncated_normal_1/TruncatedNormal"
  input: "truncated_normal_1/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "truncated_normal_1"
  op: "Add"
  input: "truncated_normal_1/mul"
  input: "truncated_normal_1/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "W_conv2"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 24
        }
        dim {
          size: 48
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_conv2/Assign"
  op: "Assign"
  input: "W_conv2"
  input: "truncated_normal_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_conv2/read"
  op: "Identity"
  input: "W_conv2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
}
node {
  name: "Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 48
          }
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "b_conv2"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 48
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_conv2/Assign"
  op: "Assign"
  input: "b_conv2"
  input: "Const_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_conv2/read"
  op: "Identity"
  input: "b_conv2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
}
node {
  name: "Conv2D_1"
  op: "Conv2D"
  input: "MaxPool"
  input: "W_conv2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "add_1"
  op: "Add"
  input: "Conv2D_1"
  input: "b_conv2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Relu_1"
  op: "Relu"
  input: "add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "MaxPool_1"
  op: "MaxPool"
  input: "Relu_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "truncated_normal_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\0000\000\000\000`\000\000\000"
      }
    }
  }
}
node {
  name: "truncated_normal_2/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "truncated_normal_2/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "truncated_normal_2/TruncatedNormal"
  op: "TruncatedNormal"
  input: "truncated_normal_2/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "truncated_normal_2/mul"
  op: "Mul"
  input: "truncated_normal_2/TruncatedNormal"
  input: "truncated_normal_2/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "truncated_normal_2"
  op: "Add"
  input: "truncated_normal_2/mul"
  input: "truncated_normal_2/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "W_conv3"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 48
        }
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_conv3/Assign"
  op: "Assign"
  input: "W_conv3"
  input: "truncated_normal_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_conv3/read"
  op: "Identity"
  input: "W_conv3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
}
node {
  name: "Const_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "b_conv"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_conv/Assign"
  op: "Assign"
  input: "b_conv"
  input: "Const_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_conv/read"
  op: "Identity"
  input: "b_conv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
}
node {
  name: "Conv2D_2"
  op: "Conv2D"
  input: "MaxPool_1"
  input: "W_conv3/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "add_2"
  op: "Add"
  input: "Conv2D_2"
  input: "b_conv/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Relu_2"
  op: "Relu"
  input: "add_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\300<\000\000"
      }
    }
  }
}
node {
  name: "Reshape_1"
  op: "Reshape"
  input: "Relu_2"
  input: "Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "truncated_normal_3/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\300<\000\000\300\000\000\000"
      }
    }
  }
}
node {
  name: "truncated_normal_3/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "truncated_normal_3/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "truncated_normal_3/TruncatedNormal"
  op: "TruncatedNormal"
  input: "truncated_normal_3/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "truncated_normal_3/mul"
  op: "Mul"
  input: "truncated_normal_3/TruncatedNormal"
  input: "truncated_normal_3/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "truncated_normal_3"
  op: "Add"
  input: "truncated_normal_3/mul"
  input: "truncated_normal_3/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "W_fc1"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 15552
        }
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_fc1/Assign"
  op: "Assign"
  input: "W_fc1"
  input: "truncated_normal_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_fc1/read"
  op: "Identity"
  input: "W_fc1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
}
node {
  name: "Const_3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "b_fc1"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_fc1/Assign"
  op: "Assign"
  input: "b_fc1"
  input: "Const_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_fc1/read"
  op: "Identity"
  input: "b_fc1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
}
node {
  name: "MatMul"
  op: "MatMul"
  input: "Reshape_1"
  input: "W_fc1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "add_3"
  op: "Add"
  input: "MatMul"
  input: "b_fc1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Relu_3"
  op: "Relu"
  input: "add_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "keep_prob"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "dropout/Shape"
  op: "Shape"
  input: "Relu_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "dropout/random_uniform/min"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "dropout/random_uniform/max"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dropout/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "dropout/Shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "dropout/random_uniform/sub"
  op: "Sub"
  input: "dropout/random_uniform/max"
  input: "dropout/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/random_uniform/mul"
  op: "Mul"
  input: "dropout/random_uniform/RandomUniform"
  input: "dropout/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/random_uniform"
  op: "Add"
  input: "dropout/random_uniform/mul"
  input: "dropout/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/add"
  op: "Add"
  input: "keep_prob"
  input: "dropout/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/Floor"
  op: "Floor"
  input: "dropout/add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/Div"
  op: "Div"
  input: "Relu_3"
  input: "keep_prob"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dropout/mul"
  op: "Mul"
  input: "dropout/Div"
  input: "dropout/Floor"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "truncated_normal_4/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\300\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "truncated_normal_4/mean"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "truncated_normal_4/stddev"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "truncated_normal_4/TruncatedNormal"
  op: "TruncatedNormal"
  input: "truncated_normal_4/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "truncated_normal_4/mul"
  op: "Mul"
  input: "truncated_normal_4/TruncatedNormal"
  input: "truncated_normal_4/stddev"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "truncated_normal_4"
  op: "Add"
  input: "truncated_normal_4/mul"
  input: "truncated_normal_4/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "W_fc2"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_fc2/Assign"
  op: "Assign"
  input: "W_fc2"
  input: "truncated_normal_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_fc2/read"
  op: "Identity"
  input: "W_fc2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
}
node {
  name: "Const_4"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        float_val: 0.10000000149
      }
    }
  }
}
node {
  name: "b_fc2"
  op: "Variable"
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_fc2/Assign"
  op: "Assign"
  input: "b_fc2"
  input: "Const_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_fc2/read"
  op: "Identity"
  input: "b_fc2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
}
node {
  name: "MatMul_1"
  op: "MatMul"
  input: "dropout/mul"
  input: "W_fc2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "add_4"
  op: "Add"
  input: "MatMul_1"
  input: "b_fc2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "softmax"
  op: "Softmax"
  input: "add_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Log"
  op: "Log"
  input: "softmax"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "mul"
  op: "Mul"
  input: "Placeholder"
  input: "Log"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Const_5"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "Sum"
  op: "Sum"
  input: "mul"
  input: "Const_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "Neg"
  op: "Neg"
  input: "Sum"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Const_6"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "Mean"
  op: "Mean"
  input: "Neg"
  input: "Const_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "gradients/Fill"
  op: "Fill"
  input: "gradients/Shape"
  input: "gradients/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Mean_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Reshape"
  op: "Reshape"
  input: "gradients/Fill"
  input: "gradients/Mean_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/Tile/multiples"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Tile"
  op: "Tile"
  input: "gradients/Mean_grad/Reshape"
  input: "gradients/Mean_grad/Tile/multiples"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
          }
        }
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Prod"
  op: "Prod"
  input: "gradients/Mean_grad/Shape"
  input: "gradients/Mean_grad/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/Mean_grad/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Prod_1"
  op: "Prod"
  input: "gradients/Mean_grad/Shape_1"
  input: "gradients/Mean_grad/Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/Mean_grad/Maximum/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/Mean_grad/Maximum"
  op: "Maximum"
  input: "gradients/Mean_grad/Prod_1"
  input: "gradients/Mean_grad/Maximum/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/floordiv"
  op: "Div"
  input: "gradients/Mean_grad/Prod"
  input: "gradients/Mean_grad/Maximum"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/Cast"
  op: "Cast"
  input: "gradients/Mean_grad/floordiv"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Mean_grad/truediv"
  op: "Div"
  input: "gradients/Mean_grad/Tile"
  input: "gradients/Mean_grad/Cast"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Neg_grad/Neg"
  op: "Neg"
  input: "gradients/Mean_grad/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Sum_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "gradients/Sum_grad/Reshape"
  op: "Reshape"
  input: "gradients/Neg_grad/Neg"
  input: "gradients/Sum_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/Shape"
  op: "Shape"
  input: "mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Sum_grad/Tile"
  op: "Tile"
  input: "gradients/Sum_grad/Reshape"
  input: "gradients/Sum_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/mul_grad/Shape"
  op: "Shape"
  input: "Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/mul_grad/Shape_1"
  op: "Shape"
  input: "Log"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/mul_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/mul_grad/Shape"
  input: "gradients/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/mul_grad/mul"
  op: "Mul"
  input: "gradients/Sum_grad/Tile"
  input: "Log"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/mul_grad/Sum"
  op: "Sum"
  input: "gradients/mul_grad/mul"
  input: "gradients/mul_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/mul_grad/Reshape"
  op: "Reshape"
  input: "gradients/mul_grad/Sum"
  input: "gradients/mul_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/mul_grad/mul_1"
  op: "Mul"
  input: "Placeholder"
  input: "gradients/Sum_grad/Tile"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/mul_grad/Sum_1"
  op: "Sum"
  input: "gradients/mul_grad/mul_1"
  input: "gradients/mul_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/mul_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/mul_grad/Sum_1"
  input: "gradients/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/mul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/mul_grad/Reshape"
  input: "^gradients/mul_grad/Reshape_1"
}
node {
  name: "gradients/mul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/mul_grad/Reshape"
  input: "^gradients/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/mul_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/mul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/mul_grad/Reshape_1"
  input: "^gradients/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/mul_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/Log_grad/Inv"
  op: "Inv"
  input: "softmax"
  input: "^gradients/mul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/Log_grad/mul"
  op: "Mul"
  input: "gradients/mul_grad/tuple/control_dependency_1"
  input: "gradients/Log_grad/Inv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_grad/mul"
  op: "Mul"
  input: "gradients/Log_grad/mul"
  input: "softmax"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_grad/Sum/reduction_indices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "gradients/softmax_grad/Sum"
  op: "Sum"
  input: "gradients/softmax_grad/mul"
  input: "gradients/softmax_grad/Sum/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/softmax_grad/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "gradients/softmax_grad/Reshape"
  op: "Reshape"
  input: "gradients/softmax_grad/Sum"
  input: "gradients/softmax_grad/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/softmax_grad/sub"
  op: "Sub"
  input: "gradients/Log_grad/mul"
  input: "gradients/softmax_grad/Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/softmax_grad/mul_1"
  op: "Mul"
  input: "gradients/softmax_grad/sub"
  input: "softmax"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/add_4_grad/Shape"
  op: "Shape"
  input: "MatMul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_4_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "gradients/add_4_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/add_4_grad/Shape"
  input: "gradients/add_4_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_4_grad/Sum"
  op: "Sum"
  input: "gradients/softmax_grad/mul_1"
  input: "gradients/add_4_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_4_grad/Reshape"
  op: "Reshape"
  input: "gradients/add_4_grad/Sum"
  input: "gradients/add_4_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_4_grad/Sum_1"
  op: "Sum"
  input: "gradients/softmax_grad/mul_1"
  input: "gradients/add_4_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_4_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/add_4_grad/Sum_1"
  input: "gradients/add_4_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_4_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/add_4_grad/Reshape"
  input: "^gradients/add_4_grad/Reshape_1"
}
node {
  name: "gradients/add_4_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/add_4_grad/Reshape"
  input: "^gradients/add_4_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_4_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/add_4_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/add_4_grad/Reshape_1"
  input: "^gradients/add_4_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_4_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/MatMul_1_grad/MatMul"
  op: "MatMul"
  input: "gradients/add_4_grad/tuple/control_dependency"
  input: "W_fc2/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/MatMul_1_grad/MatMul_1"
  op: "MatMul"
  input: "dropout/mul"
  input: "gradients/add_4_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/MatMul_1_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/MatMul_1_grad/MatMul"
  input: "^gradients/MatMul_1_grad/MatMul_1"
}
node {
  name: "gradients/MatMul_1_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/MatMul_1_grad/MatMul"
  input: "^gradients/MatMul_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/MatMul_1_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/MatMul_1_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/MatMul_1_grad/MatMul_1"
  input: "^gradients/MatMul_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/MatMul_1_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/Shape"
  op: "Shape"
  input: "dropout/Div"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/Shape_1"
  op: "Shape"
  input: "dropout/Floor"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/dropout/mul_grad/Shape"
  input: "gradients/dropout/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/mul"
  op: "Mul"
  input: "gradients/MatMul_1_grad/tuple/control_dependency"
  input: "dropout/Floor"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/Sum"
  op: "Sum"
  input: "gradients/dropout/mul_grad/mul"
  input: "gradients/dropout/mul_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/Reshape"
  op: "Reshape"
  input: "gradients/dropout/mul_grad/Sum"
  input: "gradients/dropout/mul_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/mul_1"
  op: "Mul"
  input: "dropout/Div"
  input: "gradients/MatMul_1_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/Sum_1"
  op: "Sum"
  input: "gradients/dropout/mul_grad/mul_1"
  input: "gradients/dropout/mul_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/dropout/mul_grad/Sum_1"
  input: "gradients/dropout/mul_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/dropout/mul_grad/Reshape"
  input: "^gradients/dropout/mul_grad/Reshape_1"
}
node {
  name: "gradients/dropout/mul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/dropout/mul_grad/Reshape"
  input: "^gradients/dropout/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dropout/mul_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/dropout/mul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/dropout/mul_grad/Reshape_1"
  input: "^gradients/dropout/mul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dropout/mul_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/Shape"
  op: "Shape"
  input: "Relu_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/Shape_1"
  op: "Shape"
  input: "keep_prob"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/dropout/Div_grad/Shape"
  input: "gradients/dropout/Div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/truediv"
  op: "Div"
  input: "gradients/dropout/mul_grad/tuple/control_dependency"
  input: "keep_prob"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/Sum"
  op: "Sum"
  input: "gradients/dropout/Div_grad/truediv"
  input: "gradients/dropout/Div_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/Reshape"
  op: "Reshape"
  input: "gradients/dropout/Div_grad/Sum"
  input: "gradients/dropout/Div_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/Neg"
  op: "Neg"
  input: "Relu_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/Square"
  op: "Square"
  input: "keep_prob"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/truediv_1"
  op: "Div"
  input: "gradients/dropout/Div_grad/Neg"
  input: "gradients/dropout/Div_grad/Square"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/mul"
  op: "Mul"
  input: "gradients/dropout/mul_grad/tuple/control_dependency"
  input: "gradients/dropout/Div_grad/truediv_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/Sum_1"
  op: "Sum"
  input: "gradients/dropout/Div_grad/mul"
  input: "gradients/dropout/Div_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/dropout/Div_grad/Sum_1"
  input: "gradients/dropout/Div_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/dropout/Div_grad/Reshape"
  input: "^gradients/dropout/Div_grad/Reshape_1"
}
node {
  name: "gradients/dropout/Div_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/dropout/Div_grad/Reshape"
  input: "^gradients/dropout/Div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dropout/Div_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/dropout/Div_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/dropout/Div_grad/Reshape_1"
  input: "^gradients/dropout/Div_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/dropout/Div_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/Relu_3_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/dropout/Div_grad/tuple/control_dependency"
  input: "Relu_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/add_3_grad/Shape"
  op: "Shape"
  input: "MatMul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_3_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 192
      }
    }
  }
}
node {
  name: "gradients/add_3_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/add_3_grad/Shape"
  input: "gradients/add_3_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_3_grad/Sum"
  op: "Sum"
  input: "gradients/Relu_3_grad/ReluGrad"
  input: "gradients/add_3_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_3_grad/Reshape"
  op: "Reshape"
  input: "gradients/add_3_grad/Sum"
  input: "gradients/add_3_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_3_grad/Sum_1"
  op: "Sum"
  input: "gradients/Relu_3_grad/ReluGrad"
  input: "gradients/add_3_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_3_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/add_3_grad/Sum_1"
  input: "gradients/add_3_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_3_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/add_3_grad/Reshape"
  input: "^gradients/add_3_grad/Reshape_1"
}
node {
  name: "gradients/add_3_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/add_3_grad/Reshape"
  input: "^gradients/add_3_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_3_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/add_3_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/add_3_grad/Reshape_1"
  input: "^gradients/add_3_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_3_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/MatMul_grad/MatMul"
  op: "MatMul"
  input: "gradients/add_3_grad/tuple/control_dependency"
  input: "W_fc1/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/MatMul_grad/MatMul_1"
  op: "MatMul"
  input: "Reshape_1"
  input: "gradients/add_3_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: true
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/MatMul_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/MatMul_grad/MatMul"
  input: "^gradients/MatMul_grad/MatMul_1"
}
node {
  name: "gradients/MatMul_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/MatMul_grad/MatMul"
  input: "^gradients/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/MatMul_grad/MatMul"
      }
    }
  }
}
node {
  name: "gradients/MatMul_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/MatMul_grad/MatMul_1"
  input: "^gradients/MatMul_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/MatMul_grad/MatMul_1"
      }
    }
  }
}
node {
  name: "gradients/Reshape_1_grad/Shape"
  op: "Shape"
  input: "Relu_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Reshape_1_grad/Reshape"
  op: "Reshape"
  input: "gradients/MatMul_grad/tuple/control_dependency"
  input: "gradients/Reshape_1_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Relu_2_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/Reshape_1_grad/Reshape"
  input: "Relu_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/add_2_grad/Shape"
  op: "Shape"
  input: "Conv2D_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_2_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 96
      }
    }
  }
}
node {
  name: "gradients/add_2_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/add_2_grad/Shape"
  input: "gradients/add_2_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_2_grad/Sum"
  op: "Sum"
  input: "gradients/Relu_2_grad/ReluGrad"
  input: "gradients/add_2_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_2_grad/Reshape"
  op: "Reshape"
  input: "gradients/add_2_grad/Sum"
  input: "gradients/add_2_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_2_grad/Sum_1"
  op: "Sum"
  input: "gradients/Relu_2_grad/ReluGrad"
  input: "gradients/add_2_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_2_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/add_2_grad/Sum_1"
  input: "gradients/add_2_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_2_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/add_2_grad/Reshape"
  input: "^gradients/add_2_grad/Reshape_1"
}
node {
  name: "gradients/add_2_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/add_2_grad/Reshape"
  input: "^gradients/add_2_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_2_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/add_2_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/add_2_grad/Reshape_1"
  input: "^gradients/add_2_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_2_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/Conv2D_2_grad/Shape"
  op: "Shape"
  input: "MaxPool_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Conv2D_2_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "gradients/Conv2D_2_grad/Shape"
  input: "W_conv3/read"
  input: "gradients/add_2_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/Conv2D_2_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\0000\000\000\000`\000\000\000"
      }
    }
  }
}
node {
  name: "gradients/Conv2D_2_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "MaxPool_1"
  input: "gradients/Conv2D_2_grad/Shape_1"
  input: "gradients/add_2_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/Conv2D_2_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/Conv2D_2_grad/Conv2DBackpropInput"
  input: "^gradients/Conv2D_2_grad/Conv2DBackpropFilter"
}
node {
  name: "gradients/Conv2D_2_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/Conv2D_2_grad/Conv2DBackpropInput"
  input: "^gradients/Conv2D_2_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "gradients/Conv2D_2_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/Conv2D_2_grad/Conv2DBackpropFilter"
  input: "^gradients/Conv2D_2_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "gradients/MaxPool_1_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "Relu_1"
  input: "MaxPool_1"
  input: "gradients/Conv2D_2_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "gradients/Relu_1_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/MaxPool_1_grad/MaxPoolGrad"
  input: "Relu_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/add_1_grad/Shape"
  op: "Shape"
  input: "Conv2D_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_1_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 48
      }
    }
  }
}
node {
  name: "gradients/add_1_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/add_1_grad/Shape"
  input: "gradients/add_1_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_1_grad/Sum"
  op: "Sum"
  input: "gradients/Relu_1_grad/ReluGrad"
  input: "gradients/add_1_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_1_grad/Reshape"
  op: "Reshape"
  input: "gradients/add_1_grad/Sum"
  input: "gradients/add_1_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_1_grad/Sum_1"
  op: "Sum"
  input: "gradients/Relu_1_grad/ReluGrad"
  input: "gradients/add_1_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_1_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/add_1_grad/Sum_1"
  input: "gradients/add_1_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_1_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/add_1_grad/Reshape"
  input: "^gradients/add_1_grad/Reshape_1"
}
node {
  name: "gradients/add_1_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/add_1_grad/Reshape"
  input: "^gradients/add_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_1_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/add_1_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/add_1_grad/Reshape_1"
  input: "^gradients/add_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_1_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/Conv2D_1_grad/Shape"
  op: "Shape"
  input: "MaxPool"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Conv2D_1_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "gradients/Conv2D_1_grad/Shape"
  input: "W_conv2/read"
  input: "gradients/add_1_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/Conv2D_1_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\030\000\000\0000\000\000\000"
      }
    }
  }
}
node {
  name: "gradients/Conv2D_1_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "MaxPool"
  input: "gradients/Conv2D_1_grad/Shape_1"
  input: "gradients/add_1_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/Conv2D_1_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/Conv2D_1_grad/Conv2DBackpropInput"
  input: "^gradients/Conv2D_1_grad/Conv2DBackpropFilter"
}
node {
  name: "gradients/Conv2D_1_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/Conv2D_1_grad/Conv2DBackpropInput"
  input: "^gradients/Conv2D_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "gradients/Conv2D_1_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/Conv2D_1_grad/Conv2DBackpropFilter"
  input: "^gradients/Conv2D_1_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "gradients/MaxPool_grad/MaxPoolGrad"
  op: "MaxPoolGrad"
  input: "Relu"
  input: "MaxPool"
  input: "gradients/Conv2D_1_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
        i: 2
        i: 1
      }
    }
  }
}
node {
  name: "gradients/Relu_grad/ReluGrad"
  op: "ReluGrad"
  input: "gradients/MaxPool_grad/MaxPoolGrad"
  input: "Relu"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "gradients/add_grad/Shape"
  op: "Shape"
  input: "Conv2D"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 24
      }
    }
  }
}
node {
  name: "gradients/add_grad/BroadcastGradientArgs"
  op: "BroadcastGradientArgs"
  input: "gradients/add_grad/Shape"
  input: "gradients/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_grad/Sum"
  op: "Sum"
  input: "gradients/Relu_grad/ReluGrad"
  input: "gradients/add_grad/BroadcastGradientArgs"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_grad/Reshape"
  op: "Reshape"
  input: "gradients/add_grad/Sum"
  input: "gradients/add_grad/Shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_grad/Sum_1"
  op: "Sum"
  input: "gradients/Relu_grad/ReluGrad"
  input: "gradients/add_grad/BroadcastGradientArgs:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "gradients/add_grad/Reshape_1"
  op: "Reshape"
  input: "gradients/add_grad/Sum_1"
  input: "gradients/add_grad/Shape_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/add_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/add_grad/Reshape"
  input: "^gradients/add_grad/Reshape_1"
}
node {
  name: "gradients/add_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/add_grad/Reshape"
  input: "^gradients/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_grad/Reshape"
      }
    }
  }
}
node {
  name: "gradients/add_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/add_grad/Reshape_1"
  input: "^gradients/add_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/add_grad/Reshape_1"
      }
    }
  }
}
node {
  name: "gradients/Conv2D_grad/Shape"
  op: "Shape"
  input: "Reshape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "out_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "gradients/Conv2D_grad/Conv2DBackpropInput"
  op: "Conv2DBackpropInput"
  input: "gradients/Conv2D_grad/Shape"
  input: "W_conv1/read"
  input: "gradients/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/Conv2D_grad/Shape_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\001\000\000\000\030\000\000\000"
      }
    }
  }
}
node {
  name: "gradients/Conv2D_grad/Conv2DBackpropFilter"
  op: "Conv2DBackpropFilter"
  input: "Reshape"
  input: "gradients/Conv2D_grad/Shape_1"
  input: "gradients/add_grad/tuple/control_dependency"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "gradients/Conv2D_grad/tuple/group_deps"
  op: "NoOp"
  input: "^gradients/Conv2D_grad/Conv2DBackpropInput"
  input: "^gradients/Conv2D_grad/Conv2DBackpropFilter"
}
node {
  name: "gradients/Conv2D_grad/tuple/control_dependency"
  op: "Identity"
  input: "gradients/Conv2D_grad/Conv2DBackpropInput"
  input: "^gradients/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/Conv2D_grad/Conv2DBackpropInput"
      }
    }
  }
}
node {
  name: "gradients/Conv2D_grad/tuple/control_dependency_1"
  op: "Identity"
  input: "gradients/Conv2D_grad/Conv2DBackpropFilter"
  input: "^gradients/Conv2D_grad/tuple/group_deps"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@gradients/Conv2D_grad/Conv2DBackpropFilter"
      }
    }
  }
}
node {
  name: "beta1_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.899999976158
      }
    }
  }
}
node {
  name: "beta1_power"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "beta1_power/Assign"
  op: "Assign"
  input: "beta1_power"
  input: "beta1_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "beta1_power/read"
  op: "Identity"
  input: "beta1_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
}
node {
  name: "beta2_power/initial_value"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.999000012875
      }
    }
  }
}
node {
  name: "beta2_power"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "beta2_power/Assign"
  op: "Assign"
  input: "beta2_power"
  input: "beta2_power/initial_value"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "beta2_power/read"
  op: "Identity"
  input: "beta2_power"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
}
node {
  name: "zeros"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
          dim {
            size: 3
          }
          dim {
            size: 1
          }
          dim {
            size: 24
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_conv1/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 1
        }
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_conv1/Adam/Assign"
  op: "Assign"
  input: "W_conv1/Adam"
  input: "zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_conv1/Adam/read"
  op: "Identity"
  input: "W_conv1/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
}
node {
  name: "zeros_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
          dim {
            size: 3
          }
          dim {
            size: 1
          }
          dim {
            size: 24
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_conv1/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 1
        }
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_conv1/Adam_1/Assign"
  op: "Assign"
  input: "W_conv1/Adam_1"
  input: "zeros_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_conv1/Adam_1/read"
  op: "Identity"
  input: "W_conv1/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
}
node {
  name: "zeros_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 24
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_conv1/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_conv1/Adam/Assign"
  op: "Assign"
  input: "b_conv1/Adam"
  input: "zeros_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_conv1/Adam/read"
  op: "Identity"
  input: "b_conv1/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
}
node {
  name: "zeros_3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 24
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_conv1/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_conv1/Adam_1/Assign"
  op: "Assign"
  input: "b_conv1/Adam_1"
  input: "zeros_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_conv1/Adam_1/read"
  op: "Identity"
  input: "b_conv1/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
}
node {
  name: "zeros_4"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
          dim {
            size: 3
          }
          dim {
            size: 24
          }
          dim {
            size: 48
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_conv2/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 24
        }
        dim {
          size: 48
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_conv2/Adam/Assign"
  op: "Assign"
  input: "W_conv2/Adam"
  input: "zeros_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_conv2/Adam/read"
  op: "Identity"
  input: "W_conv2/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
}
node {
  name: "zeros_5"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
          dim {
            size: 3
          }
          dim {
            size: 24
          }
          dim {
            size: 48
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_conv2/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 24
        }
        dim {
          size: 48
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_conv2/Adam_1/Assign"
  op: "Assign"
  input: "W_conv2/Adam_1"
  input: "zeros_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_conv2/Adam_1/read"
  op: "Identity"
  input: "W_conv2/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
}
node {
  name: "zeros_6"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 48
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_conv2/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 48
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_conv2/Adam/Assign"
  op: "Assign"
  input: "b_conv2/Adam"
  input: "zeros_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_conv2/Adam/read"
  op: "Identity"
  input: "b_conv2/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
}
node {
  name: "zeros_7"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 48
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_conv2/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 48
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_conv2/Adam_1/Assign"
  op: "Assign"
  input: "b_conv2/Adam_1"
  input: "zeros_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_conv2/Adam_1/read"
  op: "Identity"
  input: "b_conv2/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
}
node {
  name: "zeros_8"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
          dim {
            size: 3
          }
          dim {
            size: 48
          }
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_conv3/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 48
        }
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_conv3/Adam/Assign"
  op: "Assign"
  input: "W_conv3/Adam"
  input: "zeros_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_conv3/Adam/read"
  op: "Identity"
  input: "W_conv3/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
}
node {
  name: "zeros_9"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 3
          }
          dim {
            size: 3
          }
          dim {
            size: 48
          }
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_conv3/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 48
        }
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_conv3/Adam_1/Assign"
  op: "Assign"
  input: "W_conv3/Adam_1"
  input: "zeros_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_conv3/Adam_1/read"
  op: "Identity"
  input: "W_conv3/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
}
node {
  name: "zeros_10"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_conv/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_conv/Adam/Assign"
  op: "Assign"
  input: "b_conv/Adam"
  input: "zeros_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_conv/Adam/read"
  op: "Identity"
  input: "b_conv/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
}
node {
  name: "zeros_11"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_conv/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_conv/Adam_1/Assign"
  op: "Assign"
  input: "b_conv/Adam_1"
  input: "zeros_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_conv/Adam_1/read"
  op: "Identity"
  input: "b_conv/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
}
node {
  name: "zeros_12"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 15552
          }
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_fc1/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 15552
        }
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_fc1/Adam/Assign"
  op: "Assign"
  input: "W_fc1/Adam"
  input: "zeros_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_fc1/Adam/read"
  op: "Identity"
  input: "W_fc1/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
}
node {
  name: "zeros_13"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 15552
          }
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_fc1/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 15552
        }
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_fc1/Adam_1/Assign"
  op: "Assign"
  input: "W_fc1/Adam_1"
  input: "zeros_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_fc1/Adam_1/read"
  op: "Identity"
  input: "W_fc1/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
}
node {
  name: "zeros_14"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_fc1/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_fc1/Adam/Assign"
  op: "Assign"
  input: "b_fc1/Adam"
  input: "zeros_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_fc1/Adam/read"
  op: "Identity"
  input: "b_fc1/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
}
node {
  name: "zeros_15"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_fc1/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_fc1/Adam_1/Assign"
  op: "Assign"
  input: "b_fc1/Adam_1"
  input: "zeros_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_fc1/Adam_1/read"
  op: "Identity"
  input: "b_fc1/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
}
node {
  name: "zeros_16"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
          dim {
            size: 2
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_fc2/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_fc2/Adam/Assign"
  op: "Assign"
  input: "W_fc2/Adam"
  input: "zeros_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_fc2/Adam/read"
  op: "Identity"
  input: "W_fc2/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
}
node {
  name: "zeros_17"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
          dim {
            size: 2
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "W_fc2/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "W_fc2/Adam_1/Assign"
  op: "Assign"
  input: "W_fc2/Adam_1"
  input: "zeros_17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "W_fc2/Adam_1/read"
  op: "Identity"
  input: "W_fc2/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
}
node {
  name: "zeros_18"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_fc2/Adam"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_fc2/Adam/Assign"
  op: "Assign"
  input: "b_fc2/Adam"
  input: "zeros_18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_fc2/Adam/read"
  op: "Identity"
  input: "b_fc2/Adam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
}
node {
  name: "zeros_19"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "b_fc2/Adam_1"
  op: "Variable"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 2
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "b_fc2/Adam_1/Assign"
  op: "Assign"
  input: "b_fc2/Adam_1"
  input: "zeros_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "b_fc2/Adam_1/read"
  op: "Identity"
  input: "b_fc2/Adam_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
}
node {
  name: "Adam/learning_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 3.99999989895e-05
      }
    }
  }
}
node {
  name: "Adam/beta1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.899999976158
      }
    }
  }
}
node {
  name: "Adam/beta2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.999000012875
      }
    }
  }
}
node {
  name: "Adam/epsilon"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 9.99999993923e-09
      }
    }
  }
}
node {
  name: "Adam/update_W_conv1/ApplyAdam"
  op: "ApplyAdam"
  input: "W_conv1"
  input: "W_conv1/Adam"
  input: "W_conv1/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/Conv2D_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_b_conv1/ApplyAdam"
  op: "ApplyAdam"
  input: "b_conv1"
  input: "b_conv1/Adam"
  input: "b_conv1/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/add_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_W_conv2/ApplyAdam"
  op: "ApplyAdam"
  input: "W_conv2"
  input: "W_conv2/Adam"
  input: "W_conv2/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/Conv2D_1_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_b_conv2/ApplyAdam"
  op: "ApplyAdam"
  input: "b_conv2"
  input: "b_conv2/Adam"
  input: "b_conv2/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/add_1_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_W_conv3/ApplyAdam"
  op: "ApplyAdam"
  input: "W_conv3"
  input: "W_conv3/Adam"
  input: "W_conv3/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/Conv2D_2_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_b_conv/ApplyAdam"
  op: "ApplyAdam"
  input: "b_conv"
  input: "b_conv/Adam"
  input: "b_conv/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/add_2_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_W_fc1/ApplyAdam"
  op: "ApplyAdam"
  input: "W_fc1"
  input: "W_fc1/Adam"
  input: "W_fc1/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/MatMul_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_b_fc1/ApplyAdam"
  op: "ApplyAdam"
  input: "b_fc1"
  input: "b_fc1/Adam"
  input: "b_fc1/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/add_3_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_W_fc2/ApplyAdam"
  op: "ApplyAdam"
  input: "W_fc2"
  input: "W_fc2/Adam"
  input: "W_fc2/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/MatMul_1_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/update_b_fc2/ApplyAdam"
  op: "ApplyAdam"
  input: "b_fc2"
  input: "b_fc2/Adam"
  input: "b_fc2/Adam_1"
  input: "beta1_power/read"
  input: "beta2_power/read"
  input: "Adam/learning_rate"
  input: "Adam/beta1"
  input: "Adam/beta2"
  input: "Adam/epsilon"
  input: "gradients/add_4_grad/tuple/control_dependency_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
}
node {
  name: "Adam/mul"
  op: "Mul"
  input: "beta1_power/read"
  input: "Adam/beta1"
  input: "^Adam/update_W_conv1/ApplyAdam"
  input: "^Adam/update_b_conv1/ApplyAdam"
  input: "^Adam/update_W_conv2/ApplyAdam"
  input: "^Adam/update_b_conv2/ApplyAdam"
  input: "^Adam/update_W_conv3/ApplyAdam"
  input: "^Adam/update_b_conv/ApplyAdam"
  input: "^Adam/update_W_fc1/ApplyAdam"
  input: "^Adam/update_b_fc1/ApplyAdam"
  input: "^Adam/update_W_fc2/ApplyAdam"
  input: "^Adam/update_b_fc2/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
}
node {
  name: "Adam/Assign"
  op: "Assign"
  input: "beta1_power"
  input: "Adam/mul"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Adam/mul_1"
  op: "Mul"
  input: "beta2_power/read"
  input: "Adam/beta2"
  input: "^Adam/update_W_conv1/ApplyAdam"
  input: "^Adam/update_b_conv1/ApplyAdam"
  input: "^Adam/update_W_conv2/ApplyAdam"
  input: "^Adam/update_b_conv2/ApplyAdam"
  input: "^Adam/update_W_conv3/ApplyAdam"
  input: "^Adam/update_b_conv/ApplyAdam"
  input: "^Adam/update_W_fc1/ApplyAdam"
  input: "^Adam/update_b_fc1/ApplyAdam"
  input: "^Adam/update_W_fc2/ApplyAdam"
  input: "^Adam/update_b_fc2/ApplyAdam"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
}
node {
  name: "Adam/Assign_1"
  op: "Assign"
  input: "beta2_power"
  input: "Adam/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: false
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "Adam"
  op: "NoOp"
  input: "^Adam/update_W_conv1/ApplyAdam"
  input: "^Adam/update_b_conv1/ApplyAdam"
  input: "^Adam/update_W_conv2/ApplyAdam"
  input: "^Adam/update_b_conv2/ApplyAdam"
  input: "^Adam/update_W_conv3/ApplyAdam"
  input: "^Adam/update_b_conv/ApplyAdam"
  input: "^Adam/update_W_fc1/ApplyAdam"
  input: "^Adam/update_b_fc1/ApplyAdam"
  input: "^Adam/update_W_fc2/ApplyAdam"
  input: "^Adam/update_b_fc2/ApplyAdam"
  input: "^Adam/Assign"
  input: "^Adam/Assign_1"
}
node {
  name: "ArgMax/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ArgMax"
  op: "ArgMax"
  input: "softmax"
  input: "ArgMax/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ArgMax_1/dimension"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ArgMax_1"
  op: "ArgMax"
  input: "Placeholder"
  input: "ArgMax_1/dimension"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "Equal"
  op: "Equal"
  input: "ArgMax"
  input: "ArgMax_1"
  attr {
    key: "T"
    value {
      type: DT_INT64
    }
  }
}
node {
  name: "Cast"
  op: "Cast"
  input: "Equal"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_BOOL
    }
  }
}
node {
  name: "Const_7"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "Mean_1"
  op: "Mean"
  input: "Cast"
  input: "Const_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
node {
  name: "init"
  op: "NoOp"
  input: "^W_conv1/Assign"
  input: "^b_conv1/Assign"
  input: "^W_conv2/Assign"
  input: "^b_conv2/Assign"
  input: "^W_conv3/Assign"
  input: "^b_conv/Assign"
  input: "^W_fc1/Assign"
  input: "^b_fc1/Assign"
  input: "^W_fc2/Assign"
  input: "^b_fc2/Assign"
  input: "^beta1_power/Assign"
  input: "^beta2_power/Assign"
  input: "^W_conv1/Adam/Assign"
  input: "^W_conv1/Adam_1/Assign"
  input: "^b_conv1/Adam/Assign"
  input: "^b_conv1/Adam_1/Assign"
  input: "^W_conv2/Adam/Assign"
  input: "^W_conv2/Adam_1/Assign"
  input: "^b_conv2/Adam/Assign"
  input: "^b_conv2/Adam_1/Assign"
  input: "^W_conv3/Adam/Assign"
  input: "^W_conv3/Adam_1/Assign"
  input: "^b_conv/Adam/Assign"
  input: "^b_conv/Adam_1/Assign"
  input: "^W_fc1/Adam/Assign"
  input: "^W_fc1/Adam_1/Assign"
  input: "^b_fc1/Adam/Assign"
  input: "^b_fc1/Adam_1/Assign"
  input: "^W_fc2/Adam/Assign"
  input: "^W_fc2/Adam_1/Assign"
  input: "^b_fc2/Adam/Assign"
  input: "^b_fc2/Adam_1/Assign"
}
node {
  name: "save/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "model"
      }
    }
  }
}
node {
  name: "save/save/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 32
          }
        }
        string_val: "W_conv1"
        string_val: "W_conv1/Adam"
        string_val: "W_conv1/Adam_1"
        string_val: "W_conv2"
        string_val: "W_conv2/Adam"
        string_val: "W_conv2/Adam_1"
        string_val: "W_conv3"
        string_val: "W_conv3/Adam"
        string_val: "W_conv3/Adam_1"
        string_val: "W_fc1"
        string_val: "W_fc1/Adam"
        string_val: "W_fc1/Adam_1"
        string_val: "W_fc2"
        string_val: "W_fc2/Adam"
        string_val: "W_fc2/Adam_1"
        string_val: "b_conv"
        string_val: "b_conv/Adam"
        string_val: "b_conv/Adam_1"
        string_val: "b_conv1"
        string_val: "b_conv1/Adam"
        string_val: "b_conv1/Adam_1"
        string_val: "b_conv2"
        string_val: "b_conv2/Adam"
        string_val: "b_conv2/Adam_1"
        string_val: "b_fc1"
        string_val: "b_fc1/Adam"
        string_val: "b_fc1/Adam_1"
        string_val: "b_fc2"
        string_val: "b_fc2/Adam"
        string_val: "b_fc2/Adam_1"
        string_val: "beta1_power"
        string_val: "beta2_power"
      }
    }
  }
}
node {
  name: "save/save/shapes_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 32
          }
        }
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
        string_val: ""
      }
    }
  }
}
node {
  name: "save/save"
  op: "SaveSlices"
  input: "save/Const"
  input: "save/save/tensor_names"
  input: "save/save/shapes_and_slices"
  input: "W_conv1"
  input: "W_conv1/Adam"
  input: "W_conv1/Adam_1"
  input: "W_conv2"
  input: "W_conv2/Adam"
  input: "W_conv2/Adam_1"
  input: "W_conv3"
  input: "W_conv3/Adam"
  input: "W_conv3/Adam_1"
  input: "W_fc1"
  input: "W_fc1/Adam"
  input: "W_fc1/Adam_1"
  input: "W_fc2"
  input: "W_fc2/Adam"
  input: "W_fc2/Adam_1"
  input: "b_conv"
  input: "b_conv/Adam"
  input: "b_conv/Adam_1"
  input: "b_conv1"
  input: "b_conv1/Adam"
  input: "b_conv1/Adam_1"
  input: "b_conv2"
  input: "b_conv2/Adam"
  input: "b_conv2/Adam_1"
  input: "b_fc1"
  input: "b_fc1/Adam"
  input: "b_fc1/Adam_1"
  input: "b_fc2"
  input: "b_fc2/Adam"
  input: "b_fc2/Adam_1"
  input: "beta1_power"
  input: "beta2_power"
  attr {
    key: "T"
    value {
      list {
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/control_dependency"
  op: "Identity"
  input: "save/Const"
  input: "^save/save"
  attr {
    key: "T"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@save/Const"
      }
    }
  }
}
node {
  name: "save/RestoreV2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_conv1"
      }
    }
  }
}
node {
  name: "save/RestoreV2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2/tensor_names"
  input: "save/RestoreV2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign"
  op: "Assign"
  input: "W_conv1"
  input: "save/RestoreV2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_1/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_conv1/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_1/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_1"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_1/tensor_names"
  input: "save/RestoreV2_1/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_1"
  op: "Assign"
  input: "W_conv1/Adam"
  input: "save/RestoreV2_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_2/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_conv1/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_2/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_2"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_2/tensor_names"
  input: "save/RestoreV2_2/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_2"
  op: "Assign"
  input: "W_conv1/Adam_1"
  input: "save/RestoreV2_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_3/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_conv2"
      }
    }
  }
}
node {
  name: "save/RestoreV2_3/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_3"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_3/tensor_names"
  input: "save/RestoreV2_3/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_3"
  op: "Assign"
  input: "W_conv2"
  input: "save/RestoreV2_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_4/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_conv2/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_4/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_4"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_4/tensor_names"
  input: "save/RestoreV2_4/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_4"
  op: "Assign"
  input: "W_conv2/Adam"
  input: "save/RestoreV2_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_5/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_conv2/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_5/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_5"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_5/tensor_names"
  input: "save/RestoreV2_5/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_5"
  op: "Assign"
  input: "W_conv2/Adam_1"
  input: "save/RestoreV2_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_6/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_conv3"
      }
    }
  }
}
node {
  name: "save/RestoreV2_6/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_6"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_6/tensor_names"
  input: "save/RestoreV2_6/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_6"
  op: "Assign"
  input: "W_conv3"
  input: "save/RestoreV2_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_7/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_conv3/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_7/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_7"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_7/tensor_names"
  input: "save/RestoreV2_7/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_7"
  op: "Assign"
  input: "W_conv3/Adam"
  input: "save/RestoreV2_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_8/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_conv3/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_8/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_8"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_8/tensor_names"
  input: "save/RestoreV2_8/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_8"
  op: "Assign"
  input: "W_conv3/Adam_1"
  input: "save/RestoreV2_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv3"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_9/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_fc1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_9/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_9"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_9/tensor_names"
  input: "save/RestoreV2_9/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_9"
  op: "Assign"
  input: "W_fc1"
  input: "save/RestoreV2_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_10/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_fc1/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_10/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_10"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_10/tensor_names"
  input: "save/RestoreV2_10/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_10"
  op: "Assign"
  input: "W_fc1/Adam"
  input: "save/RestoreV2_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_11/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_fc1/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_11/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_11"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_11/tensor_names"
  input: "save/RestoreV2_11/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_11"
  op: "Assign"
  input: "W_fc1/Adam_1"
  input: "save/RestoreV2_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_12/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_fc2"
      }
    }
  }
}
node {
  name: "save/RestoreV2_12/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_12"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_12/tensor_names"
  input: "save/RestoreV2_12/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_12"
  op: "Assign"
  input: "W_fc2"
  input: "save/RestoreV2_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_13/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_fc2/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_13/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_13"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_13/tensor_names"
  input: "save/RestoreV2_13/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_13"
  op: "Assign"
  input: "W_fc2/Adam"
  input: "save/RestoreV2_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_14/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "W_fc2/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_14/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_14"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_14/tensor_names"
  input: "save/RestoreV2_14/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_14"
  op: "Assign"
  input: "W_fc2/Adam_1"
  input: "save/RestoreV2_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_15/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_conv"
      }
    }
  }
}
node {
  name: "save/RestoreV2_15/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_15"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_15/tensor_names"
  input: "save/RestoreV2_15/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_15"
  op: "Assign"
  input: "b_conv"
  input: "save/RestoreV2_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_16/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_conv/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_16/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_16"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_16/tensor_names"
  input: "save/RestoreV2_16/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_16"
  op: "Assign"
  input: "b_conv/Adam"
  input: "save/RestoreV2_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_17/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_conv/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_17/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_17"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_17/tensor_names"
  input: "save/RestoreV2_17/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_17"
  op: "Assign"
  input: "b_conv/Adam_1"
  input: "save/RestoreV2_17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_18/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_conv1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_18/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_18"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_18/tensor_names"
  input: "save/RestoreV2_18/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_18"
  op: "Assign"
  input: "b_conv1"
  input: "save/RestoreV2_18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_19/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_conv1/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_19/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_19"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_19/tensor_names"
  input: "save/RestoreV2_19/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_19"
  op: "Assign"
  input: "b_conv1/Adam"
  input: "save/RestoreV2_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_20/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_conv1/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_20/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_20"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_20/tensor_names"
  input: "save/RestoreV2_20/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_20"
  op: "Assign"
  input: "b_conv1/Adam_1"
  input: "save/RestoreV2_20"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_21/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_conv2"
      }
    }
  }
}
node {
  name: "save/RestoreV2_21/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_21"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_21/tensor_names"
  input: "save/RestoreV2_21/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_21"
  op: "Assign"
  input: "b_conv2"
  input: "save/RestoreV2_21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_22/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_conv2/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_22/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_22"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_22/tensor_names"
  input: "save/RestoreV2_22/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_22"
  op: "Assign"
  input: "b_conv2/Adam"
  input: "save/RestoreV2_22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_23/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_conv2/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_23/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_23"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_23/tensor_names"
  input: "save/RestoreV2_23/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_23"
  op: "Assign"
  input: "b_conv2/Adam_1"
  input: "save/RestoreV2_23"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_conv2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_24/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_fc1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_24/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_24"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_24/tensor_names"
  input: "save/RestoreV2_24/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_24"
  op: "Assign"
  input: "b_fc1"
  input: "save/RestoreV2_24"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_25/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_fc1/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_25/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_25"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_25/tensor_names"
  input: "save/RestoreV2_25/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_25"
  op: "Assign"
  input: "b_fc1/Adam"
  input: "save/RestoreV2_25"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_26/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_fc1/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_26/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_26"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_26/tensor_names"
  input: "save/RestoreV2_26/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_26"
  op: "Assign"
  input: "b_fc1/Adam_1"
  input: "save/RestoreV2_26"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_27/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_fc2"
      }
    }
  }
}
node {
  name: "save/RestoreV2_27/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_27"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_27/tensor_names"
  input: "save/RestoreV2_27/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_27"
  op: "Assign"
  input: "b_fc2"
  input: "save/RestoreV2_27"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_28/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_fc2/Adam"
      }
    }
  }
}
node {
  name: "save/RestoreV2_28/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_28"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_28/tensor_names"
  input: "save/RestoreV2_28/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_28"
  op: "Assign"
  input: "b_fc2/Adam"
  input: "save/RestoreV2_28"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_29/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "b_fc2/Adam_1"
      }
    }
  }
}
node {
  name: "save/RestoreV2_29/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_29"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_29/tensor_names"
  input: "save/RestoreV2_29/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_29"
  op: "Assign"
  input: "b_fc2/Adam_1"
  input: "save/RestoreV2_29"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@b_fc2"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_30/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "beta1_power"
      }
    }
  }
}
node {
  name: "save/RestoreV2_30/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_30"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_30/tensor_names"
  input: "save/RestoreV2_30/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_30"
  op: "Assign"
  input: "beta1_power"
  input: "save/RestoreV2_30"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/RestoreV2_31/tensor_names"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: "beta2_power"
      }
    }
  }
}
node {
  name: "save/RestoreV2_31/shape_and_slices"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 1
          }
        }
        string_val: ""
      }
    }
  }
}
node {
  name: "save/RestoreV2_31"
  op: "RestoreV2"
  input: "save/Const"
  input: "save/RestoreV2_31/tensor_names"
  input: "save/RestoreV2_31/shape_and_slices"
  attr {
    key: "dtypes"
    value {
      list {
        type: DT_FLOAT
      }
    }
  }
}
node {
  name: "save/Assign_31"
  op: "Assign"
  input: "beta2_power"
  input: "save/RestoreV2_31"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@W_conv1"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "save/restore_all"
  op: "NoOp"
  input: "^save/Assign"
  input: "^save/Assign_1"
  input: "^save/Assign_2"
  input: "^save/Assign_3"
  input: "^save/Assign_4"
  input: "^save/Assign_5"
  input: "^save/Assign_6"
  input: "^save/Assign_7"
  input: "^save/Assign_8"
  input: "^save/Assign_9"
  input: "^save/Assign_10"
  input: "^save/Assign_11"
  input: "^save/Assign_12"
  input: "^save/Assign_13"
  input: "^save/Assign_14"
  input: "^save/Assign_15"
  input: "^save/Assign_16"
  input: "^save/Assign_17"
  input: "^save/Assign_18"
  input: "^save/Assign_19"
  input: "^save/Assign_20"
  input: "^save/Assign_21"
  input: "^save/Assign_22"
  input: "^save/Assign_23"
  input: "^save/Assign_24"
  input: "^save/Assign_25"
  input: "^save/Assign_26"
  input: "^save/Assign_27"
  input: "^save/Assign_28"
  input: "^save/Assign_29"
  input: "^save/Assign_30"
  input: "^save/Assign_31"
}
versions {
  producer: 15
}
