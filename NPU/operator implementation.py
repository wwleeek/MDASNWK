import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check
from te.utils import shape_util
from tbe.common.utils import shape_util
from te.utils.error_manager import error_manager_vector
from tbe import dsl


@register_op_compute("l2_distance")
def l2_distance_compute(x, y, z, kernel_name="l2_distance"):

    data_sub = dsl.vsub(x, y)
    data_mul = dsl.vmul(data_sub, data_sub)
    res = dsl.reduce_sum(data_mul, axis = 1)

    return res

@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def l2_distance(x, y, z, kernel_name="l2_distance"):
    #初始化输入tensor，为输入tensor进行占位
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    data_y = tvm.placeholder(y.get("shape"), dtype=y.get("dtype"), name="data_y")
    res = l2_distance_compute(data_x, data_y, z, kernel_name)
    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)
    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_y, res]}
    tbe.build(schedule, config)
    # 算子调用
if __name__ == '__main__':
    input_output_dict = {"shape":[1000, 128], "format": "ND", "ori_shape":[1000, 128],
                         "ori_format": "ND","dtype":"float16"}
    l2_distance(input_output_dict, input_output_dict, input_output_dict, kernel_name="l2_distance")