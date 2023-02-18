import torch
from torch.autograd import gradcheck
from torch.autograd import Function


class MinMaxFunction(Function):

    @staticmethod
    def forward(ctx, input, R):
        ctx.save_for_backward(input)
        
        sorted, indices = torch.sort(input, dim=2, descending=True)
        ctx.indices_max = indices.narrow(2, 0, R)
        output = sorted.narrow(2, 0, R)

        ctx.indices_min = indices.narrow(2, -R, R)
        output = torch.cat((output, sorted.narrow(2, -R, R)), dim=2)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        R = ctx.indices_max.size(2)

        grad_output_max = grad_output.narrow(2, 0, R)
        grad_output_min = grad_output.narrow(2, R, R)

        grad_input = grad_output.new().resize_as_(input).fill_(0)

        grad_input.scatter_(2, ctx.indices_max, grad_output_max)
        grad_input.scatter_(2, ctx.indices_min, grad_output_min)

        return grad_input, None

def test_minmaxfunction():
    torch.manual_seed(0)
    input = torch.randn(4, 4, 6, dtype=torch.double, requires_grad=True)
    R = torch.Tensor([2])
    test = gradcheck(MinMaxFunction.apply, (input.double(), R.double()), eps=1e-6, atol=1e-4)
    assert test, "Test failed"

if __name__ == '__main__':
    test_minmaxfunction()
