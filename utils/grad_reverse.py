from torch.autograd import Function

class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
	return GradReverse.apply(x, alpha)
	