"""PCGrad multi-task gradient surgery.

Extracted verbatim from the original OmniText notebooks.
"""
import torch


class PCGrad:
    def __init__(self, optimizer):
        """
        PCGrad wrapper for an optimizer (e.g., Adam, SGD).
        Args:
            optimizer: A PyTorch optimizer instance.
        """
        self.optimizer = optimizer

    def step(self, task_gradients):
        """
        Perform a PCGrad step.
        Args:
            task_gradients: List of gradients for each task. Each element is a list of parameter gradients.
        """
        # Initialize the final gradient list with zeros
        final_gradients = [torch.zeros_like(param) for param in task_gradients[0]]

        # Apply PCGrad for each task
        for i, grad_i in enumerate(task_gradients):
            # Copy the gradient to modify
            projected_grad = grad_i.copy()

            for j, grad_j in enumerate(task_gradients):
                if i != j:
                    # Compute the dot product of gradients
                    dot_product = sum(torch.dot(g_i.view(-1), g_j.view(-1)) for g_i, g_j in zip(grad_i, grad_j))

                    # If gradients conflict (dot product < 0), project grad_i onto the normal plane of grad_j
                    if dot_product < 0:
                        norm_grad_j = sum(torch.norm(g_j.view(-1))**2 for g_j in grad_j)
                        if norm_grad_j > 0:  # Avoid division by zero
                            projection = [(dot_product / norm_grad_j) * g_j for g_j in grad_j]
                            projected_grad = [g_i - proj for g_i, proj in zip(projected_grad, projection)]

            # Add the modified gradient to the final gradient
            final_gradients = [fg + pg for fg, pg in zip(final_gradients, projected_grad)]

        # Assign the final gradients to the optimizer's parameters
        for param, grad in zip(self.optimizer.param_groups[0]['params'], final_gradients):
            param.grad = grad

        # Perform the optimizer step
        self.optimizer.step()

    def zero_grad(self):
        """Clear gradients in the optimizer."""
        self.optimizer.zero_grad()
