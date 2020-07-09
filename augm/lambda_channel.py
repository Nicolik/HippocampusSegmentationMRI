import torch
import torchio
from torchio import TYPE, DATA, Subject
from torchio.transforms import Lambda


class LambdaChannel(Lambda):
    def apply_transform(self, sample: Subject) -> dict:
        for image in sample.get_images(intensity_only=False):

            image_type = image[TYPE]
            if self.types_to_apply is not None:
                if image_type not in self.types_to_apply:
                    continue

            function_arg = image[DATA]
            result = self.function(function_arg)
            if not isinstance(result, torch.Tensor):
                message = (
                    'The returned value from the callable argument must be'
                    f' of type {torch.Tensor}, not {type(result)}'
                )
                raise ValueError(message)
            if result.dtype != torch.float32:
                message = (
                    'The data type of the returned value must be'
                    f' of type {torch.float32}, not {result.dtype}'
                )
                raise ValueError(message)
            if result.ndim != function_arg.ndim:
                message = (
                    'The number of dimensions of the returned value must'
                    f' be {function_arg.ndim}, not {result.ndim}'
                )
                raise ValueError(message)
            image[DATA] = result
        return sample
