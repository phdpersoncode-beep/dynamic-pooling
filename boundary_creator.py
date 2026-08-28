import torch


class BoundaryCreator():
    def __init__(self, whitespace_id):
        self.whitespace_id = whitespace_id

    def get_boundaries(self, txt=None, tensor=None):
        """
            Function that generates boundaries for given tensor of data

            Attributes:
                data - (torch.LongTensor) - [seq_len x batch_size]

            Returns:
                boundaries - (torch.BoolTensor) - [batch_size x seq_len]
        """
        assert tensor is not None
        data = tensor

        data = data.transpose(0, 1)  # batch_size x seq_len
        boundaries = torch.zeros_like(data, dtype=torch.bool)

        boundaries |= (data == self.whitespace_id)

        return boundaries
